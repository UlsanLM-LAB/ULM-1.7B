"""Run UlsanBench comparison with explicit inputs and recorded execution provenance."""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT / "src", ROOT / "scripts"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from evaluate_ulsanbench_v2 import SYSTEM, Embedder, save, score_rows  # noqa: E402
from ulm.evaluation.comparison import (  # noqa: E402
    DECODING_MODES,
    DTYPES,
    TASKS,
    build_request,
    hardware_info,
    load_benchmark,
    utc_timestamp,
    validate_name,
)


def format_chat_prompt(tokenizer: AutoTokenizer, prompt_text: str) -> str:
    """Format prompt with the official model chat template."""
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt_text}]
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
    except TypeError:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def clean_output(raw_text: str) -> str:
    return raw_text.split("</think>")[-1].strip()


def generated_token_count(tokens, eos_token_id) -> int:
    """Count through the first EOS (inclusive), excluding subsequent batch padding."""
    ids = tokens.tolist() if hasattr(tokens, "tolist") else list(tokens)
    eos_ids = set(eos_token_id if isinstance(eos_token_id, (list, tuple)) else [eos_token_id])
    for index, token_id in enumerate(ids):
        if token_id in eos_ids:
            return index + 1
    return len(ids)


def _revision_kwargs(identity: dict | None) -> dict:
    if identity and identity["kind"] == "hub":
        return {"revision": identity["resolved_commit"]}
    return {}


@torch.inference_mode()
def run_model_inference(
    model_name_or_path: str,
    adapter_path: str | None,
    rows: list[dict],
    device: str = "cuda:0",
    dtype: torch.dtype = torch.bfloat16,
    batch_size: int = 8,
    decoding_mode: str = "neutral",
    assets: dict | None = None,
) -> tuple[list[str], dict]:
    """Generate deterministically; throughput counts generated tokens, including EOS."""
    if batch_size < 1 or decoding_mode not in DECODING_MODES:
        raise ValueError("Invalid batch size or decoding mode")
    if any(row.get("task") not in TASKS for row in rows):
        raise ValueError("Unsupported UlsanBench task")
    assets = assets or {}
    model_source = assets.get("model", {}).get("path", model_name_or_path)
    adapter_source = (assets.get("adapter") or {}).get("path", adapter_path)
    is_cuda = torch.device(device).type == "cuda"
    if is_cuda and not torch.cuda.is_available():
        raise ValueError(f"CUDA is unavailable; choose --device cpu instead of {device}")
    print(f"Loading tokenizer/model: {model_name_or_path} (adapter={adapter_path})", flush=True)
    t0_load = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(
        model_source, trust_remote_code=True, **_revision_kwargs(assets.get("model"))
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.bos_token
    if tokenizer.pad_token_id is None:
        raise ValueError("Tokenizer has no pad, EOS or BOS token for batched generation")
    if is_cuda:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    model = AutoModelForCausalLM.from_pretrained(
        model_source,
        dtype=dtype,
        device_map=device,
        attn_implementation="sdpa",
        trust_remote_code=True,
        **_revision_kwargs(assets.get("model")),
    ).eval()
    if adapter_path:
        model = PeftModel.from_pretrained(
            model, adapter_source, **_revision_kwargs(assets.get("adapter"))
        ).eval()
    eos = tokenizer.eos_token_id
    if eos is None:
        eos = model.generation_config.eos_token_id
    if is_cuda:
        torch.cuda.synchronize(device)
    load_time = time.perf_counter() - t0_load
    outputs = [None] * len(rows)
    total_tokens_generated = 0
    t0_gen = time.perf_counter()
    try:
        for task in TASKS:
            indices = [i for i, row in enumerate(rows) if row["task"] == task]
            limit = 16 if task == "identification" else 96
            extra = (
                {"repetition_penalty": 1.10, "no_repeat_ngram_size": 3}
                if decoding_mode == "context_guard" and task == "context"
                else {}
            )
            for start in range(0, len(indices), batch_size):
                batch_idx = indices[start : start + batch_size]
                prompts = [format_chat_prompt(tokenizer, rows[i]["prompt"]) for i in batch_idx]
                enc = tokenizer(
                    prompts, padding=True, truncation=True, max_length=1024, return_tensors="pt"
                ).to(device)
                gen = model.generate(
                    **enc,
                    max_new_tokens=limit,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=eos,
                    **extra,
                )
                if len(gen) != len(batch_idx):
                    raise RuntimeError("Model returned a different number of generated sequences")
                for i, seq in zip(batch_idx, gen, strict=True):
                    new_tokens = seq[enc.input_ids.shape[1] :]
                    count = generated_token_count(new_tokens, eos)
                    total_tokens_generated += count
                    outputs[i] = clean_output(
                        tokenizer.decode(new_tokens[:count], skip_special_tokens=True)
                    )
            if indices:
                print(f"Evaluated task '{task}': {len(indices)} items", flush=True)
        if is_cuda:
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - t0_gen
        runtime = {
            "load_time_seconds": round(load_time, 2),
            "total_inference_time_seconds": round(elapsed, 2),
            "avg_latency_ms_per_item": round(elapsed / max(1, len(rows)) * 1000, 2),
            "latency_definition": "batch inference wall time divided by item count; not TTFT",
            "total_tokens_generated": total_tokens_generated,
            "token_count_definition": "first EOS inclusive; subsequent padding excluded",
            "tokens_per_second": round(total_tokens_generated / max(0.01, elapsed), 2),
            "peak_vram_gib": round(torch.cuda.max_memory_allocated(device) / 1024**3, 2)
            if is_cuda
            else None,
            "decoding_mode": decoding_mode,
            "batch_size": batch_size,
            "device": device,
            "dtype": str(dtype).removeprefix("torch."),
        }
        return outputs, runtime
    finally:
        del model, tokenizer
        if is_cuda:
            torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description="UlsanBench v2 model comparison")
    parser.add_argument("--model", required=True, help="Hugging Face model ID or local directory")
    parser.add_argument("--adapter", help="PEFT adapter directory or Hugging Face ID")
    parser.add_argument("--revision", help="Model revision; resolved to an immutable commit")
    parser.add_argument("--adapter-revision", help="Adapter revision")
    parser.add_argument("--name", required=True, help="Safe identifier for output filenames")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dataset", required=True, help="UlsanBench JSONL input")
    parser.add_argument("--raw-dir", help="Defaults to <output-dir>/raw")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16", choices=DTYPES)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--decoding-mode", default="neutral", choices=DECODING_MODES)
    parser.add_argument("--subset", type=int, default=0, help="0 = full; deterministic task sample")
    args = parser.parse_args()
    try:
        validate_name(args.name)
        rows, dataset = load_benchmark(args.dataset, args.subset)
        cfg = {
            "id": args.name,
            "model": args.model,
            "adapter": args.adapter,
            "revision": args.revision,
            "adapter_revision": args.adapter_revision,
            "decoding_mode": args.decoding_mode,
        }
        request = build_request(
            ROOT,
            cfg,
            dataset,
            device=args.device,
            dtype=args.dtype,
            batch_size=args.batch_size,
            subset=args.subset,
        )
    except ValueError as exc:
        parser.error(str(exc))
    started_at = utc_timestamp()
    hardware = hardware_info(args.device)
    print(f"Running benchmark: {len(rows)} items", flush=True)
    outputs, runtime = run_model_inference(
        args.model,
        args.adapter,
        rows,
        device=args.device,
        dtype=getattr(torch, args.dtype),
        batch_size=args.batch_size,
        decoding_mode=args.decoding_mode,
        assets=request["assets"],
    )
    output_dir = Path(args.output_dir)
    raw_dir = Path(args.raw_dir) if args.raw_dir else output_dir / "raw"
    print("Scoring outputs with unchanged UlsanBench v2 rubric", flush=True)
    metric = Embedder(
        device=args.device, revision=request["assets"]["encoder"].get("resolved_commit")
    )
    summary = score_rows(rows, outputs, output_dir, args.model, args.adapter, metric=metric)
    summary["display_name"] = args.name
    summary["runtime"] = runtime
    summary["config"]["decoding_mode"] = args.decoding_mode
    if args.decoding_mode == "context_guard":
        summary["config"]["context_task_extra"] = {
            "repetition_penalty": 1.10,
            "no_repeat_ngram_size": 3,
        }
    summary["provenance"] = {
        "version": 1,
        "started_at": started_at,
        "completed_at": utc_timestamp(),
        "request": request,
        "hardware": hardware,
    }
    save(output_dir / "summary.json", summary)
    raw_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(output_dir / "predictions.jsonl", raw_dir / f"{args.name}.jsonl")
    print(f"Saved summary: {output_dir / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
