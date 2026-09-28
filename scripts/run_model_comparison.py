"""Universal benchmark runner for UlsanBench v2 model comparison.

Runs evaluation on the 500-item UlsanBench v2 dataset with deterministic decoding,
official model chat templates, runtime benchmarking (VRAM, latency, tokens/sec),
and unchanged metric rubrics.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from peft import PeftModel
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

# Adjust path so evaluate_ulsanbench_v2 can be imported
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from evaluate_ulsanbench_v2 import (
    CLASSES,
    MARKERS,
    REPETITION,
    SYSTEM,
    Embedder,
    jsonl,
    save,
    score_rows,
)


def format_chat_prompt(tokenizer: AutoTokenizer, prompt_text: str) -> str:
    """Format prompt with the official model chat template."""
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": prompt_text},
    ]
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )


def clean_output(raw_text: str) -> str:
    """Strip special tokens and thinking traces."""
    cleaned = raw_text.split("</think>")[-1].strip()
    return cleaned


@torch.inference_mode()
def run_model_inference(
    model_name_or_path: str,
    adapter_path: str | None,
    rows: list[dict],
    device: str = "cuda:0",
    dtype: torch.dtype = torch.bfloat16,
    batch_size: int = 8,
    decoding_mode: str = "neutral",
) -> tuple[list[str], dict]:
    """Execute batch inference and measure runtime statistics."""
    print(f"Loading tokenizer for: {model_name_or_path} ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.bos_token

    print(f"Loading model: {model_name_or_path} (adapter={adapter_path}) ...", flush=True)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    t0_load = time.time()

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        dtype=dtype,
        device_map=device,
        attn_implementation="sdpa",
        trust_remote_code=True,
    ).eval()

    if adapter_path:
        print(f"Attaching PEFT adapter: {adapter_path} ...", flush=True)
        model = PeftModel.from_pretrained(model, adapter_path).eval()

    load_time = time.time() - t0_load
    print(f"Model loaded in {load_time:.2f}s", flush=True)

    outputs = [None] * len(rows)
    total_tokens_generated = 0
    t0_gen = time.time()
    task_order = ("comprehension", "generation", "identification", "grammar", "context")

    for task in task_order:
        indices = [i for i, r in enumerate(rows) if r["task"] == task]
        if not indices:
            continue
        limit = 16 if task == "identification" else 96

        extra = {}
        if decoding_mode == "deployed" and task == "context":
            extra = {"repetition_penalty": 1.10, "no_repeat_ngram_size": 3}

        for start in range(0, len(indices), batch_size):
            batch_idx = indices[start : start + batch_size]
            prompt_texts = [
                format_chat_prompt(tokenizer, rows[i]["prompt"]) for i in batch_idx
            ]
            enc = tokenizer(
                prompt_texts,
                padding=True,
                truncation=True,
                max_length=1024,
                return_tensors="pt",
            ).to(device)

            gen = model.generate(
                **enc,
                max_new_tokens=limit,
                do_sample=False,
                temperature=0.0,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                **extra,
            )

            prompt_len = enc.input_ids.shape[1]
            for i, seq in zip(batch_idx, gen):
                new_tokens = seq[prompt_len:]
                total_tokens_generated += len(new_tokens)
                raw = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
                outputs[i] = clean_output(raw)

        print(f"Evaluated task '{task}': {len(indices)} items", flush=True)

    total_gen_time = time.time() - t0_gen
    peak_vram = torch.cuda.max_memory_allocated(device) / (1024**3)

    runtime_stats = {
        "load_time_seconds": round(load_time, 2),
        "total_inference_time_seconds": round(total_gen_time, 2),
        "avg_latency_ms_per_item": round((total_gen_time / max(1, len(rows))) * 1000, 2),
        "total_tokens_generated": total_tokens_generated,
        "tokens_per_second": round(total_tokens_generated / max(0.01, total_gen_time), 2),
        "peak_vram_gib": round(peak_vram, 2),
        "decoding_mode": decoding_mode,
        "batch_size": batch_size,
    }

    del model
    del tokenizer
    torch.cuda.empty_cache()
    return outputs, runtime_stats


def main():
    parser = argparse.ArgumentParser(description="Universal UlsanBench v2 model comparison runner")
    parser.add_argument("--model", required=True, help="Hugging Face model ID or local directory")
    parser.add_argument("--adapter", default=None, help="PEFT adapter directory if any")
    parser.add_argument("--name", required=True, help="Short identifier name for model")
    parser.add_argument("--output-dir", required=True, help="Directory to store outputs and summaries")
    parser.add_argument("--dataset", default=str(ROOT / "data/ulsanbench_v1/benchmark.jsonl"), help="Path to benchmark.jsonl")
    parser.add_argument("--raw-dir", default=str(ROOT / "reports/model-comparison/raw"), help="Path to store raw prediction jsonl")
    parser.add_argument("--device", default="cuda:0", help="CUDA device")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"], help="Model dtype")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for generation")
    parser.add_argument("--decoding-mode", default="neutral", choices=["neutral", "deployed"], help="neutral or deployed decoding")
    parser.add_argument("--subset", type=int, default=0, help="Subset size for smoke testing (0 = full benchmark)")

    args = parser.parse_args()

    torch_dtype = getattr(torch, args.dtype)
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found at: {dataset_path}")

    rows = jsonl(dataset_path)
    if args.subset > 0:
        # Balanced stratified sample across tasks
        by_task = defaultdict(list)
        for r in rows:
            by_task[r["task"]].append(r)
        sampled = []
        for task, t_rows in by_task.items():
            count = max(1, round(args.subset * len(t_rows) / len(rows)))
            sampled.extend(t_rows[:count])
        rows = sampled
        print(f"Running subset of {len(rows)} items across tasks: {dict((t, sum(1 for r in rows if r['task']==t)) for t in by_task)}")
    else:
        print(f"Running full benchmark: {len(rows)} items")

    outputs, runtime = run_model_inference(
        model_name_or_path=args.model,
        adapter_path=args.adapter,
        rows=rows,
        device=args.device,
        dtype=torch_dtype,
        batch_size=args.batch_size,
        decoding_mode=args.decoding_mode,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = Path(args.raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)

    print("Scoring outputs using UlsanBench v2 evaluator ...", flush=True)
    embedder = Embedder(device=args.device)
    summary = score_rows(rows, outputs, output_dir, args.model, args.adapter, metric=embedder)

    # Attach runtime info and identifier
    summary["display_name"] = args.name
    summary["runtime"] = runtime
    save(output_dir / "summary.json", summary)

    # Copy predictions to raw_dir
    pred_path = output_dir / "predictions.jsonl"
    target_raw_path = raw_dir / f"{args.name}.jsonl"
    if pred_path.exists():
        target_raw_path.write_text(pred_path.read_text(encoding="utf-8"), encoding="utf-8")
        print(f"Saved raw predictions to: {target_raw_path}", flush=True)

    print("\n" + "=" * 60)
    print(f"BENCHMARK RESULTS: {args.name}")
    print(f"Model: {args.model} | Adapter: {args.adapter}")
    print(f"Runtime: {runtime['total_inference_time_seconds']}s | Tok/s: {runtime['tokens_per_second']} | VRAM: {runtime['peak_vram_gib']} GiB")
    print("-" * 60)
    for task, m in summary["metrics"].items():
        sem = m.get("semantic_similarity", 0.0)
        dia = m.get("dialectness_proxy", 0.0)
        acc = m.get("accuracy", 0.0)
        rep = m.get("repetition_count", 0)
        mal = m.get("malformed_count", 0)
        line = f"  {task:<15}: sem={sem:.4f}"
        if dia is not None and dia > 0:
            line += f" | dialect={dia:.4f}"
        if acc is not None and acc > 0:
            line += f" | accuracy={acc:.4f}"
        line += f" | rep={rep} | mal={mal}"
        print(line)
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
