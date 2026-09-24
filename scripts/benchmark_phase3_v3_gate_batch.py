"""Measure batched generation without changing the scored gate evaluator."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

from evaluate_preservation import format_chat_prompt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/home/ubuntu/models/Qwen3.8-4B-Distill")
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--benchmark", default="reports/phase3-v3/gate-60.jsonl")
    parser.add_argument("--output", default="reports/phase3-v3/gate-batch-trials.json")
    parser.add_argument("--count", type=int, default=16)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="cuda:0", attn_implementation="sdpa"
    )
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()
    items = [json.loads(line) for line in Path(args.benchmark).read_text().splitlines()]
    prompts = [format_chat_prompt(tokenizer, item) for item in items[: args.count]]

    results = []
    for batch_size in (1, 4, 8, 16):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        start = time.monotonic()
        generated_tokens = 0
        try:
            for offset in range(0, len(prompts), batch_size):
                batch = tokenizer(
                    prompts[offset : offset + batch_size],
                    return_tensors="pt",
                    padding=True,
                ).to("cuda:0")
                prompt_width = batch.input_ids.shape[1]
                set_seed(42)
                with torch.inference_mode():
                    output = model.generate(
                        **batch,
                        max_new_tokens=256,
                        temperature=0.7,
                        top_p=0.9,
                        top_k=20,
                        repetition_penalty=1.1,
                        do_sample=True,
                    )
                generated_tokens += output.shape[0] * (output.shape[1] - prompt_width)
            torch.cuda.synchronize()
            elapsed = time.monotonic() - start
            row = {
                "batch_size": batch_size,
                "prompts": len(prompts),
                "runtime_s": round(elapsed, 3),
                "prompts_per_second": round(len(prompts) / elapsed, 3),
                "generated_tokens_per_second": round(generated_tokens / elapsed, 3),
                "peak_vram_gib": round(torch.cuda.max_memory_reserved() / 2**30, 3),
            }
        except torch.cuda.OutOfMemoryError:
            row = {"batch_size": batch_size, "status": "OOM"}
        print(json.dumps(row), flush=True)
        results.append(row)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
