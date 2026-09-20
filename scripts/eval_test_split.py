#!/usr/bin/env python3
"""Evaluate Phase 2 adapter on test split."""

from __future__ import annotations

import json
import math
from pathlib import Path
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer
from peft import PeftModel
from ulm.training.sft import _prepare_dataset

def main():
    base_model_name = "outputs/ulm-1.7b-stage1-merged"
    adapter_path = "outputs/ulm-1.7b-phase2-l40s"
    test_file = "data/ulsan_dialect_phase2/test.jsonl"

    print("=== Evaluating Phase 2 on Test Split ===")
    tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(base_model, adapter_path)

    raw_dataset = load_dataset("json", data_files={"test": test_file})
    eval_dataset = _prepare_dataset(raw_dataset)["test"]

    args = TrainingArguments(
        output_dir="/tmp/eval_test_tmp",
        per_device_eval_batch_size=8,
        bf16=True,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        args=args,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )

    metrics = trainer.evaluate()
    eval_loss = metrics.get("eval_loss")
    if eval_loss is not None:
        try:
            metrics["perplexity"] = math.exp(eval_loss)
        except OverflowError:
            metrics["perplexity"] = float("inf")

    print("\nTest Evaluation Results:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    out_file = Path("reports/test_eval_metrics.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved test metrics to {out_file}")

if __name__ == "__main__":
    main()
