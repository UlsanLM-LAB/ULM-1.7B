"""TRL DPOTrainer script for ULM-1.7B Phase 4 Preference Optimization."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from trl import DPOConfig, DPOTrainer

DEFAULT_SYSTEM_PROMPT = (
    "울산 지역어 대화 assistant로서 자연스럽고 일상적인 울산 사투리로 상대방과 친근하게 대화한다. "
    "자연스럽고 편안한 일상 울산 말투를 기본으로 구사한다."
)


def format_dpo_sample(example: dict, tokenizer) -> dict:
    """Formats prompt with chat template so prompt ends with assistant generation prompt."""
    prompt_text = example["prompt"]
    messages = [
        {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
        {"role": "user", "content": prompt_text},
    ]
    template_kwargs = {"tokenize": False, "add_generation_prompt": True, "enable_thinking": False}
    try:
        formatted_prompt = tokenizer.apply_chat_template(messages, **template_kwargs)
    except TypeError:
        template_kwargs.pop("enable_thinking", None)
        formatted_prompt = tokenizer.apply_chat_template(messages, **template_kwargs)

    chosen_text = example["chosen"].strip()
    rejected_text = example["rejected"].strip()

    return {
        "prompt": formatted_prompt,
        "chosen": chosen_text,
        "rejected": rejected_text,
    }


def main():
    parser = argparse.ArgumentParser(description="Run DPO training on ULM-1.7B")
    parser.add_argument("--model-name", default="outputs/ulm-1.7b-phase4-sft-interim")
    parser.add_argument("--data-dir", default="data/ulsan_preference_phase4")
    parser.add_argument("--output-dir", default="outputs/ulm-1.7b-phase4-dpo-l40s")
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--num-epochs", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--max-prompt-length", type=int, default=256)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--eval-steps", type=int, default=50)
    parser.add_argument("--save-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--report-json", default=None)
    args = parser.parse_args()

    set_seed(args.seed)
    print(f"=== ULM-1.7B Phase 4 DPO Training ===")
    print(f"Base Model:    {args.model_name}")
    print(f"Data Dir:      {args.data_dir}")
    print(f"Output Dir:    {args.output_dir}")
    print(f"LR:            {args.learning_rate}")
    print(f"Beta:          {args.beta}")
    print(f"Batch Size:    {args.batch_size} (Grad Accum: {args.grad_accum}, Effective: {args.batch_size * args.grad_accum})")
    print(f"Epochs:        {args.num_epochs}, Max Steps: {args.max_steps}")
    print(f"LoRA:          r={args.lora_r}, alpha={args.lora_alpha}, dropout={args.lora_dropout}")

    # 1. Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. Datasets
    raw_train = load_dataset("json", data_files=os.path.join(args.data_dir, "train.jsonl"))["train"]
    raw_val = load_dataset("json", data_files=os.path.join(args.data_dir, "validation.jsonl"))["train"]

    train_dataset = raw_train.map(lambda ex: format_dpo_sample(ex, tokenizer), remove_columns=raw_train.column_names)
    eval_dataset = raw_val.map(lambda ex: format_dpo_sample(ex, tokenizer), remove_columns=raw_val.column_names)
    print(f"Dataset formatted: Train={len(train_dataset)}, Val={len(eval_dataset)}")

    # 3. Model
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
    )

    # 4. LoRA Config
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    # 5. DPO Config
    dpo_args = DPOConfig(
        output_dir=args.output_dir,
        beta=args.beta,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        max_length=args.max_length,
        bf16=True,
        logging_steps=args.logging_steps,
        eval_strategy="steps" if (args.max_steps > 0 or len(eval_dataset) > 0) else "no",
        eval_steps=args.eval_steps,
        save_strategy="steps" if args.max_steps > 0 else "epoch",
        save_steps=args.save_steps,
        save_total_limit=3,
        optim="adamw_torch_fused",
        lr_scheduler_type="cosine",
        warmup_steps=10,
        remove_unused_columns=False,
        report_to=[],
        seed=args.seed,
    )

    # 6. Trainer
    trainer = DPOTrainer(
        model=model,
        ref_model=None,  # Implicit reference model by disabling LoRA adapters
        args=dpo_args,
        peft_config=peft_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )

    t0 = time.perf_counter()
    train_result = trainer.train()
    elapsed = time.perf_counter() - t0

    print(f"\nDPO Training Finished in {elapsed:.2f}s!")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    metrics = train_result.metrics
    metrics["train_runtime"] = elapsed
    metrics["steps_per_second"] = round(train_result.global_step / max(elapsed, 0.001), 3)

    if torch.cuda.is_available():
        metrics["peak_vram_gib"] = round(torch.cuda.max_memory_allocated() / (1024 ** 3), 2)

    print("\nTraining Metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    # Evaluate final metrics
    eval_metrics = trainer.evaluate()
    print("\nEvaluation Metrics:")
    for k, v in eval_metrics.items():
        print(f"  {k}: {v}")
    metrics.update({f"final_{k}": v for k, v in eval_metrics.items()})

    if args.report_json:
        os.makedirs(os.path.dirname(args.report_json), exist_ok=True)
        with open(args.report_json, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        print(f"Report saved to {args.report_json}")


if __name__ == "__main__":
    main()
