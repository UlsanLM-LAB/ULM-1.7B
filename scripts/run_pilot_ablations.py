"""Run pilot ablation experiments (35 optimizer steps each) on EC2 L40S.

Candidates:
- Pilot A: Current config reproduction (LR 5e-5, raw v3 dataset, assistant_only_loss=False)
- Pilot B: LR 1.5e-5 (LR 1.5e-5, raw v3 dataset, assistant_only_loss=False)
- Pilot C: LR 1.5e-5 + Rebalanced mixture (LR 1.5e-5, rebalanced 45/40/15, assistant_only_loss=False)
- Pilot D: LR 1.5e-5 + Rebalanced mixture + assistant_only_loss=True
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import shutil
import sys
import time
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from trl import SFTConfig, SFTTrainer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_preservation import run_evaluation
from train_phase3_v2 import TARGET_MODULES

BASE_MODEL = "/home/ubuntu/models/Qwen3.8-4B-Distill"
GATE_PATH = "reports/phase3-v3/gate-60.jsonl"
NVME_PILOTS_DIR = Path("/opt/dlami/nvme/phase3-v3-pilots")
REPORTS_DIR = Path("reports/phase3-v3")


def run_single_pilot(
    name: str,
    train_path: str,
    val_path: str,
    lr: float,
    assistant_only_loss: bool,
    max_steps: int = 35,
    micro_batch: int = 8,
    grad_accum: int = 4,
) -> dict:
    print(f"\n{'='*70}")
    print(f">>> Running {name} <<<")
    print(f"Dataset: {train_path}")
    print(f"LR: {lr}, Assistant Only Loss: {assistant_only_loss}")
    print(f"Steps: {max_steps}, Effective Batch: {micro_batch * grad_accum}")
    print(f"{'='*70}\n")

    pilot_out = NVME_PILOTS_DIR / name
    if pilot_out.exists():
        shutil.rmtree(pilot_out)
    pilot_out.mkdir(parents=True, exist_ok=True)

    set_seed(42)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        dtype=torch.bfloat16,
        device_map="cuda:0",
        attn_implementation="sdpa",
    )

    train_ds = load_dataset("json", data_files=train_path, split="train")
    val_ds = load_dataset("json", data_files=val_path, split="train")

    sft_args = SFTConfig(
        output_dir=str(pilot_out),
        max_length=2048,
        per_device_train_batch_size=micro_batch,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        max_steps=max_steps,
        warmup_steps=8,
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        optim="adamw_torch",
        bf16=True,
        logging_steps=5,
        save_strategy="no",
        eval_strategy="no",
        gradient_checkpointing=True,
        assistant_only_loss=assistant_only_loss,
        seed=42,
        data_seed=42,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        peft_config=LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=TARGET_MODULES,
        ),
        processing_class=tokenizer,
    )

    t0 = time.monotonic()
    result = trainer.train()
    train_time = round(time.monotonic() - t0, 2)
    print(f"[{name}] Train completed in {train_time}s! Loss: {result.training_loss:.4f}")

    adapter_save_path = pilot_out / "final_adapter"
    trainer.save_model(str(adapter_save_path))
    tokenizer.save_pretrained(str(adapter_save_path))

    del model
    del trainer
    gc.collect()
    torch.cuda.empty_cache()

    # Evaluate adapter on gate-60
    print(f"[{name}] Evaluating on 60-prompt preservation gate...")
    t_eval = time.monotonic()
    rows, summary = run_evaluation(
        model_path=BASE_MODEL,
        adapter_path=str(adapter_save_path),
        benchmark_path=GATE_PATH,
        device="cuda:0",
        max_new_tokens=256,
    )
    eval_time = round(time.monotonic() - t_eval, 2)

    eval_out_jsonl = REPORTS_DIR / f"{name}_eval.jsonl"
    with open(eval_out_jsonl, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    summary["pilot_name"] = name
    summary["train_time_s"] = train_time
    summary["eval_time_s"] = eval_time
    summary["final_loss"] = round(result.training_loss, 4)
    summary["lr"] = lr
    summary["assistant_only_loss"] = assistant_only_loss
    summary["dataset"] = train_path

    print(f"\n--- {name} Results ---")
    print(f"  Factual QA:       {summary['factual_qa']['accuracy_pct']}% ({summary['factual_qa']['hits']}/{summary['factual_qa']['total']})")
    print(f"  Multi-turn Memory:{summary['multi_turn']['accuracy_pct']}% ({summary['multi_turn']['hits']}/{summary['multi_turn']['total']})")
    print(f"  Instruction/Trap: {summary['instruction_trap']['accuracy_pct']}% ({summary['instruction_trap']['hits']}/{summary['instruction_trap']['total']})")
    print(f"  Dialect Score:    {summary['dialect_eval']['accuracy_pct']}% ({summary['dialect_eval']['hits']}/{summary['dialect_eval']['total']})")
    print(f"  Avg Latency:      {summary['avg_latency_ms']} ms | Avg Tokens: {summary['avg_tokens']}")
    print(f"{'='*70}\n")

    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", default=None, help="Run only specific pilot: A, B, C, D")
    parser.add_argument("--steps", type=int, default=35)
    args = parser.parse_args()

    NVME_PILOTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    pilots = {
        "Pilot_A": {
            "train_path": "data/ulsan_dialect_phase3_v3/train.jsonl",
            "val_path": "data/ulsan_dialect_phase3_v3/validation.jsonl",
            "lr": 5e-5,
            "assistant_only_loss": False,
        },
        "Pilot_B": {
            "train_path": "data/ulsan_dialect_phase3_v3/train.jsonl",
            "val_path": "data/ulsan_dialect_phase3_v3/validation.jsonl",
            "lr": 1.5e-5,
            "assistant_only_loss": False,
        },
        "Pilot_C": {
            "train_path": "data/ulsan_dialect_phase3_v3_rebalanced/train.jsonl",
            "val_path": "data/ulsan_dialect_phase3_v3_rebalanced/validation.jsonl",
            "lr": 1.5e-5,
            "assistant_only_loss": False,
        },
        "Pilot_D": {
            "train_path": "data/ulsan_dialect_phase3_v3_rebalanced/train.jsonl",
            "val_path": "data/ulsan_dialect_phase3_v3_rebalanced/validation.jsonl",
            "lr": 1.5e-5,
            "assistant_only_loss": True,
        },
    }

    results = {}
    out_summary_file = REPORTS_DIR / "pilot_ablation_results.json"
    if out_summary_file.exists():
        try:
            results = json.loads(out_summary_file.read_text())
        except Exception:
            results = {}

    for name, cfg in pilots.items():
        if args.only and args.only.upper() not in name.upper():
            continue
        summary = run_single_pilot(
            name=name,
            train_path=cfg["train_path"],
            val_path=cfg["val_path"],
            lr=cfg["lr"],
            assistant_only_loss=cfg["assistant_only_loss"],
            max_steps=args.steps,
        )
        results[name] = summary
        out_summary_file.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n")

    print("\n================ FINAL PILOT COMPARISON ================")
    print(f"{'Pilot':<10} | {'LR':<7} | {'Replay?':<7} | {'Masking':<7} | {'Factual':<8} | {'Memory':<8} | {'Instruction':<11} | {'Dialect':<8}")
    print("-" * 80)
    for name, res in results.items():
        f = res["factual_qa"]["accuracy_pct"]
        m = res["multi_turn"]["accuracy_pct"]
        i = res["instruction_trap"]["accuracy_pct"]
        d = res["dialect_eval"]["accuracy_pct"]
        rep = "Yes" if "rebalanced" in res["dataset"] else "No"
        mask = "AsstOnly" if res["assistant_only_loss"] else "None"
        lr_s = f"{res['lr']:.1e}"
        print(f"{name:<10} | {lr_s:<7} | {rep:<7} | {mask:<7} | {f:<7.1f}% | {m:<7.1f}% | {i:<10.1f}% | {d:<7.1f}%")
    print("========================================================\n")


if __name__ == "__main__":
    main()
