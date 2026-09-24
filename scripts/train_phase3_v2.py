"""Phase 3 v2 SFT Training Pipeline for ULM-4B (Qwen3.8-4B-Distill).

Features:
- LoRA SFT on Qwen3.8-4B-Distill with all 12 linear projections:
  ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj',
   'in_proj_qkv', 'in_proj_a', 'in_proj_b', 'in_proj_z', 'out_proj']
- Effective batch size = 32 (micro_batch=2, grad_accum=16)
- Learning rate = 5e-5, cosine schedule, warmup 0.03, max_seq_length = 2048, bf16
- Smoke test mode: 50 samples, forward/backward, save, adapter load/unload, non-thinking eval
- Gate-controlled checkpointing: Periodic evaluation against 100-prompt preservation benchmark
- Best checkpoint selection prioritizing preservation + dialect gain
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
    set_seed,
)
from trl import SFTConfig, SFTTrainer

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Import our evaluation module
from evaluate_preservation import run_evaluation

TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
    "in_proj_qkv", "in_proj_a", "in_proj_b", "in_proj_z", "out_proj",
]


class PreservationGateCallback(TrainerCallback):
    """Evaluates the 100-prompt preservation benchmark at specified step intervals."""

    def __init__(
        self,
        model_path: str,
        benchmark_path: str,
        eval_steps_list: list[int],
        reports_dir: Path,
        base_baseline_summary_path: Path,
    ):
        self.model_path = model_path
        self.benchmark_path = benchmark_path
        self.eval_steps_list = eval_steps_list
        self.reports_dir = reports_dir
        self.eval_history: list[dict[str, Any]] = []
        self.consecutive_violations = 0
        self.should_early_stop = False

        self.base_factual = 87.5
        self.base_memory = 90.0
        self.base_instruction = 100.0
        if base_baseline_summary_path.exists():
            try:
                with open(base_baseline_summary_path, "r", encoding="utf-8") as f:
                    b = json.load(f)
                    self.base_factual = b["factual_qa"]["accuracy_pct"]
                    self.base_memory = b["multi_turn"]["accuracy_pct"]
                    self.base_instruction = b["instruction_trap"]["accuracy_pct"]
            except Exception as e:
                print("Failed loading base baseline:", e)

    def _run_gate_eval(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, step: int):
        print(f"\n{'='*60}")
        print(f">>> [Gate Callback] Preservation Evaluation at Step {step} <<<")
        print(f"{'='*60}")

        ckpt_dir = Path(args.output_dir) / f"checkpoint-{step}"
        if not ckpt_dir.exists():
            print(f"Checkpoint dir {ckpt_dir} not yet on disk, waiting up to 10s...")
            for _ in range(5):
                if ckpt_dir.exists():
                    break
                time.sleep(2)

        if not ckpt_dir.exists():
            print(f"Warning: {ckpt_dir} still not found. Skipping gate evaluation.")
            return

        adapter_path = str(ckpt_dir)
        out_jsonl = self.reports_dir / f"checkpoint-{step}-eval.jsonl"
        out_summary = self.reports_dir / f"checkpoint-{step}-summary.json"

        # Run evaluation using GPU
        torch.cuda.empty_cache()
        rows, summary = run_evaluation(
            model_path=self.model_path,
            adapter_path=adapter_path,
            benchmark_path=self.benchmark_path,
            device="cuda:0",
            max_new_tokens=256,
        )

        with open(out_jsonl, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        summary["global_step"] = step
        with open(out_summary, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        self.eval_history.append(summary)

        # Check gates vs baseline
        factual_drop = self.base_factual - summary["factual_qa"]["accuracy_pct"]
        memory_drop = self.base_memory - summary["multi_turn"]["accuracy_pct"]
        inst_drop = self.base_instruction - summary["instruction_trap"]["accuracy_pct"]

        print(f"\n--- Gate Check Results (Step {step}) ---")
        print(f"  Factual QA:       {summary['factual_qa']['accuracy_pct']}% (Base: {self.base_factual}%, Drop: {factual_drop:+.1f}pp)")
        print(f"  Multi-turn Memory:{summary['multi_turn']['accuracy_pct']}% (Base: {self.base_memory}%, Drop: {memory_drop:+.1f}pp)")
        print(f"  Instruction/Trap: {summary['instruction_trap']['accuracy_pct']}% (Base: {self.base_instruction}%, Drop: {inst_drop:+.1f}pp)")
        print(f"  Dialect Score:    {summary['dialect_eval']['accuracy_pct']}% (Base: 30.0%)")

        violation = False
        if factual_drop >= 10.0:
            print(f"  [GATE ALERT] Critical Factual Drop >= 10pp ({factual_drop:.1f}pp)!")
            violation = True
        elif factual_drop >= 5.0:
            print(f"  [GATE WARNING] Factual Drop >= 5pp ({factual_drop:.1f}pp)")

        if memory_drop >= 10.0:
            print(f"  [GATE WARNING] Multi-turn Memory Drop >= 10pp ({memory_drop:.1f}pp)")
            violation = True

        if inst_drop >= 5.0:
            print(f"  [GATE WARNING] Instruction Drop >= 5pp ({inst_drop:.1f}pp)")
            violation = True

        if violation:
            self.consecutive_violations += 1
            if self.consecutive_violations >= 2:
                print("  [GATE STOP] 2 Consecutive Gate Violations! Triggering Early Stop!")
                control.should_training_stop = True
                self.should_early_stop = True
        else:
            self.consecutive_violations = 0

        print(f"{'='*60}\n")

    def on_save(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        step = state.global_step
        if step in self.eval_steps_list:
            self._run_gate_eval(args, state, control, step)


def run_smoke_test(model_path: str, data_path: str, output_dir: str):
    print("\n=======================================================")
    print(">>> Starting Phase 3 v2 Smoke Test (50 samples, 10 steps) <<<")
    print("=======================================================")

    smoke_out = Path(output_dir)
    smoke_out.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16, device_map="cuda:0")

    lora_cfg = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=TARGET_MODULES,
    )

    ds = load_dataset("json", data_files=data_path, split="train[:50]")

    sft_args = SFTConfig(
        output_dir=str(smoke_out),
        max_length=512,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=2,
        learning_rate=5e-5,
        max_steps=5,
        bf16=True,
        logging_steps=1,
        save_strategy="steps",
        save_steps=5,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_args,
        train_dataset=ds,
        peft_config=lora_cfg,
        processing_class=tokenizer,
    )

    train_res = trainer.train()
    loss = train_res.training_loss
    peak_vram = round(torch.cuda.max_memory_allocated() / (1024**3), 2)
    print(f"Smoke train finished: Loss={loss:.4f} (finite: {torch.isfinite(torch.tensor(loss))}), Peak VRAM={peak_vram} GB")

    trainer.save_model(str(smoke_out / "final_adapter"))
    tokenizer.save_pretrained(str(smoke_out / "final_adapter"))
    assert (smoke_out / "final_adapter" / "adapter_model.safetensors").exists(), "Smoke test adapter not saved!"
    print("Adapter saved successfully!")

    # Verify adapter load and non-thinking inference
    del model
    del trainer
    torch.cuda.empty_cache()

    base_m = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16, device_map="cuda:0")
    peft_m = PeftModel.from_pretrained(base_m, str(smoke_out / "final_adapter"))

    test_msg = [{"role": "user", "content": "울산 방언으로 '반갑습니다'를 어떻게 말해?"}]
    p = tokenizer.apply_chat_template(test_msg, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    inp = tokenizer(p, return_tensors="pt").to("cuda:0")
    with torch.no_grad():
        out = peft_m.generate(**inp, max_new_tokens=40, do_sample=False)
    gen = tokenizer.decode(out[0][inp.input_ids.shape[1]:], skip_special_tokens=True)
    print("Smoke non-thinking generation output:\n", repr(gen))

    # Unload adapter
    del peft_m
    del base_m
    torch.cuda.empty_cache()
    gc.collect()

    print("\n>>> SMOKE TEST RESULT: PASS <<<")
    print(f"Total smoke test duration: {time.time()-t0:.2f}s\n")
    return True


def run_full_training(
    model_path: str,
    train_data_path: str,
    val_data_path: str,
    output_dir: str,
    benchmark_path: str,
    reports_dir: str,
    batch_size: int = 2,
    gradient_accumulation_steps: int = 16,
    learning_rate: float = 5e-5,
    num_train_epochs: int = 2,
    max_seq_length: int = 2048,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    seed: int = 42,
):
    print("\n=======================================================")
    print(">>> Starting Phase 3 v2 Full Knowledge-Preserving SFT <<<")
    print(f"Base Model: {model_path}")
    print(f"Train Dataset: {train_data_path}")
    print(f"Output Dir: {output_dir}")
    print(f"Micro-batch: {batch_size}, GradAccum: {gradient_accumulation_steps} (Effective Batch: {batch_size*gradient_accumulation_steps})")
    print(f"LR: {learning_rate}, Epochs: {num_train_epochs}, SeqLen: {max_seq_length}")
    print("=======================================================")

    out_p = Path(output_dir)
    out_p.mkdir(parents=True, exist_ok=True)
    rep_p = Path(reports_dir)
    rep_p.mkdir(parents=True, exist_ok=True)

    set_seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16, device_map="cuda:0")

    lora_cfg = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=TARGET_MODULES,
    )

    train_ds = load_dataset("json", data_files=train_data_path, split="train")
    val_ds = load_dataset("json", data_files=val_data_path, split="train")
    print(f"Loaded datasets: Train={len(train_ds)}, Val={len(val_ds)}")

    total_steps = (len(train_ds) // (batch_size * gradient_accumulation_steps)) * num_train_epochs
    print(f"Estimated total optimizer steps: ~{total_steps}")
    # Evaluate at step 70, 140 (1 epoch), 210, 280 (2 epochs)
    eval_steps_list = [70, 140, 210, total_steps]

    gate_callback = PreservationGateCallback(
        model_path=model_path,
        benchmark_path=benchmark_path,
        eval_steps_list=eval_steps_list,
        reports_dir=rep_p,
        base_baseline_summary_path=rep_p / "base-baseline-summary.json",
    )

    sft_args = SFTConfig(
        output_dir=str(out_p),
        max_length=max_seq_length,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        num_train_epochs=num_train_epochs,
        warmup_steps=8,
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        optim="adamw_torch",
        bf16=True,
        logging_steps=10,
        save_strategy="steps",
        save_steps=70,
        eval_strategy="no",
        save_total_limit=5,
        gradient_checkpointing=True,
        seed=seed,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_args,
        train_dataset=train_ds,
        peft_config=lora_cfg,
        processing_class=tokenizer,
        callbacks=[gate_callback],
    )

    t0 = time.time()
    train_res = trainer.train()
    train_duration = round(time.time() - t0, 2)
    peak_vram = round(torch.cuda.max_memory_allocated() / (1024**3), 2)

    print(f"\nTraining Complete in {train_duration}s! Final Loss: {train_res.training_loss:.4f}, Peak VRAM: {peak_vram} GB")

    # Final checkpoint save
    final_ckpt = out_p / "checkpoint-final"
    trainer.save_model(str(final_ckpt))
    tokenizer.save_pretrained(str(final_ckpt))

    # Evaluate final checkpoint if not already evaluated
    final_step = trainer.state.global_step
    final_eval_summary = rep_p / f"checkpoint-{final_step}-summary.json"
    if not final_eval_summary.exists():
        print(f"Running final checkpoint evaluation at step {final_step}...")
        del model
        del trainer
        torch.cuda.empty_cache()
        rows, summary = run_evaluation(
            model_path=model_path,
            adapter_path=str(final_ckpt),
            benchmark_path=benchmark_path,
            device="cuda:0",
            max_new_tokens=256,
        )
        summary["global_step"] = final_step
        with open(final_eval_summary, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        with open(rep_p / f"checkpoint-{final_step}-eval.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        gate_callback.eval_history.append(summary)

    # Checkpoint selection:
    # Priority:
    # 1. factual >= 80% (base 87.5%)
    # 2. memory >= 80% (base 90%)
    # 3. instruction >= 90% (base 100%)
    # 4. Highest dialect score
    print("\n=======================================================")
    print(">>> Best Checkpoint Selection <<<")
    print("=======================================================")

    best_ckpt = None
    best_dialect_score = -1.0
    selection_log = []

    for ev in gate_callback.eval_history:
        st = ev.get("global_step", "unknown")
        fact = ev["factual_qa"]["accuracy_pct"]
        mem = ev["multi_turn"]["accuracy_pct"]
        inst = ev["instruction_trap"]["accuracy_pct"]
        dia = ev["dialect_eval"]["accuracy_pct"]

        passed_gates = (fact >= 80.0) and (mem >= 80.0) and (inst >= 90.0)
        selection_log.append({
            "step": st,
            "factual": fact,
            "memory": mem,
            "instruction": inst,
            "dialect": dia,
            "passed_preservation_gates": passed_gates,
        })
        print(f"Step {st}: Factual={fact}%, Memory={mem}%, Inst={inst}%, Dialect={dia}% | Gates Passed: {passed_gates}")

        if passed_gates and dia > best_dialect_score:
            best_dialect_score = dia
            best_ckpt = st

    if best_ckpt is None:
        # Fallback to the one with highest factual + dialect
        print("Warning: No checkpoint met all strict gates. Selecting highest harmonic score...")
        best_ckpt = max(gate_callback.eval_history, key=lambda x: x["factual_qa"]["accuracy_pct"] + x["dialect_eval"]["accuracy_pct"]).get("global_step")

    print(f"\n>>> SELECTED BEST CHECKPOINT: checkpoint-{best_ckpt} (Dialect: {best_dialect_score}%) <<<")

    # Copy best adapter to final destination
    best_adapter_dir = out_p.parent / "ulm-4b-phase3-v2-lora"
    if (out_p / f"checkpoint-{best_ckpt}").exists():
        src_adapter = out_p / f"checkpoint-{best_ckpt}"
    else:
        src_adapter = final_ckpt

    if best_adapter_dir.exists():
        shutil.rmtree(best_adapter_dir)
    shutil.copytree(src_adapter, best_adapter_dir)
    print(f"Best adapter ({src_adapter.name}) copied to: {best_adapter_dir}")

    # Save training metadata
    train_meta = {
        "base_model": model_path,
        "train_samples": len(train_ds),
        "validation_samples": len(val_ds),
        "total_steps": final_step,
        "runtime_s": train_duration,
        "peak_vram_gb": peak_vram,
        "best_checkpoint": f"checkpoint-{best_ckpt}",
        "best_adapter_path": str(best_adapter_dir),
        "selection_history": selection_log,
        "effective_batch_size": batch_size * gradient_accumulation_steps,
        "learning_rate": learning_rate,
        "lora_r": lora_r,
        "lora_alpha": lora_alpha,
    }
    with open(rep_p / "training_metadata.json", "w", encoding="utf-8") as f:
        json.dump(train_meta, f, indent=2, ensure_ascii=False)

    return best_adapter_dir, train_meta


def main():
    parser = argparse.ArgumentParser(description="Phase 3 v2 Training Pipeline")
    parser.add_argument("--model-path", default="/home/ubuntu/models/Qwen3.8-4B-Distill")
    parser.add_argument("--train-data", default="data/ulsan_dialect_phase3_v2/train.jsonl")
    parser.add_argument("--val-data", default="data/ulsan_dialect_phase3_v2/validation.jsonl")
    parser.add_argument("--output-dir", default="outputs/ulm-4b-phase3-v2-lora-training")
    parser.add_argument("--benchmark-path", default="data/preservation_benchmark_100.jsonl")
    parser.add_argument("--reports-dir", default="reports/phase3-v2")
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    if args.smoke_test:
        success = run_smoke_test(
            model_path=args.model_path,
            data_path=args.train_data,
            output_dir="/home/ubuntu/ULM-1.7B/outputs/smoke_test_lora",
        )
        sys.exit(0 if success else 1)

    run_full_training(
        model_path=args.model_path,
        train_data_path=args.train_data,
        val_data_path=args.val_data,
        output_dir=args.output_dir,
        benchmark_path=args.benchmark_path,
        reports_dir=args.reports_dir,
    )


if __name__ == "__main__":
    main()
