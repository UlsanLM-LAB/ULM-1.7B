"""Resumable Phase 3 v3 LoRA SFT with NVMe checkpoints and EBS gate copies."""

from __future__ import annotations

import argparse
import gc
import json
import math
import random
import re
import shutil
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback, set_seed
from trl import SFTConfig, SFTTrainer

from evaluate_preservation import run_evaluation
from train_phase3_v2 import TARGET_MODULES
from phase3_v3_data import completion_dataset

GATES = {"factual_qa": 80.0, "multi_turn": 85.0, "instruction_trap": 95.0, "dialect_eval": 60.0}
REQUIRED_STATE = ("optimizer.pt", "scheduler.pt", "trainer_state.json")


def stalled_with_forgetting(current: dict, previous: dict) -> bool:
    """Stop when dialect fails to improve while knowledge or memory regresses."""
    return current["dialect_eval"]["accuracy_pct"] <= previous["dialect_eval"]["accuracy_pct"] and (
        current["factual_qa"]["accuracy_pct"] < previous["factual_qa"]["accuracy_pct"]
        or current["multi_turn"]["accuracy_pct"] < previous["multi_turn"]["accuracy_pct"]
    )


@contextmanager
def preserved_rng_state():
    """Keep gate sampling and model loading out of the training RNG stream."""
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all()
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)
        torch.cuda.set_rng_state_all(cuda_states)


def checked_copy(source: Path, target: Path) -> None:
    """Copy a full resumable checkpoint before publishing it on EBS."""
    temporary = target.with_name(target.name + ".copying")
    if temporary.exists():
        shutil.rmtree(temporary)
    shutil.copytree(source, temporary)
    missing = [name for name in REQUIRED_STATE if not (temporary / name).is_file()]
    if missing:
        shutil.rmtree(temporary)
        raise RuntimeError(f"Incomplete checkpoint {source}: {missing}")
    if target.exists():
        shutil.rmtree(target)
    temporary.rename(target)


class V3GateCallback(TrainerCallback):
    def __init__(
        self, model_path: str, gate_path: str, reports: Path, ebs: Path, nvme: Path, interval: int,
        independent_gate_path: str | None = None,
    ) -> None:
        self.model_path = model_path
        self.gate_path = gate_path
        self.reports = reports
        self.ebs = ebs
        self.nvme = nvme
        self.interval = interval
        self.independent_gate_path = independent_gate_path
        self.history = sorted(
            (json.loads(p.read_text()) for p in reports.glob("gate-step-*.json")),
            key=lambda row: row["global_step"],
        )
        baseline_path = reports / "base-gate-40-summary.json"
        if not baseline_path.exists():
            baseline_path = Path(gate_path).with_name("base-gate-40-summary.json")
        self.base_summary = json.loads(baseline_path.read_text()) if baseline_path.exists() else None

    def _mirror(self, source: Path, step: int, kind: str) -> None:
        target = self.ebs / f"{kind}-checkpoint-{step}"
        checked_copy(source, target)
        for previous in self.ebs.glob(f"{kind}-checkpoint-*"):
            if previous != target and previous.is_dir():
                shutil.rmtree(previous)

    def on_save(self, args, state, control, **kwargs):
        step = state.global_step
        source = self.nvme / f"checkpoint-{step}"
        self._mirror(source, step, "latest")
        if step % self.interval and step != state.max_steps:
            return control
        if any(row["global_step"] == step for row in self.history):
            return control

        torch.cuda.empty_cache()
        start = time.monotonic()
        with preserved_rng_state():
            rows, summary = run_evaluation(
                model_path=self.model_path,
                adapter_path=str(source),
                benchmark_path=self.gate_path,
                device="cuda:0",
                max_new_tokens=256,
            )
        summary["global_step"] = step
        summary["runtime_s"] = round(time.monotonic() - start, 2)
        summary["prompt_echo_count"] = sum(
            len(row["prompt"]) >= 20 and row["prompt"][:20] in row["final_answer"] for row in rows
        )
        summary["dialect_ending_overuse_count"] = sum(
            sum(row["final_answer"].count(marker) for marker in ("데이", "제", "노")) > 3
            for row in rows
        )
        summary["english_leakage_count"] = sum(
            bool(re.search(r"\b[A-Za-z]{4,}\b", row["final_answer"]))
            for row in rows
            if row["category"] == "dialect_eval"
        )
        if self.independent_gate_path:
            with preserved_rng_state():
                independent_rows, independent_summary = run_evaluation(
                    model_path=self.model_path, adapter_path=str(source),
                    benchmark_path=self.independent_gate_path, device="cuda:0", max_new_tokens=256,
                )
            summary["independent20"] = independent_summary["dialect_eval"]
            (self.reports / f"independent-step-{step}.json").write_text(
                json.dumps(independent_summary, ensure_ascii=False, indent=2) + "\n")
            with (self.reports / f"independent-step-{step}.jsonl").open("w") as file:
                for row in independent_rows:
                    file.write(json.dumps(row, ensure_ascii=False) + "\n")
        (self.reports / f"gate-step-{step}.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
        )
        with (self.reports / f"gate-step-{step}.jsonl").open("w") as file:
            for row in rows:
                file.write(json.dumps(row, ensure_ascii=False) + "\n")

        previous = self.history[-1] if self.history else self.base_summary
        if self.base_summary is None:
            raise FileNotFoundError("base-gate-40-summary.json is required for gate decisions")
        self.history.append(summary)
        passed = all(summary[key]["accuracy_pct"] >= threshold for key, threshold in GATES.items())
        def acceptable(row):
            if self.independent_gate_path:
                return (row["factual_qa"]["accuracy_pct"] >= 80.0
                        and row["multi_turn"]["accuracy_pct"] >= 85.0
                        and row["instruction_trap"]["accuracy_pct"] >= 95.0)
            return (row["factual_qa"]["accuracy_pct"] >= 83.3
                    and row["multi_turn"]["accuracy_pct"] >= 87.5
                    and row["instruction_trap"]["accuracy_pct"] == 100.0)
        def quality(row):
            if self.independent_gate_path:
                return (row["dialect_eval"]["accuracy_pct"] + row["independent20"]["accuracy_pct"],
                        row["dialect_eval"]["accuracy_pct"],
                        row["factual_qa"]["accuracy_pct"] + row["multi_turn"]["accuracy_pct"]
                        + row["instruction_trap"]["accuracy_pct"])
            return (row["dialect_eval"]["accuracy_pct"],
                    row["factual_qa"]["accuracy_pct"] + row["multi_turn"]["accuracy_pct"]
                    + row["instruction_trap"]["accuracy_pct"])
        earlier = [row for row in self.history[:-1] if acceptable(row)]
        independent_base = None
        if self.independent_gate_path:
            independent_base = json.loads((self.reports / "base-independent-20-summary.json").read_text())["dialect_eval"]["accuracy_pct"]
        success_candidate = (self.independent_gate_path and step >= 140 and passed
                             and summary["independent20"]["accuracy_pct"] >= independent_base + 10.0)
        if (acceptable(summary) and (independent_base is None or summary["independent20"]["accuracy_pct"] > independent_base)
                and (success_candidate or not earlier or quality(summary) > max(map(quality, earlier)))):
            target = self.ebs / f"best-checkpoint-{step}"
            temp = target.with_name(target.name + ".copying")
            if temp.exists():
                shutil.rmtree(temp)
            shutil.copytree(source, temp, ignore=shutil.ignore_patterns(
                "optimizer.pt", "scheduler.pt", "rng_state.pth", "training_args.bin"))
            temp.rename(target)
            for old in self.ebs.glob("best-checkpoint-*"):
                if old != target and old.is_dir():
                    shutil.rmtree(old)
        if self.independent_gate_path:
            pilot = json.loads((self.reports / "pilot-summary.json").read_text())
            pilot20 = json.loads((self.reports / "pilot-independent-20-summary.json").read_text())["dialect_eval"]["accuracy_pct"]
            old_score = summary["dialect_eval"]["accuracy_pct"]
            mini_score = summary["independent20"]["accuracy_pct"]
            old_floor = max(25.0, pilot["dialect_eval"]["accuracy_pct"])
            mini_floor = max(independent_base, pilot20)
            if step == 70:
                progressing = old_score >= old_floor and mini_score >= mini_floor and (old_score > old_floor or mini_score > mini_floor)
            else:
                progressing = (previous is not None and old_score >= previous["dialect_eval"]["accuracy_pct"]
                               and mini_score >= previous["independent20"]["accuracy_pct"]
                               and (old_score > previous["dialect_eval"]["accuracy_pct"] or mini_score > previous["independent20"]["accuracy_pct"]))
            success = (step >= 140 and passed and mini_score >= independent_base + 10.0)
            if not acceptable(summary) or success or not progressing:
                control.should_training_stop = True
            summary["gate_decision"] = "success" if success else ("continue" if progressing and acceptable(summary) else "stop")
            (self.reports / f"gate-step-{step}.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
        else:
            if passed or not acceptable(summary) or (step == 70 and summary["dialect_eval"]["accuracy_pct"] <= self.base_summary["dialect_eval"]["accuracy_pct"]):
                control.should_training_stop = True
            if step >= 140 and previous and summary["dialect_eval"]["accuracy_pct"] <= previous["dialect_eval"]["accuracy_pct"]:
                control.should_training_stop = True
        print(
            f"Gate {step}: factual={summary['factual_qa']['accuracy_pct']} "
            f"memory={summary['multi_turn']['accuracy_pct']} "
            f"instruction={summary['instruction_trap']['accuracy_pct']} "
            f"dialect={summary['dialect_eval']['accuracy_pct']} pass={passed}",
            flush=True,
        )
        gc.collect()
        torch.cuda.empty_cache()
        return control


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/home/ubuntu/models/Qwen3.8-4B-Distill")
    parser.add_argument("--train", default="data/ulsan_dialect_phase3_v3_candidate_b/train.jsonl")
    parser.add_argument("--validation", default="data/ulsan_dialect_phase3_v3_candidate_b/validation.jsonl")
    parser.add_argument("--gate", default="reports/phase3-v3/gate-40.jsonl")
    parser.add_argument("--nvme", default="/opt/dlami/nvme/phase3-v3")
    parser.add_argument("--ebs", default="outputs/ulm-4b-phase3-v3")
    parser.add_argument("--reports", default="reports/phase3-v3/recovery")
    parser.add_argument("--micro-batch", type=int, required=True)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--group-by-length", action="store_true")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--max-steps", type=int, default=210)
    parser.add_argument("--independent-gate", default=None)
    parser.add_argument("--sequential", action="store_true")
    args = parser.parse_args()
    if 32 % args.micro_batch:
        parser.error("micro batch must divide effective batch 32")
    nvme = Path(args.nvme)
    ebs = Path(args.ebs)
    reports = Path(args.reports)
    for directory in (nvme, ebs, reports):
        directory.mkdir(parents=True, exist_ok=True)
    existing = list(nvme.glob("checkpoint-*")) + list(ebs.glob("latest-checkpoint-*"))
    if existing and not args.resume:
        parser.error("existing checkpoint found; pass --resume with its path")

    set_seed(42)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="cuda:0", attn_implementation="sdpa"
    )
    train = completion_dataset(load_dataset("json", data_files=args.train, split="train"), tokenizer)
    validation = completion_dataset(load_dataset("json", data_files=args.validation, split="train"), tokenizer)
    config = SFTConfig(
        output_dir=str(nvme),
        max_length=2048,
        per_device_train_batch_size=args.micro_batch,
        gradient_accumulation_steps=32 // args.micro_batch,
        learning_rate=args.lr,
        max_steps=args.max_steps,
        warmup_steps=8,
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        optim="adamw_torch",
        bf16=True,
        save_strategy="steps",
        save_steps=70,
        save_total_limit=2,
        eval_strategy="no",
        logging_steps=10,
        gradient_checkpointing=args.gradient_checkpointing,
        completion_only_loss=True,
        train_sampling_strategy="sequential" if args.sequential else ("group_by_length" if args.group_by_length else "random"),
        dataloader_num_workers=args.workers,
        dataloader_pin_memory=True,
        dataloader_persistent_workers=args.workers > 0,
        dataloader_prefetch_factor=2 if args.workers else None,
        seed=42,
        data_seed=42,
        report_to="none",
    )
    callback = V3GateCallback(args.model, args.gate, reports, ebs, nvme, 70, args.independent_gate)
    trainer = SFTTrainer(
        model=model,
        args=config,
        train_dataset=train,
        eval_dataset=validation,
        peft_config=LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=TARGET_MODULES,
        ),
        processing_class=tokenizer,
        callbacks=[callback],
    )
    labels = next(iter(trainer.get_train_dataloader()))["labels"][0]
    assert (labels == -100).any() and (labels != -100).any()
    assert "<|im_start|>user" not in tokenizer.decode(labels[labels != -100])
    start = time.monotonic()
    result = trainer.train(resume_from_checkpoint=args.resume)
    if not math.isfinite(result.training_loss):
        raise RuntimeError("non-finite training loss")
    metadata = {
        "global_step": trainer.state.global_step,
        "max_steps": trainer.state.max_steps,
        "training_loss": result.training_loss,
        "runtime_s": round(time.monotonic() - start, 2),
        "peak_vram_gib": round(torch.cuda.max_memory_reserved() / 2**30, 3),
        "micro_batch": args.micro_batch,
        "gradient_accumulation": 32 // args.micro_batch,
        "effective_batch": 32,
        "gradient_checkpointing": args.gradient_checkpointing,
        "group_by_length": args.group_by_length,
        "dataloader_num_workers": args.workers,
        "packing": False,
        "attention_backend": model.config._attn_implementation,
        "resume_from": args.resume,
        "gate_steps": [row["global_step"] for row in callback.history],
    }
    (reports / "training_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(metadata, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
