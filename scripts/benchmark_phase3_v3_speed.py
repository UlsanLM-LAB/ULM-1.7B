"""Short, disposable Phase 3 v3 throughput trials on the training GPU."""

from __future__ import annotations

import argparse
import gc
import json
import math
import subprocess
import threading
import time
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback, set_seed
from trl import SFTConfig, SFTTrainer

from train_phase3_v2 import TARGET_MODULES


class StepClock(TrainerCallback):
    def __init__(self) -> None:
        self.start = 0.0
        self.steps: list[float] = []

    def on_train_begin(self, args, state, control, **kwargs):
        self.start = time.perf_counter()

    def on_step_end(self, args, state, control, **kwargs):
        now = time.perf_counter()
        self.steps.append(now - self.start)
        self.start = now


def monitor_gpu(stop: threading.Event, values: list[int]) -> None:
    while not stop.wait(1):
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0:
            try:
                values.append(int(result.stdout.splitlines()[0].strip()))
            except (ValueError, IndexError):
                pass


def trial(
    name: str,
    micro: int,
    checkpointing: bool,
    group_by_length: bool,
    workers: int,
    model_path: str,
    dataset,
    tokenizer,
    steps: int,
    packing: bool = False,
) -> dict:
    set_seed(42)
    torch.cuda.empty_cache()
    model = trainer = None
    stop = threading.Event()
    gpu_values: list[int] = []
    watcher = None
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path, dtype=torch.bfloat16, device_map="cuda:0"
        )
        args = SFTConfig(
            output_dir=f"/opt/dlami/nvme/phase3-v3-benchmark/{name}",
            max_length=2048,
            packing=packing,
            per_device_train_batch_size=micro,
            gradient_accumulation_steps=32 // micro,
            max_steps=steps,
            learning_rate=5e-5,
            warmup_steps=8,
            bf16=True,
            optim="adamw_torch",
            save_strategy="no",
            eval_strategy="no",
            logging_steps=1,
            report_to="none",
            gradient_checkpointing=checkpointing,
            train_sampling_strategy="group_by_length" if group_by_length else "random",
            dataloader_num_workers=workers,
            dataloader_pin_memory=True,
            dataloader_persistent_workers=workers > 0,
            dataloader_prefetch_factor=2 if workers > 0 else None,
            seed=42,
            data_seed=42,
        )
        clock = StepClock()
        trainer = SFTTrainer(
            model=model,
            args=args,
            train_dataset=dataset,
            peft_config=LoraConfig(
                r=16,
                lora_alpha=32,
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=TARGET_MODULES,
            ),
            processing_class=tokenizer,
            callbacks=[clock],
        )
        lengths = [len(ids) for ids in trainer.train_dataset["input_ids"]]
        # Every trial traverses the same 96 examples over three optimizer steps.
        useful_tokens = sum(lengths[: steps * 32])
        torch.cuda.reset_peak_memory_stats()
        watcher = threading.Thread(target=monitor_gpu, args=(stop, gpu_values), daemon=True)
        watcher.start()
        start = time.perf_counter()
        result = trainer.train()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        return {
            "name": name,
            "micro_batch": micro,
            "gradient_accumulation": 32 // micro,
            "effective_batch": 32,
            "gradient_checkpointing": checkpointing,
            "group_by_length": group_by_length,
            "packing": packing,
            "dataloader_num_workers": workers,
            "attention_backend": model.config._attn_implementation,
            "seconds": round(elapsed, 3),
            "seconds_per_step": round(elapsed / steps, 3),
            "steady_seconds_per_step": round(
                sum(clock.steps[1:]) / max(1, len(clock.steps) - 1), 3
            ),
            "samples_per_second": round(steps * 32 / elapsed, 3),
            "useful_tokens_per_second": round(useful_tokens / elapsed, 3),
            "mean_gpu_utilization_pct": round(sum(gpu_values) / len(gpu_values), 1)
            if gpu_values
            else None,
            "peak_vram_gib": round(torch.cuda.max_memory_reserved() / 2**30, 3),
            "loss": round(result.training_loss, 5),
            "finite_loss": math.isfinite(result.training_loss),
        }
    except torch.cuda.OutOfMemoryError as exc:
        return {"name": name, "micro_batch": micro, "status": "OOM", "error": str(exc)[:300]}
    except Exception as exc:
        return {"name": name, "micro_batch": micro, "status": "error", "error": repr(exc)}
    finally:
        stop.set()
        if watcher:
            watcher.join(timeout=2)
        del trainer, model
        gc.collect()
        torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/home/ubuntu/models/Qwen3.8-4B-Distill")
    parser.add_argument("--train", default="data/ulsan_dialect_phase3_v3/train.jsonl")
    parser.add_argument("--output", default="reports/phase3-v3/speed_trials.json")
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--extra", action="store_true")
    parser.add_argument("--packing", action="store_true")
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    dataset = load_dataset("json", data_files=args.train, split="train")
    dataset = dataset.select(range(4000 if args.packing else args.steps * 32))
    baseline_specs = [
        ("a_b2_g16_gc", 2, True, False, 0),
        ("b_b4_g8_gc", 4, True, False, 0),
        ("c_b8_g4_gc", 8, True, False, 0),
        ("d_b16_g2_gc", 16, True, False, 0),
        ("b_b4_g8_nogc", 4, False, False, 0),
        ("c_b8_g4_nogc", 8, False, False, 0),
        ("d_b16_g2_nogc", 16, False, False, 0),
    ]
    extra_specs = [
        ("c_b8_g4_gc_length", 8, True, True, 0),
        ("c_b8_g4_gc_workers2", 8, True, False, 2),
        ("b_b4_g8_gc_length", 4, True, True, 0),
    ]
    specs = (
        [("packing_b2_g16_gc", 2, True, False, 0)]
        if args.packing
        else extra_specs
        if args.extra
        else baseline_specs
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    results: list[dict] = []
    for spec in specs:
        print(f"TRIAL {spec[0]}", flush=True)
        row = trial(*spec, args.model, dataset, tokenizer, args.steps, args.packing)
        print(json.dumps(row, ensure_ascii=False), flush=True)
        results.append(row)
        output.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
