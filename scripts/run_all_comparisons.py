"""Orchestrate full 500-item UlsanBench v2 comparison runs across all models.

Sequential execution to prevent GPU contention and OOM.
Saves raw predictions to reports/model-comparison/raw/<name>.jsonl.
Generates consolidated reports/model-comparison/summary.json, config.json, models.json.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON_BIN = "/home/ubuntu/ULM-1.7B/.venv/bin/python"
SCRIPT_RUNNER = ROOT / "scripts/run_model_comparison.py"
OUTPUT_DIR = ROOT / "reports/model-comparison"
RAW_DIR = OUTPUT_DIR / "raw"

MODELS_CONFIG = [
    {
        "id": "ulm-4b-arm-b-neutral",
        "display_name": "ULM-4B Arm B (Neutral)",
        "model": "/home/ubuntu/models/Qwen3.8-4B-Distill",
        "adapter": "/home/ubuntu/models/ULM-4B-Arm-B",
        "model_id": "UlsanLM-LAB/ULM-4B-Arm-B (Base: empero-ai/Qwen3.8-4B-Distill)",
        "params": "4.21B",
        "decoding_mode": "neutral",
        "quantization": "None (bf16)",
        "type": "dialect_specialized",
    },
    {
        "id": "ulm-4b-arm-b-deployed",
        "display_name": "ULM-4B Arm B (Deployed)",
        "model": "/home/ubuntu/models/Qwen3.8-4B-Distill",
        "adapter": "/home/ubuntu/models/ULM-4B-Arm-B",
        "model_id": "UlsanLM-LAB/ULM-4B-Arm-B (Base: empero-ai/Qwen3.8-4B-Distill)",
        "params": "4.21B",
        "decoding_mode": "deployed",
        "quantization": "None (bf16)",
        "type": "dialect_specialized_deployed",
    },
    {
        "id": "qwen3.8-4b-base",
        "display_name": "Qwen3.8-4B-Distill (Base)",
        "model": "/home/ubuntu/models/Qwen3.8-4B-Distill",
        "adapter": None,
        "model_id": "empero-ai/Qwen3.8-4B-Distill",
        "params": "4.21B",
        "decoding_mode": "neutral",
        "quantization": "None (bf16)",
        "type": "direct_base",
    },
    {
        "id": "qwen2.5-3b-instruct",
        "display_name": "Qwen2.5-3B-Instruct",
        "model": "Qwen/Qwen2.5-3B-Instruct",
        "adapter": None,
        "model_id": "Qwen/Qwen2.5-3B-Instruct",
        "params": "3.09B",
        "decoding_mode": "neutral",
        "quantization": "None (bf16)",
        "type": "general_open_llm_3b",
    },
    {
        "id": "qwen2.5-7b-instruct",
        "display_name": "Qwen2.5-7B-Instruct",
        "model": "Qwen/Qwen2.5-7B-Instruct",
        "adapter": None,
        "model_id": "Qwen/Qwen2.5-7B-Instruct",
        "params": "7.61B",
        "decoding_mode": "neutral",
        "quantization": "None (bf16)",
        "type": "general_open_llm_7b",
    },
    {
        "id": "llama-3.2-korean-bllossom-3b",
        "display_name": "Llama-3.2-Korean-Bllossom-3B",
        "model": "Bllossom/llama-3.2-Korean-Bllossom-3B",
        "adapter": None,
        "model_id": "Bllossom/llama-3.2-Korean-Bllossom-3B",
        "params": "3.21B",
        "decoding_mode": "neutral",
        "quantization": "None (bf16)",
        "type": "korean_open_llm_3b",
    },
]


def run_benchmark_for_model(m_cfg: dict, batch_size: int = 16) -> dict:
    model_id = m_cfg["id"]
    model_out = OUTPUT_DIR / model_id
    model_out.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    cmd = [
        PYTHON_BIN,
        str(SCRIPT_RUNNER),
        "--model", m_cfg["model"],
        "--name", model_id,
        "--output-dir", str(model_out),
        "--raw-dir", str(RAW_DIR),
        "--batch-size", str(batch_size),
        "--decoding-mode", m_cfg["decoding_mode"],
        "--dtype", "bfloat16",
        "--device", "cuda:0",
    ]
    if m_cfg["adapter"]:
        cmd.extend(["--adapter", m_cfg["adapter"]])

    print(f"\n{'='*70}\n[START] {m_cfg['display_name']} ({model_id})\n{'='*70}", flush=True)
    t0 = time.time()
    res = subprocess.run(cmd, text=True, check=True)
    elapsed = time.time() - t0
    print(f"[DONE] {m_cfg['display_name']} finished in {elapsed:.1f}s", flush=True)

    summary_file = model_out / "summary.json"
    if not summary_file.exists():
        raise RuntimeError(f"Missing summary file: {summary_file}")

    with summary_file.open("r", encoding="utf-8") as f:
        summary_data = json.load(f)

    # Attach config metadata
    summary_data["meta"] = m_cfg
    return summary_data


def main():
    parser = argparse.ArgumentParser(description="Run full UlsanBench v2 comparison")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    # Save models.json and config.json upfront
    models_metadata = {m["id"]: m for m in MODELS_CONFIG}
    with (OUTPUT_DIR / "models.json").open("w", encoding="utf-8") as f:
        json.dump(models_metadata, f, indent=2, ensure_ascii=False)

    global_config = {
        "benchmark": "UlsanBench v2",
        "benchmark_dataset": "data/ulsanbench_v1/benchmark.jsonl",
        "total_items": 500,
        "categories": {
            "comprehension": 100,
            "generation": 150,
            "identification": 100,
            "grammar": 75,
            "context": 75,
        },
        "evaluation_pipeline": "evaluate_ulsanbench_v2.py (paraphrase-multilingual-MiniLM-L12-v2 embedding + lexical & ending matching)",
        "hardware": {
            "aws_region": "ap-northeast-2",
            "ec2_instance": "i-0f732bf7d1cc409b4",
            "instance_type": "g6e.xlarge",
            "gpu": "NVIDIA L40S 46GB",
            "cuda_version": "13.2",
            "driver_version": "595.91.07",
            "pytorch_version": "2.14.0+cu130",
        },
        "evaluation_date": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d"),
        "decoding": {
            "neutral": {
                "do_sample": False,
                "temperature": 0.0,
                "enable_thinking": False,
                "max_new_tokens": {"identification": 16, "other": 96},
            },
            "deployed": {
                "do_sample": False,
                "temperature": 0.0,
                "enable_thinking": False,
                "max_new_tokens": {"identification": 16, "other": 96},
                "context_task_extra": {"repetition_penalty": 1.10, "no_repeat_ngram_size": 3},
            },
        },
    }
    with (OUTPUT_DIR / "config.json").open("w", encoding="utf-8") as f:
        json.dump(global_config, f, indent=2, ensure_ascii=False)

    all_summaries = {}
    for m in MODELS_CONFIG:
        summary_path = OUTPUT_DIR / m["id"] / "summary.json"
        if args.skip_existing and summary_path.exists():
            print(f"Skipping existing: {m['id']}")
            with summary_path.open("r", encoding="utf-8") as f:
                all_summaries[m["id"]] = json.load(f)
            continue

        summary = run_benchmark_for_model(m, batch_size=args.batch_size)
        all_summaries[m["id"]] = summary

        # Write intermediate combined summary
        with (OUTPUT_DIR / "summary.json").open("w", encoding="utf-8") as f:
            json.dump(all_summaries, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 80)
    print("ALL BENCHMARKS COMPLETED SUCCESSFULLY!")
    print(f"Summaries saved to: {OUTPUT_DIR / 'summary.json'}")
    print("=" * 80)


if __name__ == "__main__":
    main()
