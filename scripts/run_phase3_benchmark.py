"""Phase 3 L40S 20-Step Benchmark Runner.

Benchmarks Candidate A (b8, ga2, no-gc) vs Candidate B (b16, ga1, no-gc).
Measures:
- 20-step runtime
- sec/step
- steps/sec
- peak VRAM
- average GPU utilization
- max temperature
- max power
- train loss
- eval loss
- OOM 여부
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from pathlib import Path


def run_benchmark_candidate(name: str, config_path: str) -> dict[str, any]:
    print(f"\n==================================================")
    print(f"  Running Benchmark: {name} ({config_path})")
    print(f"==================================================")

    # Clean existing output dir
    cfg_name = Path(config_path).stem
    out_dir = Path("outputs") / f"ulm-1.7b-phase3-bench-{name.lower()}"
    if out_dir.exists():
        shutil.rmtree(out_dir)

    log_file = Path(f"/tmp/gpu_mon_{name}.csv")
    if log_file.exists():
        log_file.unlink()

    # Start GPU monitor (500ms intervals)
    mon_cmd = [
        "nvidia-smi",
        "--query-gpu=utilization.gpu,memory.used,temperature.gpu,power.draw",
        "--format=csv,noheader,nounits",
        "-lms",
        "500",
    ]
    with open(log_file, "w") as mf:
        mon_proc = subprocess.Popen(mon_cmd, stdout=mf, stderr=subprocess.DEVNULL)

    train_cmd = [
        "uv",
        "run",
        "ulm-train-sft",
        "--config",
        config_path,
    ]
    env = os.environ.copy()
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    t0 = time.time()
    oom = False
    train_loss = None
    eval_loss = None
    stdout_lines = []

    try:
        proc = subprocess.run(
            train_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
            check=True,
        )
        stdout_lines = proc.stdout.splitlines()
    except subprocess.CalledProcessError as e:
        stdout_lines = e.stdout.splitlines() if e.stdout else []
        if "out of memory" in (e.stdout or "").lower() or "cuda oom" in (e.stdout or "").lower():
            oom = True
            print(f"[{name}] OOM detected!")
        else:
            print(f"[{name}] Training failed with exit code {e.returncode}")

    t1 = time.time()
    runtime = t1 - t0

    # Stop GPU monitor
    mon_proc.terminate()
    try:
        mon_proc.wait(timeout=2)
    except subprocess.TimeoutExpired:
        mon_proc.kill()

    # Parse GPU monitor metrics
    utils = []
    vrams = []
    temps = []
    powers = []

    if log_file.exists():
        with open(log_file, "r") as mf:
            for line in mf:
                parts = [p.strip() for p in line.strip().split(",")]
                if len(parts) >= 4:
                    try:
                        u = float(parts[0])
                        v = float(parts[1])
                        t = float(parts[2])
                        p = float(parts[3])
                        utils.append(u)
                        vrams.append(v)
                        temps.append(t)
                        powers.append(p)
                    except ValueError:
                        continue

    peak_vram = max(vrams) if vrams else 0.0
    avg_util = sum(utils) / max(len(utils), 1) if utils else 0.0
    max_temp = max(temps) if temps else 0.0
    max_power = max(powers) if powers else 0.0

    # Parse losses from stdout
    for line in stdout_lines:
        if "'train_loss':" in line:
            # e.g. {'train_loss': '2.145', ...}
            try:
                part = line.split("'train_loss':", 1)[1].split(",", 1)[0].strip().strip("'\"")
                train_loss = float(part)
            except Exception:
                pass
        if "'eval_loss':" in line:
            try:
                part = line.split("'eval_loss':", 1)[1].split(",", 1)[0].strip().strip("'\"")
                eval_loss = float(part)
            except Exception:
                pass

    sec_per_step = runtime / 20.0 if not oom else None
    steps_per_sec = 20.0 / runtime if not oom and runtime > 0 else None

    result = {
        "candidate": name,
        "config_path": config_path,
        "oom": oom,
        "runtime_seconds": round(runtime, 2),
        "sec_per_step": round(sec_per_step, 3) if sec_per_step else None,
        "steps_per_sec": round(steps_per_sec, 3) if steps_per_sec else None,
        "peak_vram_mib": round(peak_vram, 1),
        "peak_vram_gib": round(peak_vram / 1024.0, 2),
        "avg_gpu_util_percent": round(avg_util, 1),
        "max_temperature_c": round(max_temp, 1),
        "max_power_w": round(max_power, 1),
        "train_loss": train_loss,
        "eval_loss": eval_loss,
    }

    print(f"[{name}] Result: {json.dumps(result, indent=2)}")
    return result


def main():
    results = {}
    # Candidate A: batch 8, ga 2
    res_a = run_benchmark_candidate("Candidate_A", "configs/sft/qwen3_1.7b_phase3_bench_cand_a.yaml")
    results["Candidate_A"] = res_a

    # Candidate B: batch 16, ga 1
    res_b = run_benchmark_candidate("Candidate_B", "configs/sft/qwen3_1.7b_phase3_bench_cand_b.yaml")
    results["Candidate_B"] = res_b

    out_path = Path("reports/phase3_l40s_benchmark_results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\nBenchmark completed! Saved to {out_path}")


if __name__ == "__main__":
    main()
