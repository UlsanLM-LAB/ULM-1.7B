"""Detached recovery: one pilot, conditional fresh run, selection and final checks."""

import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = Path("data/ulsan_dialect_phase3_v3_recovery_v2")
REPORTS = Path("reports/phase3-v3/recovery-v2")
OUTPUTS = Path("outputs/ulm-4b-phase3-v3-recovery-v2")
NVME = Path("/opt/dlami/nvme/phase3-v3-recovery-v2")
PYTHON = sys.executable
LR = 7e-6


def run(*args):
    print("RUN", *map(str, args), flush=True)
    subprocess.run([PYTHON, *map(str, args)], check=True, cwd=ROOT)


def good(summary):
    return (summary["factual_qa"]["accuracy_pct"] >= 83.3
            and summary["multi_turn"]["accuracy_pct"] >= 87.5
            and summary["instruction_trap"]["accuracy_pct"] == 100.0)


def main():
    REPORTS.mkdir(parents=True, exist_ok=False)
    OUTPUTS.mkdir(parents=True, exist_ok=False)
    if NVME.exists():
        raise FileExistsError(NVME)
    NVME.mkdir(parents=True)
    (REPORTS / "config.json").write_text(json.dumps({"base": "/home/ubuntu/models/Qwen3.8-4B-Distill",
        "lr": LR, "pilot_steps": 30, "full_gate_steps": [70, 140], "effective_batch": 32,
        "completion_only_loss": True, "train": str(DATA / "train.jsonl")}, indent=2) + "\n")
    run("scripts/build_phase3_v3_recovery_v2_dataset.py")

    import run_pilot_ablations as pilot
    pilot.NVME_PILOTS_DIR = NVME / "pilots"
    pilot.REPORTS_DIR = REPORTS
    pilot.NVME_PILOTS_DIR.mkdir(parents=True)
    summary = pilot.run_single_pilot("pilot30", str(DATA / "train.jsonl"),
        str(DATA / "validation.jsonl"), LR, max_steps=30)
    (REPORTS / "pilot-summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    baseline = json.loads(Path("reports/phase3-v3/base-gate-40-summary.json").read_text())
    if not good(summary) or summary["dialect_eval"]["accuracy_pct"] <= baseline["dialect_eval"]["accuracy_pct"]:
        (REPORTS / "decision.json").write_text(json.dumps({"decision": "stop_after_pilot", "reason": "dialect or preservation gate failed"}) + "\n")
        print("Pilot failed: stopping before full run", flush=True)
        return

    shutil.copy2("reports/phase3-v3/base-gate-40-summary.json", REPORTS / "base-gate-40-summary.json")
    run("scripts/train_phase3_v3.py", "--model", "/home/ubuntu/models/Qwen3.8-4B-Distill",
        "--train", DATA / "train.jsonl", "--validation", DATA / "validation.jsonl",
        "--gate", "reports/phase3-v3/gate-40.jsonl", "--nvme", NVME / "full",
        "--ebs", OUTPUTS, "--reports", REPORTS, "--micro-batch", "8",
        "--gradient-checkpointing", "--lr", str(LR), "--max-steps", "140")
    gates = [json.loads(p.read_text()) for p in REPORTS.glob("gate-step-*.json")]
    candidates = [(summary["dialect_eval"]["accuracy_pct"],
                   sum(summary[k]["accuracy_pct"] for k in ("factual_qa", "multi_turn", "instruction_trap")),
                   NVME / "pilots/pilot30/final_adapter", "pilot30")]
    for row in gates:
        if good(row):
            step = row["global_step"]
            adapter = OUTPUTS / f"best-checkpoint-{step}"
            if adapter.exists():
                candidates.append((row["dialect_eval"]["accuracy_pct"],
                    sum(row[k]["accuracy_pct"] for k in ("factual_qa", "multi_turn", "instruction_trap")),
                    adapter, f"step{step}"))
    selected = max(candidates, key=lambda row: row[:2])
    (REPORTS / "selection.json").write_text(json.dumps({"selected": selected[3], "adapter": str(selected[2]),
        "gate_dialect_pct": selected[0]}, indent=2) + "\n")
    merged = OUTPUTS / "selected-merged-fp32"
    run("scripts/merge_and_verify_phase3_v2.py", "--base-model-path", "/home/ubuntu/models/Qwen3.8-4B-Distill",
        "--adapter-path", selected[2], "--merged-output-path", merged,
        "--parity-output", REPORTS / "merge-parity.json", "--dtype", "fp32")
    for name, benchmark in (("regression150", "data/regression_benchmark_150.jsonl"),
                            ("dialect50", "reports/phase3-v3/final-dialect-holdout-50.jsonl")):
        run("scripts/evaluate_preservation.py", "--model-path", merged, "--benchmark-path", benchmark,
            "--output-jsonl", REPORTS / f"{name}.jsonl", "--output-summary", REPORTS / f"{name}-summary.json",
            "--dtype", "fp32")
    (REPORTS / "decision.json").write_text(json.dumps({"decision": "complete", "selected": selected[3]}) + "\n")
    print("Recovery v2 complete", flush=True)


if __name__ == "__main__":
    main()
