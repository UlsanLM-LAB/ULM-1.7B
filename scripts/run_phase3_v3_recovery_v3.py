"""Detached v4 curriculum pilot and conditional fresh-base recovery."""

import json
import shutil
import subprocess
import sys
import traceback
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = "/home/ubuntu/models/Qwen3.8-4B-Distill"
DATA = Path("data/ulsan_dialect_skill_v4")
REPORTS = Path("reports/phase3-v3/recovery-v3")
REPORT = Path("reports/PHASE3_V3_RECOVERY_V3_REPORT.md")
OUTPUTS = Path("outputs/ulm-4b-phase3-v3-recovery-v3")
NVME = Path("/opt/dlami/nvme/phase3-v3-recovery-v3")
INDEPENDENT = Path("reports/phase3-v3/independent-dialect-20.jsonl")
LR = 7e-6


def run(*args):
    print("RUN", *map(str, args), flush=True)
    subprocess.run([sys.executable, *map(str, args)], cwd=ROOT, check=True)


def read_json(path):
    return json.loads(Path(path).read_text())


def evaluate(name, model=BASE, adapter=None, benchmark=INDEPENDENT, dtype="bf16"):
    args = ["scripts/evaluate_preservation.py", "--model-path", model,
            "--benchmark-path", benchmark, "--output-jsonl", REPORTS / f"{name}.jsonl",
            "--output-summary", REPORTS / f"{name}-summary.json", "--dtype", dtype]
    if adapter:
        args += ["--adapter-path", adapter]
    run(*args)
    return read_json(REPORTS / f"{name}-summary.json")


def score(row, category="dialect_eval"):
    return row[category]["accuracy_pct"]


def preservation(row, strict=True):
    floors = (83.0, 85.0, 95.0) if strict else (80.0, 85.0, 95.0)
    return all(score(row, key) >= floor for key, floor in zip(
        ("factual_qa", "multi_turn", "instruction_trap"), floors))


def token_counts():
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(BASE)
    counts = Counter()
    with (DATA / "full-train.jsonl").open() as f:
        for line in f:
            row = json.loads(line)
            counts[row["category"]] += len(tokenizer.encode(row["messages"][-1]["content"], add_special_tokens=False))
    share = 100 * counts["dialect_skill"] / sum(counts.values())
    if not 50 <= share <= 60:
        raise RuntimeError(f"Dialect target-token share outside 50–60%: {share:.1f}%")
    summary = read_json(DATA / "summary.json")
    summary["target_token_counts"] = dict(counts)
    summary["dialect_target_token_share_pct"] = round(share, 1)
    (DATA / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    return summary


def report(state):
    dataset = state.get("dataset", {})
    pilot = state.get("pilot", {})
    pilot20 = state.get("pilot20", {})
    gates = state.get("gates", [])
    final = state.get("final", {})
    def pct(row, key):
        return row.get(key, {}).get("accuracy_pct", "pending")
    lines = ["# Phase3 v3 Recovery v3", "", f"Verdict: **{state.get('verdict', 'RUNNING')}**", "",
        "## Previous dataset failure", "",
        "Recovery v2 used 2,800 dialect samples, all framed as conversion. Its train split had 2,654 such samples,",
        "approximately 1,327 responses presented twice. It had no dedicated meaning, ending-selection, or correction tasks.",
        "`아이가` appeared in 1,590 train targets, while `퍼뜩`, `단디`, `뭇나`, `온나`, `천지빼까리`, and `파이다` were absent.",
        "The v2 step-30 pilot preserved factual/memory/instruction (91.7/87.5/100) but dialect fell to 16.7% from 25% base.", "",
        "## v4 skill dataset", "",
        f"Counts: {dataset.get('skill_counts', 'pending')}. Mixture: {dataset.get('mixture_counts', 'pending')}.",
        f"Dialect target-token share: {dataset.get('dialect_target_token_share_pct', 'pending')}%. ",
        f"Exact benchmark prompt overlap: {dataset.get('benchmark_exact_prompt_overlap', 'pending')}; "
        f"maximum identical response opening: {dataset.get('max_response_opening_count', 'pending')}.",
        "Sources: quality-A, non-synthetic U0/U1 real dialect pairs; curated regional lexicon; capped factual/general and memory/instruction replay.",
        "The dialect corpus is regional Gyeongsang speech usable in Ulsan; rare terms are not claimed to be exclusive to Ulsan.",
        "Limit: some source transcriptions remain conversational fragments; the 20-item independent gate checks transfer across tasks.", "",
        "## Curriculum pilot", "",
        "15 sequential steps at 80/20 skill/replay, then 15 at 55/30/15, from the original base. LR 7e-6; micro 8; accumulation 4; SDPA; completion-only loss.",
        f"Old gate: factual {pct(pilot, 'factual_qa')}%, memory {pct(pilot, 'multi_turn')}%, "
        f"instruction {pct(pilot, 'instruction_trap')}%, dialect {pct(pilot, 'dialect_eval')}%.",
        f"Independent20: base {pct(state.get('base20', {}), 'dialect_eval')}%, pilot {pct(pilot20, 'dialect_eval')}%.",
        f"Pilot decision: {state.get('pilot_decision', 'pending')}.", "", "## Full gates", ""]
    if gates:
        for row in sorted(gates, key=lambda r: r["global_step"]):
            lines.append(f"Step {row['global_step']}: factual {pct(row, 'factual_qa')}%, memory {pct(row, 'multi_turn')}%, "
                         f"instruction {pct(row, 'instruction_trap')}%, old dialect {pct(row, 'dialect_eval')}%, "
                         f"independent20 {pct(row, 'independent20')}%; {row.get('gate_decision', 'unknown')}.")
    else:
        lines.append("No full-run gates reached.")
    lines += ["", "## Final", "",
              f"Selected adapter: {state.get('adapter', 'none')}",
              f"Merged model: {state.get('merged', 'none')}",
              f"Regression150: factual {pct(final.get('regression150', {}), 'factual_qa')}%, "
              f"memory {pct(final.get('regression150', {}), 'multi_turn')}%, "
              f"instruction {pct(final.get('regression150', {}), 'instruction_trap')}%.",
              f"Dialect50: {pct(final.get('dialect50', {}), 'dialect_eval')}%; "
              f"independent20: {pct(final.get('independent20', {}), 'dialect_eval')}%.",
              f"FP32 adapter/merged parity: {final.get('parity', {}).get('parity_matches', 'pending')}/20.",
              f"Final decision: **{state.get('verdict', 'RUNNING')}**. {state.get('reason', '')}", ""]
    REPORT.write_text("\n".join(lines))
    (REPORTS / "state.json").write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n")


def publish():
    # Publish only the small final report and JSON summaries from a clean worktree.
    worktree = Path("/tmp/ulm-phase3-v3-recovery-v3-report")
    if worktree.exists():
        raise FileExistsError(worktree)
    subprocess.run(["git", "fetch", "origin", "main"], cwd=ROOT, check=True)
    subprocess.run(["git", "worktree", "add", "-b", "recovery-v3-auto-report", str(worktree), "origin/main"], cwd=ROOT, check=True)
    target = worktree / REPORT
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(REPORT, target)
    for name in ("state.json", "pilot-summary.json", "pilot-independent-20-summary.json", "base-independent-20-summary.json",
                 "gate-step-70.json", "gate-step-140.json", "gate-step-210.json", "selection.json",
                 "regression150-summary.json", "dialect50-summary.json", "independent20-summary.json", "merge-parity.json"):
        source = REPORTS / name
        if source.exists():
            dest = worktree / source
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, dest)
    subprocess.run(["git", "add", str(REPORT), str(REPORTS)], cwd=worktree, check=True)
    subprocess.run(["git", "commit", "-m", "Report Phase3 v3 recovery v3 result"], cwd=worktree, check=True)
    subprocess.run(["git", "push", "origin", "HEAD:main"], cwd=worktree, check=True)


def main():
    REPORTS.mkdir(parents=True, exist_ok=False)
    OUTPUTS.mkdir(parents=True, exist_ok=False)
    if NVME.exists():
        raise FileExistsError(NVME)
    NVME.mkdir(parents=True)
    state = {"verdict": "RUNNING", "base": BASE, "lr": LR, "pilot_steps": 30}
    try:
        run("scripts/build_phase3_v3_skill_v4.py")
        state["dataset"] = token_counts()
        report(state)
        state["base20"] = evaluate("base-independent-20")
        import run_pilot_ablations as pilot
        pilot.NVME_PILOTS_DIR = NVME / "pilots"
        pilot.REPORTS_DIR = REPORTS
        pilot.NVME_PILOTS_DIR.mkdir(parents=True)
        state["pilot"] = pilot.run_single_pilot("pilot30", str(DATA / "pilot.jsonl"),
            str(DATA / "pilot.jsonl"), LR, max_steps=30, micro_batch=8, grad_accum=4, sequential=True)
        (REPORTS / "pilot-summary.json").write_text(json.dumps(state["pilot"], ensure_ascii=False, indent=2) + "\n")
        pilot_adapter = NVME / "pilots/pilot30/final_adapter"
        state["pilot20"] = evaluate("pilot-independent-20", adapter=pilot_adapter)
        old = score(state["pilot"])
        mini = score(state["pilot20"])
        base_mini = score(state["base20"])
        if not preservation(state["pilot"]):
            state.update(verdict="FAIL", pilot_decision="FAIL", reason="preservation gate failed")
            return
        if old <= 25.0:
            state.update(verdict="FAIL", pilot_decision="FAIL", reason="old dialect gate did not improve")
            return
        if old < 35.0:
            state.update(verdict="PARTIAL", pilot_decision="BORDERLINE_STOP", reason="old dialect gate below 35%; dataset review required")
            return
        if mini <= base_mini:
            state.update(verdict="PARTIAL", pilot_decision="OVERFIT_STOP", reason="independent20 did not improve")
            return
        state["pilot_decision"] = "PASS_TO_FULL"
        report(state)
        shutil.copy2("reports/phase3-v3/base-gate-40-summary.json", REPORTS / "base-gate-40-summary.json")
        run("scripts/train_phase3_v3.py", "--model", BASE,
            "--train", DATA / "full-curriculum.jsonl", "--validation", DATA / "pilot.jsonl",
            "--gate", "reports/phase3-v3/gate-40.jsonl", "--independent-gate", INDEPENDENT,
            "--nvme", NVME / "full", "--ebs", OUTPUTS, "--reports", REPORTS,
            "--micro-batch", "8", "--gradient-checkpointing", "--sequential",
            "--lr", str(LR), "--max-steps", "210")
        state["gates"] = [read_json(path) for path in REPORTS.glob("gate-step-*.json")]
        successes = [r for r in state["gates"] if r.get("gate_decision") == "success"]
        if not successes:
            state.update(verdict="PARTIAL", reason="full run did not reach the success gate")
            return
        chosen = min(successes, key=lambda r: r["global_step"])
        step = chosen["global_step"]
        adapter = OUTPUTS / f"best-checkpoint-{step}"
        if not adapter.exists():
            raise FileNotFoundError(f"successful checkpoint missing: {adapter}")
        state["adapter"] = str(adapter)
        (REPORTS / "selection.json").write_text(json.dumps({"step": step, "adapter": str(adapter)}, indent=2) + "\n")
        merged = OUTPUTS / "selected-merged-fp32"
        run("scripts/merge_and_verify_phase3_v2.py", "--base-model-path", BASE,
            "--adapter-path", adapter, "--merged-output-path", merged,
            "--parity-output", REPORTS / "merge-parity.json", "--dtype", "fp32")
        state["merged"] = str(merged)
        state["final"] = {"parity": read_json(REPORTS / "merge-parity.json")}
        state["final"]["regression150"] = evaluate("regression150", model=merged,
            benchmark="data/regression_benchmark_150.jsonl", dtype="fp32")
        state["final"]["dialect50"] = evaluate("dialect50", model=merged,
            benchmark="reports/phase3-v3/final-dialect-holdout-50.jsonl", dtype="fp32")
        state["final"]["independent20"] = evaluate("independent20", model=merged,
            benchmark=INDEPENDENT, dtype="fp32")
        accepted = (preservation(state["final"]["regression150"], strict=False)
            and score(state["final"]["dialect50"]) >= 60.0
            and score(state["final"]["independent20"]) >= base_mini + 10.0
            and state["final"]["parity"]["parity_pass"])
        state["verdict"] = "PASS" if accepted else "PARTIAL"
        state["reason"] = "final acceptance passed" if accepted else "one or more final acceptance gates failed"
    except Exception as exc:
        state["verdict"] = "FAIL"
        state["reason"] = f"pipeline error: {exc}"
        traceback.print_exc()
    finally:
        report(state)
        try:
            publish()
        except Exception as exc:
            print(f"Report push failed: {exc}", flush=True)
        print(json.dumps({"verdict": state["verdict"], "reason": state.get("reason")}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
