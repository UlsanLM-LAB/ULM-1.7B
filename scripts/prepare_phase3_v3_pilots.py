"""Derive fixed 40-prompt gate and 50/35/15 candidate from existing clean data."""

import json
import random
from collections import Counter
from pathlib import Path


ROOT = Path("data/ulsan_dialect_phase3_v3_rebalanced")
OUT = Path("data/ulsan_dialect_phase3_v3_candidate_b")
REPORTS = Path("reports/phase3-v3")


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))


def main():
    rng = random.Random(42)
    source = read_jsonl(ROOT / "train.jsonl") + read_jsonl(ROOT / "validation.jsonl")
    groups = {key: [row for row in source if row["category"] == key] for key in
              ("dialect_sft", "factual_preservation", "instruction_preservation")}
    targets = {"dialect_sft": 2700, "factual_preservation": 1890,
               "instruction_preservation": 810}
    selected = []
    for key, count in targets.items():
        assert len(groups[key]) >= count, (key, len(groups[key]), count)
        rng.shuffle(groups[key])
        selected += groups[key][:count]
    rng.shuffle(selected)
    write_jsonl(OUT / "train.jsonl", selected[:5000])
    write_jsonl(OUT / "validation.jsonl", selected[5000:])
    (OUT / "summary.json").write_text(json.dumps({"total": len(selected),
        "counts": dict(Counter(row["category"] for row in selected))}, indent=2) + "\n")

    gate = read_jsonl(REPORTS / "gate-60.jsonl")
    quotas = {"factual_qa": 12, "multi_turn": 8, "instruction_trap": 8,
              "dialect_eval": 12}
    chosen = []
    for category, count in quotas.items():
        matching = [row for row in gate if row["category"] == category]
        assert len(matching) >= count
        chosen.extend(matching[:count])
    write_jsonl(REPORTS / "gate-40.jsonl", chosen)
    baseline = {row["id"]: row for row in read_jsonl(REPORTS / "base-gate-60.jsonl")}
    hits = {"factual_qa": "factual_hit", "multi_turn": "memory_hit",
            "instruction_trap": "instruction_hit", "dialect_eval": "dialect_hit"}
    summary = {key: {"hits": sum(bool(baseline[row["id"]][hits[key]])
                                 for row in chosen if row["category"] == key),
                     "total": count} for key, count in quotas.items()}
    for value in summary.values():
        value["accuracy_pct"] = round(100 * value["hits"] / value["total"], 1)
    (REPORTS / "base-gate-40-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(summary)


if __name__ == "__main__":
    main()
