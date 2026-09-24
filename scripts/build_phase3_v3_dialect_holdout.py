"""Extend the unchanged 25 regression dialect prompts with 25 held-out v3 prompts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

MARKERS = ("아이가", "단디", "쫌", "억수로", "마이", "데이", "맞나", "하모", "천지빼까리")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def build_holdout(regression: Path, test_data: Path) -> list[dict]:
    existing = [row for row in read_jsonl(regression) if row["category"] == "dialect_eval"]
    if len(existing) != 25:
        raise ValueError(f"Expected 25 established dialect prompts, got {len(existing)}")

    candidates = []
    for sample in read_jsonl(test_data):
        if sample["category"] != "dialect_transformation":
            continue
        messages = sample["messages"]
        if len(messages) != 2 or [msg["role"] for msg in messages] != ["user", "assistant"]:
            continue
        prompt, reference = (msg["content"] for msg in messages)
        markers = [marker for marker in MARKERS if marker in reference and marker not in prompt]
        if not markers:
            continue
        candidates.append(
            {
                "id": f"v3-test-{sample['id']}",
                "category": "dialect_eval",
                "prompt": prompt,
                "dialect_gen": True,
                "markers": markers,
                "source": "phase3_v3_test",
            }
        )
    if len(candidates) < 25:
        raise ValueError(f"Only {len(candidates)} eligible held-out prompts")
    candidates.sort(key=lambda row: hashlib.sha256(row["id"].encode()).digest())
    return existing + candidates[:25]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--regression", default="data/regression_benchmark_150.jsonl")
    parser.add_argument("--test", default="data/ulsan_dialect_phase3_v3/test.jsonl")
    parser.add_argument("--output", default="reports/phase3-v3/final-dialect-holdout-50.jsonl")
    args = parser.parse_args()
    rows = build_holdout(Path(args.regression), Path(args.test))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    print(f"Wrote {len(rows)} prompts to {output}")


if __name__ == "__main__":
    main()
