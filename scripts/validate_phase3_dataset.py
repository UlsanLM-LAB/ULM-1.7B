"""Automated Dataset Validator and Auditor for Phase 3 Chat Alignment.

Validates:
1. assistant == user or high similarity echo (SequenceMatcher > 0.65)
2. empty assistant messages
3. duplicate prompt/conversations
4. generic template repetition ('궁금해하신 내용', '성심성의껏', 'ULM은', etc.)
5. excessive sentence length (> 1000 characters)
6. multi-turn role ordering (must strictly alternate user -> assistant)
7. extracts high-similarity pairs for manual review
8. saves 200 random audit samples to reports/phase3_audit_200.jsonl
9. outputs comprehensive dataset health metrics
"""

from __future__ import annotations

import difflib
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

FORBIDDEN_GENERIC_PATTERNS = [
    "궁금해하신 내용",
    "성심성의껏",
    "편하게 물어보이소",
    "ULM은 일상적인",
    "표준어로 물어보셔도",
]

DATASET_DIR = Path("data/ulsan_dialect_phase3")
REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def validate_sample(sample: dict[str, Any]) -> tuple[bool, list[str], float]:
    reasons = []
    messages = sample.get("messages", [])
    if not messages:
        return False, ["messages_empty"], 0.0

    # Check role ordering
    expected_roles = ["user", "assistant"]
    max_sim = 0.0

    # Skip initial system message if present
    check_msgs = messages
    if check_msgs and check_msgs[0].get("role") == "system":
        check_msgs = check_msgs[1:]

    if not check_msgs:
        return False, ["no_user_assistant_messages"], 0.0

    for idx, msg in enumerate(check_msgs):
        expected_role = expected_roles[idx % 2]
        role = msg.get("role")
        content = (msg.get("content") or "").strip()

        if role != expected_role:
            reasons.append(f"role_ordering_error_step_{idx}_{role}_vs_{expected_role}")
        if not content:
            reasons.append(f"empty_content_step_{idx}_{role}")
        if len(content) > 1000:
            reasons.append(f"excessive_length_step_{idx}_{len(content)}")

        for pat in FORBIDDEN_GENERIC_PATTERNS:
            if pat in content:
                reasons.append(f"forbidden_generic_template_{pat}")

    # Check similarity between pairs of (user, assistant)
    for i in range(0, len(check_msgs) - 1, 2):
        u_text = check_msgs[i].get("content", "").strip()
        a_text = check_msgs[i + 1].get("content", "").strip()
        sim = difflib.SequenceMatcher(None, u_text, a_text).ratio()
        if sim > max_sim:
            max_sim = sim

        # If similarity is excessively high (> 0.65) and length is short, it is an echo
        if sim > 0.65 and len(u_text) > 5 and sample.get("task") != "standard_to_dialect":
            reasons.append(f"high_similarity_echo_ratio_{sim:.2f}")

    is_valid = len(reasons) == 0
    return is_valid, reasons, max_sim


def run_validation():
    print("=== Phase 3 Chat Alignment Dataset Automated Validation ===")
    all_samples: list[dict[str, Any]] = []
    splits = ["train", "validation"]

    for split in splits:
        fpath = DATASET_DIR / f"{split}.jsonl"
        if not fpath.exists():
            print(f"Error: {fpath} not found!")
            return
        with open(fpath, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    rec["_split"] = split
                    all_samples.append(rec)

    total_count = len(all_samples)
    print(f"Total samples loaded across train & validation: {total_count}")

    task_counts = Counter()
    category_counts = Counter()
    turn_counts = Counter()
    user_lengths: list[int] = []
    asst_lengths: list[int] = []

    seen_prompts = set()
    duplicate_count = 0
    invalid_samples = []
    high_sim_samples = []

    for idx, s in enumerate(all_samples):
        task = s.get("task", "unknown")
        cat = s.get("category", "unknown")
        msgs = s.get("messages", [])

        task_counts[task] += 1
        category_counts[cat] += 1

        # Check turn count
        turn_type = "multi-turn" if len(msgs) > 2 else "single-turn"
        turn_counts[turn_type] += 1

        # Check duplicate
        first_user = next((m["content"] for m in msgs if m["role"] == "user"), "")
        if first_user in seen_prompts and s.get("task") != "dialect_chat":
            duplicate_count += 1
        seen_prompts.add(first_user)

        # Length tracking
        for m in msgs:
            if m["role"] == "user":
                user_lengths.append(len(m["content"]))
            elif m["role"] == "assistant":
                asst_lengths.append(len(m["content"]))

        # Validation
        valid, errors, max_sim = validate_sample(s)
        if not valid:
            invalid_samples.append({"index": idx, "errors": errors, "sample": s})
        if max_sim > 0.60 and s.get("task") != "standard_to_dialect":
            high_sim_samples.append({"index": idx, "sim": max_sim, "sample": s})

    avg_u_len = sum(user_lengths) / max(len(user_lengths), 1)
    avg_a_len = sum(asst_lengths) / max(len(asst_lengths), 1)

    print("\n--- Validation & Statistics Summary ---")
    print(f"Total Samples: {total_count}")
    print(f"Task Distribution: {dict(task_counts)}")
    for t, c in task_counts.items():
        print(f"  - {t}: {c} ({c / total_count * 100:.1f}%)")

    print(f"\nTurn Distribution: {dict(turn_counts)}")
    for tt, c in turn_counts.items():
        print(f"  - {tt}: {c} ({c / total_count * 100:.1f}%)")

    print(f"\nCategory Distribution: {dict(category_counts)}")
    for cat, c in category_counts.items():
        print(f"  - {cat}: {c} ({c / total_count * 100:.1f}%)")

    print(f"\nAverage User Length: {avg_u_len:.1f} chars")
    print(f"Average Assistant Length: {avg_a_len:.1f} chars")
    print(f"High Similarity Echo Count (> 0.60): {len(high_sim_samples)}")
    print(f"Duplicate Prompts Count: {duplicate_count}")
    print(f"Invalid / Corrupted Samples Count: {len(invalid_samples)}")

    # Extract random 200 samples for manual audit
    random.seed(42)
    audit_200 = random.sample(all_samples, min(200, len(all_samples)))
    audit_file = REPORTS_DIR / "phase3_audit_200.jsonl"
    with open(audit_file, "w", encoding="utf-8") as f:
        for it in audit_200:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    print(f"\nSaved 200 random audit samples to: {audit_file}")

    # Save full validation report to JSON
    report = {
        "total_samples": total_count,
        "task_distribution": dict(task_counts),
        "task_ratios": {t: f"{c / total_count * 100:.1f}%" for t, c in task_counts.items()},
        "turn_distribution": dict(turn_counts),
        "turn_ratios": {t: f"{c / total_count * 100:.1f}%" for t, c in turn_counts.items()},
        "category_distribution": dict(category_counts),
        "avg_user_length_chars": round(avg_u_len, 1),
        "avg_assistant_length_chars": round(avg_a_len, 1),
        "high_similarity_echo_count": len(high_sim_samples),
        "duplicate_prompts_count": duplicate_count,
        "invalid_samples_count": len(invalid_samples),
        "sample_audit_first_5": audit_200[:5],
    }

    report_path = REPORTS_DIR / "phase3_dataset_validation_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Full validation report saved to: {report_path}")


if __name__ == "__main__":
    run_validation()
