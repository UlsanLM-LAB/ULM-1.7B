#!/usr/bin/env python3
"""Stage 1 vs Phase 2 comparative inference test."""

from __future__ import annotations

import json
from pathlib import Path
from ulm.inference.cli import generate_text

TEST_SENTENCES = [
    "오늘 학교 끝나고 뭐 할 거야?",
    "지금 어디 가고 있어?",
    "밥 먹었어?",
    "왜 이렇게 늦게 왔어?",
    "오늘 날씨가 정말 덥네.",
    "너 지금 뭐 하고 있니?",
    "그러니까 내가 아까 말했잖아.",
    "많이 먹었네.",
    "어떻게 그렇게 빨리 왔어?",
    "그거 아니지 않냐?",
]

STAGE1_BASE = "Qwen/Qwen3-1.7B"
STAGE1_ADAPTER = "outputs/qwen3-1.7b-sft-dense-t4-1epoch"

PHASE2_BASE = "outputs/ulm-1.7b-stage1-merged"
PHASE2_ADAPTER = "outputs/ulm-1.7b-phase2-l40s"


def run_comparison():
    results = []

    print("=== Running Stage 1 vs Phase 2 Comparison Tests ===")

    # 1. 10 sentences with default strength 2
    for idx, sentence in enumerate(TEST_SENTENCES, start=1):
        print(f"[{idx}/10] Testing (strength=2): {sentence}")

        s1_out = generate_text(
            STAGE1_BASE,
            sentence,
            adapter_path=STAGE1_ADAPTER,
            dialect_strength=2,
            do_sample=False,
            load_in_4bit=False,
        )

        p2_out = generate_text(
            PHASE2_BASE,
            sentence,
            adapter_path=PHASE2_ADAPTER,
            dialect_strength=2,
            do_sample=False,
            load_in_4bit=False,
        )

        results.append({
            "idx": idx,
            "input": sentence,
            "strength": 2,
            "stage1": s1_out,
            "phase2": p2_out,
        })

    # 2. Test strength 1, 2, 3 on selected representative sentences
    rep_sentences = [
        "오늘 학교 끝나고 뭐 할 거야?",
        "지금 어디 가고 있어?",
        "왜 이렇게 늦게 왔어?",
    ]
    strength_results = []
    for s in rep_sentences:
        for strength in [1, 2, 3]:
            print(f"Testing strength={strength} for: {s}")
            s1_out = generate_text(
                STAGE1_BASE,
                s,
                adapter_path=STAGE1_ADAPTER,
                dialect_strength=strength,
                do_sample=False,
                load_in_4bit=False,
            )
            p2_out = generate_text(
                PHASE2_BASE,
                s,
                adapter_path=PHASE2_ADAPTER,
                dialect_strength=strength,
                do_sample=False,
                load_in_4bit=False,
            )
            strength_results.append({
                "input": s,
                "strength": strength,
                "stage1": s1_out,
                "phase2": p2_out,
            })

    output_data = {
        "sentences_10": results,
        "strength_comparison": strength_results,
    }

    out_path = Path("reports/phase2_comparison_results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output_data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved comparison results to {out_path}")


if __name__ == "__main__":
    run_comparison()
