"""Comparative Evaluation of Base, Phase 2, and Phase 3 Smoke models.

Evaluates the 16 mandatory test sentences across:
- Base: Qwen/Qwen3-1.7B
- Phase 2: outputs/ulm-1.7b-phase2-merged
- Phase 3 Smoke: Phase 2 + outputs/ulm-1.7b-phase3-smoke adapter

Saves results to reports/phase3_smoke_comparison_results.json
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

TEST_PROMPTS_16 = [
    "게이야",
    "너 뭐하노?",
    "오늘 학교 끝나고 뭐할까?",
    "배고픈데 뭐 먹지?",
    "울산에서 놀러갈 만한 데 있어?",
    "너 이름이 뭐야?",
    "1+1은 뭐야?",
    "파이썬으로 리스트 정렬하는 법 알려줘",
    "오늘 기분이 별로다",
    "내일 시험인데 공부하기 싫다",
    "왜 하늘은 파래?",
    "아니 그게 무슨 말이야?",
    "ㅋㅋㅋㅋ",
    "뭐라고?",
    "안녕",
    "너 울산 사람이야?",
]

SYSTEM_PROMPT = "울산 지역어 대화 assistant로서 자연스럽고 일상적인 울산 사투리로 상대방과 친근하게 대화한다."
P2_MODEL_PATH = "outputs/ulm-1.7b-phase2-merged"
P3_SMOKE_ADAPTER = "outputs/ulm-1.7b-phase3-smoke"


def generate_reply(
    model: Any,
    tokenizer: Any,
    prompt_text: str,
    *,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_new_tokens: int = 128,
    seed: int = 42,
) -> str:
    set_seed(seed)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt_text},
    ]
    template_kwargs: dict[str, Any] = {
        "tokenize": False,
        "add_generation_prompt": True,
        "enable_thinking": False,
    }
    try:
        formatted = tokenizer.apply_chat_template(messages, **template_kwargs)
    except TypeError:
        template_kwargs.pop("enable_thinking", None)
        formatted = tokenizer.apply_chat_template(messages, **template_kwargs)

    inputs = tokenizer(formatted, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=max(temperature, 0.01),
            top_p=top_p,
            do_sample=temperature > 0.05,
        )
    in_len = inputs["input_ids"].shape[-1]
    reply = tokenizer.decode(outputs[0][in_len:], skip_special_tokens=True).strip()

    if "<think>" in reply and "</think>" in reply:
        reply = reply.split("</think>", 1)[1].strip()
    elif "<think>" in reply:
        reply = reply.split("<think>", 1)[0].strip()

    return reply


def main():
    print("=== Evaluating Phase 3 Smoke on 16 Test Prompts ===")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(P2_MODEL_PATH, use_fast=True)

    print(f"Loading Base Phase 2 Merged Model from {P2_MODEL_PATH}...")
    base_p2 = AutoModelForCausalLM.from_pretrained(
        P2_MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map=device,
    ).eval()

    print(f"Loading Phase 3 Smoke LoRA from {P3_SMOKE_ADAPTER}...")
    p3_model = PeftModel.from_pretrained(base_p2, P3_SMOKE_ADAPTER).eval()

    p3_results = []
    for q in TEST_PROMPTS_16:
        ans = generate_reply(p3_model, tokenizer, q)
        print(f"\n[Prompt]: {q}\n[Phase 3 Smoke]: {ans}")
        p3_results.append({"prompt": q, "response": ans})

    # Read previous Base & Phase 2 results for full comparison
    prev_investigation = Path("reports/investigation_results.json")
    base_results = []
    p2_results = []
    if prev_investigation.exists():
        with open(prev_investigation, "r", encoding="utf-8") as f:
            d = json.load(f)
            base_results = d.get("models_compared", {}).get("Base_Qwen3_1.7B", [])
            p2_results = d.get("models_compared", {}).get("Phase2_merged", [])

    final_comparison = {
        "prompts_evaluated": len(TEST_PROMPTS_16),
        "phase3_smoke_results": p3_results,
        "phase2_results": p2_results,
        "base_results": base_results,
    }

    out_file = Path("reports/phase3_smoke_comparison_results.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(final_comparison, f, ensure_ascii=False, indent=2)

    print(f"\nComparison results written to {out_file}")


if __name__ == "__main__":
    main()
