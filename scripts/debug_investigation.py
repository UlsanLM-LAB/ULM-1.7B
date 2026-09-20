"""ULM-1.7B Deep Diagnostic and Comparative Evaluation Script.

Performs:
1. Debug dump of exact prompt/chat_template applied to '게이야'
2. A/B/C Comparative test on 16 test sentences:
   - A: Qwen/Qwen3-1.7B base
   - B: ULM Stage 1 (Base + Stage 1 LoRA)
   - C: ULM Phase 2 merged
3. Prompt Minimization test (No system vs Minimal system vs Web system)
4. Generic response repetition and diversity analysis on 35 distinct prompts
5. Saves all results to reports/investigation_results.json
"""

from __future__ import annotations

import json
import random
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

ADDITIONAL_PROMPTS = [
    "지구는 왜 둥글어?",
    "울산 12경이 뭐야?",
    "퇴근하고 영화 보러 갈래?",
    "내일 비 온다던데 우산 챙길까?",
    "요즘 재미있는 드라마 추천해줘",
    "자전거 타기 좋은 곳 추천해줘",
    "간절곶 가봤나?",
    "커피 한잔 할까?",
    "감기 걸린 것 같아",
    "주말에 부산 가려고 하는데 어때?",
    "고양이 키우고 싶다",
    "컴퓨터 살 때 뭐 봐야 해?",
    "짜장면이 좋아 짬뽕이 좋아?",
    "시간이 왜 이렇게 빨리 가지?",
    "일하기 너무 싫다",
    "울산역에서 삼산동 어떻게 가?",
    "고래고기 먹어봤어?",
    "비빔밥 맛있게 만드는 법",
    "방언과 사투리의 차이가 뭐야?",
]  # Total 16 + 19 = 35 prompts

GENERIC_PHRASES = [
    "궁금해하신 내용",
    "성심성의껏",
    "편하게 물어보",
    "ULM은",
    "표준어로 물어보셔도",
    "자연스러운 울산말",
    "답변해 드립니다",
]

BASE_MODEL_NAME = "Qwen/Qwen3-1.7B"
STAGE1_ADAPTER = "outputs/qwen3-1.7b-sft-dense-t4-1epoch"
PHASE2_MERGED = "outputs/ulm-1.7b-phase2-merged"

DEFAULT_SYSTEM_PROMPT = "울산 지역어 대화 assistant로서 자연스럽고 일상적인 울산 사투리로 상대방과 친근하게 대화한다."
MINIMAL_SYSTEM_PROMPT = "사용자의 질문에 직접 답한다. 자연스러운 울산말을 사용하되 의미와 정확성을 우선한다."
WEB_SYSTEM_PROMPT = (
    "울산 지역어 대화 assistant로서 자연스럽고 일상적인 울산 사투리로 상대방과 친근하게 대화한다. "
    "자연스럽고 편안한 일상 울산 말투를 기본으로 구사한다."
)


def generate_reply(
    model: Any,
    tokenizer: Any,
    messages: list[dict[str, str]],
    *,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_new_tokens: int = 128,
    seed: int = 42,
) -> str:
    set_seed(seed)
    template_kwargs: dict[str, Any] = {
        "tokenize": False,
        "add_generation_prompt": True,
        "enable_thinking": False,
    }
    try:
        prompt = tokenizer.apply_chat_template(messages, **template_kwargs)
    except TypeError:
        template_kwargs.pop("enable_thinking", None)
        prompt = tokenizer.apply_chat_template(messages, **template_kwargs)

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
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

    # If <think> tag was output, strip reasoning
    if "<think>" in reply and "</think>" in reply:
        reply = reply.split("</think>", 1)[1].strip()
    elif "<think>" in reply:
        reply = reply.split("<think>", 1)[0].strip()

    return reply


def main():
    print("=== ULM-1.7B Deep Diagnostic & Investigation ===")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)

    # 1. Inspect Tokenizer and Chat Template on "게이야"
    tokenizer = AutoTokenizer.from_pretrained(PHASE2_MERGED, use_fast=True)
    sample_msgs = [
        {"role": "system", "content": WEB_SYSTEM_PROMPT},
        {"role": "user", "content": "게이야"},
    ]
    raw_prompt = tokenizer.apply_chat_template(
        sample_msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    print("\n[1] Debug Final Prompt for '게이야':")
    print("-" * 50)
    print(raw_prompt)
    print("-" * 50)

    results: dict[str, Any] = {
        "debug_prompt_geiya": raw_prompt,
        "models_compared": {},
        "prompt_minimization": {},
        "generic_repetition_check": {},
    }

    # 2. Test Phase 2 Merged Model
    print(f"\n[2] Loading Phase 2 Merged Model ({PHASE2_MERGED})...")
    p2_model = AutoModelForCausalLM.from_pretrained(
        PHASE2_MERGED,
        torch_dtype=torch.bfloat16,
        device_map=device,
    ).eval()

    p2_replies = []
    print("Running Phase 2 on 16 test prompts...")
    for q in TEST_PROMPTS_16:
        msgs = [
            {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
            {"role": "user", "content": q},
        ]
        ans = generate_reply(p2_model, tokenizer, msgs)
        p2_replies.append({"prompt": q, "response": ans})
        print(f"Q: {q}\nA: {ans}\n")
    results["models_compared"]["Phase2_merged"] = p2_replies

    # Prompt Minimization on Phase 2
    print("\n[3] Running Prompt Minimization on Phase 2...")
    prompt_min_res = []
    for q in ["게이야", "너 뭐하노?", "오늘 학교 끝나고 뭐할까?", "1+1은 뭐야?"]:
        # A. No system
        ans_no_sys = generate_reply(
            p2_model, tokenizer, [{"role": "user", "content": q}]
        )
        # B. Minimal system
        ans_minimal = generate_reply(
            p2_model,
            tokenizer,
            [{"role": "system", "content": MINIMAL_SYSTEM_PROMPT}, {"role": "user", "content": q}],
        )
        # C. Web system
        ans_web = generate_reply(
            p2_model,
            tokenizer,
            [{"role": "system", "content": WEB_SYSTEM_PROMPT}, {"role": "user", "content": q}],
        )
        prompt_min_res.append({
            "prompt": q,
            "no_system": ans_no_sys,
            "minimal_system": ans_minimal,
            "web_system": ans_web,
        })
    results["prompt_minimization"] = prompt_min_res

    # Generic Response Check on 35 Prompts with Phase 2
    print("\n[4] Running Generic Repetition Check on 35 prompts...")
    all_35 = TEST_PROMPTS_16 + ADDITIONAL_PROMPTS
    generic_occurrences: dict[str, int] = {k: 0 for k in GENERIC_PHRASES}
    all_35_replies = []

    for q in all_35:
        msgs = [
            {"role": "system", "content": WEB_SYSTEM_PROMPT},
            {"role": "user", "content": q},
        ]
        ans = generate_reply(p2_model, tokenizer, msgs)
        all_35_replies.append({"prompt": q, "response": ans})
        for phrase in GENERIC_PHRASES:
            if phrase in ans:
                generic_occurrences[phrase] += 1

    results["generic_repetition_check"] = {
        "phrase_counts": generic_occurrences,
        "sample_responses": all_35_replies[:10],
    }

    # Free Phase 2 model
    del p2_model
    torch.cuda.empty_cache()

    # 3. Test Base Model (Qwen/Qwen3-1.7B)
    print(f"\n[5] Loading Base Model ({BASE_MODEL_NAME})...")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map=device,
    ).eval()

    base_replies = []
    print("Running Base model on 16 test prompts...")
    for q in TEST_PROMPTS_16:
        msgs = [
            {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
            {"role": "user", "content": q},
        ]
        ans = generate_reply(base_model, tokenizer, msgs)
        base_replies.append({"prompt": q, "response": ans})
        print(f"Q: {q}\nA: {ans}\n")
    results["models_compared"]["Base_Qwen3_1.7B"] = base_replies

    # 4. Test Stage 1 Model (Base + Stage 1 LoRA)
    print(f"\n[6] Loading Stage 1 Adapter ({STAGE1_ADAPTER})...")
    s1_model = PeftModel.from_pretrained(base_model, STAGE1_ADAPTER).eval()

    s1_replies = []
    print("Running Stage 1 model on 16 test prompts...")
    for q in TEST_PROMPTS_16:
        msgs = [
            {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
            {"role": "user", "content": q},
        ]
        ans = generate_reply(s1_model, tokenizer, msgs)
        s1_replies.append({"prompt": q, "response": ans})
        print(f"Q: {q}\nA: {ans}\n")
    results["models_compared"]["Stage1_LoRA"] = s1_replies

    # Save all results to JSON
    out_file = reports_dir / "investigation_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n=== Investigation Complete! Results written to {out_file} ===")


if __name__ == "__main__":
    main()
