"""Evaluation script for Phase 3 Full Training on AWS L40S.

Evaluates:
1. Epoch 1 checkpoint (outputs/ulm-1.7b-phase3-chat-l40s/checkpoint-537)
2. Epoch 2 / Final checkpoint (outputs/ulm-1.7b-phase3-chat-l40s)
Across 20 mandatory evaluation prompts.

3. Dialect strength control preservation test:
- '너 지금 뭐 하고 있니?' (strength 1, 2, 3)
- '왜 이렇게 늦게 왔어?' (strength 1, 2, 3)
- '오늘 학교 끝나고 뭐 할 거야?' (strength 1, 2, 3)
Across Phase 2 vs Phase 3.

Saves results to reports/phase3_full_eval_results.json
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

TEST_PROMPTS_20 = [
    "게이야",
    "너 뭐하노?",
    "오늘 학교 끝나고 뭐할까?",
    "배고픈데 뭐 먹지?",
    "울산에서 놀러갈 만한 데 있어?",
    "너 이름이 뭐야?",
    "1+1은 뭐야?",
    "12 * 13은 얼마야?",
    "파이썬으로 리스트 정렬하는 법 알려줘",
    "파이썬 딕셔너리가 뭐야?",
    "오늘 기분이 별로다",
    "내일 시험인데 공부하기 싫다",
    "왜 하늘은 파래?",
    "물이 100도에서 왜 끓어?",
    "아니 그게 무슨 말이야?",
    "ㅋㅋㅋㅋ",
    "뭐라고?",
    "안녕",
    "너 울산 사람이야?",
    "서울에서 부산까지 KTX 타면 얼마나 걸려?",
]

STRENGTH_TEST_SENTENCES = [
    "너 지금 뭐 하고 있니?",
    "왜 이렇게 늦게 왔어?",
    "오늘 학교 끝나고 뭐 할 거야?",
]

SYSTEM_PROMPT = "울산 지역어 대화 assistant로서 자연스럽고 일상적인 울산 사투리로 상대방과 친근하게 대화한다."
BASE_MODEL = "outputs/ulm-1.7b-phase2-merged"
CKPT_EPOCH1 = "outputs/ulm-1.7b-phase3-chat-l40s/checkpoint-537"
CKPT_FINAL = "outputs/ulm-1.7b-phase3-chat-l40s"


def generate_text(model: Any, tokenizer: Any, messages: list[dict[str, str]], seed: int = 42) -> str:
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
            max_new_tokens=128,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
        )
    in_len = inputs["input_ids"].shape[-1]
    reply = tokenizer.decode(outputs[0][in_len:], skip_special_tokens=True).strip()

    if "<think>" in reply and "</think>" in reply:
        reply = reply.split("</think>", 1)[1].strip()
    elif "<think>" in reply:
        reply = reply.split("<think>", 1)[0].strip()

    return reply


def evaluate_model_on_20(model: Any, tokenizer: Any, tag: str) -> list[dict[str, str]]:
    print(f"\nEvaluating {tag} on 20 prompts...")
    results = []
    for q in TEST_PROMPTS_20:
        msgs = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": q}]
        ans = generate_text(model, tokenizer, msgs)
        results.append({"prompt": q, "response": ans})
        print(f"[{tag}] Q: {q} -> A: {ans[:60]}...")
    return results


def evaluate_strength_control(model: Any, tokenizer: Any, tag: str) -> list[dict[str, Any]]:
    print(f"\nEvaluating Strength Control on {tag}...")
    results = []
    for sent in STRENGTH_TEST_SENTENCES:
        sent_res = {"sentence": sent, "strengths": {}}
        for s in (1, 2, 3):
            user_content = f"다음 문장을 울산 지역어로 바꿔라.\ndialect_strength: {s}\n입력: {sent}"
            msgs = [
                {"role": "system", "content": "의미를 보존하면서 요청된 강도의 울산 지역어로 표현한다."},
                {"role": "user", "content": user_content},
            ]
            ans = generate_text(model, tokenizer, msgs)
            sent_res["strengths"][f"strength_{s}"] = ans
            print(f"[{tag}] Sent: {sent} (S{s}) -> {ans}")
        results.append(sent_res)
    return results


def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
    all_eval_results: dict[str, Any] = {}

    # 1. Base Phase 2 Model
    print("Loading Base Phase 2 Merged Model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map=device,
    ).eval()

    all_eval_results["strength_phase2"] = evaluate_strength_control(base_model, tokenizer, "Phase2_Base")

    # 2. Epoch 1 Checkpoint
    print(f"\nLoading Epoch 1 Checkpoint from {CKPT_EPOCH1}...")
    epoch1_model = PeftModel.from_pretrained(base_model, CKPT_EPOCH1).eval()
    all_eval_results["epoch1_20_prompts"] = evaluate_model_on_20(epoch1_model, tokenizer, "Epoch1_ckpt537")

    # Unload epoch 1
    del epoch1_model
    torch.cuda.empty_cache()

    # 3. Final / Epoch 2 Checkpoint
    print(f"\nLoading Final / Epoch 2 Checkpoint from {CKPT_FINAL}...")
    final_model = PeftModel.from_pretrained(base_model, CKPT_FINAL).eval()
    all_eval_results["epoch2_20_prompts"] = evaluate_model_on_20(final_model, tokenizer, "Epoch2_Final")
    all_eval_results["strength_phase3_final"] = evaluate_strength_control(final_model, tokenizer, "Phase3_Final")

    out_file = Path("reports/phase3_full_eval_results.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(all_eval_results, f, ensure_ascii=False, indent=2)

    print(f"\n=== Full Evaluation Complete! Results saved to {out_file} ===")


if __name__ == "__main__":
    main()
