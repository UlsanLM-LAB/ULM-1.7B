"""Merge LoRA adapter into base model and verify parity across 20 test prompts."""

from __future__ import annotations

import argparse
import difflib
import gc
import json
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

PARITY_PROMPTS = [
    # Factual QA (10)
    {"id": 1, "prompt": "대한민국의 수도는 어디야?"},
    {"id": 2, "prompt": "태양계에서 가장 큰 행성은 뭐야?"},
    {"id": 3, "prompt": "한글을 창제한 조선의 국왕은 누구야?"},
    {"id": 4, "prompt": "인류가 최초로 달에 착륙한 연도는 몇 년도야?"},
    {"id": 5, "prompt": "원소 기호 O는 무슨 원소야?"},
    {"id": 6, "prompt": "세계에서 가장 면적이 넓은 국가는 어디야?"},
    {"id": 7, "prompt": "지구에서 가장 높은 산은 뭐야?"},
    {"id": 8, "prompt": "프랑스의 수도는 어디야?"},
    {"id": 9, "prompt": "물 분자의 화학식은 뭐야?"},
    {"id": 10, "prompt": "빛의 진공 속 속도는 초당 약 몇 km야?"},
    # Multi-turn (5)
    {
        "id": 11,
        "prompt": "내 이름이 뭐라고 했지?",
        "history": [["내 이름은 민수야. 기억해 줘.", "네, 민수 님! 기억하고 있겠습니다."]],
    },
    {
        "id": 12,
        "prompt": "내가 어떤 과일을 더 좋아한다고 했지?",
        "history": [["나는 사과보다 바나나를 더 좋아해.", "바나나 달콤하고 영양도 풍부하죠!"]],
    },
    {
        "id": 13,
        "prompt": "내가 언제 어디로 출장 간다고 했어?",
        "history": [["다음 주 화요일에 부산으로 출장 가.", "부산 출장 일정 잘 챙기세요!"]],
    },
    {
        "id": 14,
        "prompt": "내 취미가 뭐라고 했는지 기억나?",
        "history": [
            ["내 취미는 주말마다 자전거 타는 거야.", "자전거 타기는 정말 상쾌한 운동이죠."]
        ],
    },
    {
        "id": 15,
        "prompt": "우리 강아지 이름이 뭐였지?",
        "history": [
            ["우리 집 강아지 이름은 초코야. 갈색 푸들이야.", "초코라는 이름 정말 사랑스럽네요."]
        ],
    },
    # Dialect (5)
    {"id": 16, "prompt": "경상도/울산 방언에서 '단디 해라'가 표준어로 무슨 뜻이야?"},
    {"id": 17, "prompt": "경상도 방언 '와이리 덥노?'를 표준어로 바꾸면 어떤 뜻이야?"},
    {
        "id": 18,
        "prompt": "표준어 문장 '오늘 날씨가 정말 좋다'를 자연스러운 울산/경상 방언으로 바꿔줘.",
    },
    {
        "id": 19,
        "prompt": "표준어 문장 '이 음식 정말 맛있다'를 자연스러운 울산/경상 방언으로 바꿔줘.",
    },
    {
        "id": 20,
        "prompt": "표준어 질문 '너 지금 뭐 하고 있어?'를 자연스러운 울산/경상 방언으로 바꿔줘.",
    },
]


def format_prompt(tokenizer: Any, item: dict) -> str:
    messages = []
    if "history" in item:
        for u, a in item["history"]:
            messages.append({"role": "user", "content": u})
            messages.append({"role": "assistant", "content": a})
    messages.append({"role": "user", "content": item["prompt"]})
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
    except TypeError:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def generate_greedy(
    model: Any, tokenizer: Any, prompt: str, device: str = "cuda:0", max_new_tokens: int = 128
) -> str:
    set_seed(42)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_len = inputs.input_ids.shape[1]
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    gen_text = tokenizer.decode(out[0][input_len:], skip_special_tokens=True)
    return gen_text.strip()


def main():
    parser = argparse.ArgumentParser(description="Merge LoRA and verify parity")
    parser.add_argument("--base-model-path", default="/home/ubuntu/models/Qwen3.8-4B-Distill")
    parser.add_argument(
        "--adapter-path", default="/home/ubuntu/ULM-1.7B/outputs/ulm-4b-phase3-v2-lora"
    )
    parser.add_argument(
        "--merged-output-path", default="/home/ubuntu/ULM-1.7B/outputs/ulm-4b-phase3-v2-best-merged"
    )
    parser.add_argument("--parity-output", default="reports/phase3-v2/merge_parity.json")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    merged_out = Path(args.merged_output_path)
    merged_out.mkdir(parents=True, exist_ok=True)

    print("\n=======================================================")
    print(">>> 1. Loading Base Model + Adapter for Parity Pre-Run <<<")
    print(f"Base: {args.base_model_path}")
    print(f"Adapter: {args.adapter_path}")
    print("=======================================================")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model_path)
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model_path, dtype=torch.bfloat16, device_map=args.device
    )
    peft_model = PeftModel.from_pretrained(base_model, args.adapter_path)
    peft_model.eval()

    adapter_outputs = []
    print("Collecting adapter model outputs across 20 prompts (greedy)...")
    for item in PARITY_PROMPTS:
        p_text = format_prompt(tokenizer, item)
        ans = generate_greedy(peft_model, tokenizer, p_text, device=args.device)
        adapter_outputs.append({"id": item["id"], "prompt": item["prompt"], "answer": ans})

    # Merge model
    print("\n=======================================================")
    print(">>> 2. Merging LoRA into Base Model <<<")
    print("=======================================================")
    merged_model = peft_model.merge_and_unload()
    print("Merge completed! Saving merged model weights...")
    merged_model.save_pretrained(str(merged_out))
    tokenizer.save_pretrained(str(merged_out))
    print(f"Saved merged model to {merged_out}")

    # Reload merged model cleanly from disk to verify standalone loading
    del peft_model
    del merged_model
    del base_model
    torch.cuda.empty_cache()
    gc.collect()

    print("\n=======================================================")
    print(">>> 3. Reloading Merged Standalone Model for Verification <<<")
    print("=======================================================")
    standalone_model = AutoModelForCausalLM.from_pretrained(
        str(merged_out), dtype=torch.bfloat16, device_map=args.device
    )
    standalone_model.eval()

    merged_outputs = []
    parity_matches = 0
    print("Comparing standalone merged model outputs against adapter outputs...")
    for idx, (item, ad_res) in enumerate(zip(PARITY_PROMPTS, adapter_outputs), start=1):
        p_text = format_prompt(tokenizer, item)
        mg_ans = generate_greedy(standalone_model, tokenizer, p_text, device=args.device)
        merged_outputs.append({"id": item["id"], "prompt": item["prompt"], "answer": mg_ans})

        sim = difflib.SequenceMatcher(None, ad_res["answer"], mg_ans).ratio()
        is_match = sim >= 0.95 or ad_res["answer"] == mg_ans
        if is_match:
            parity_matches += 1
        status_str = "MATCH" if is_match else f"MISMATCH (sim={sim:.2f})"
        print(f"  [{idx:02d}/20] {status_str} | Prompt: {item['prompt'][:30]}")

    parity_pct = round(parity_matches / len(PARITY_PROMPTS) * 100, 1)
    parity_pass = parity_matches >= 19  # at least 95% exact/semantic match

    parity_summary = {
        "total_prompts": len(PARITY_PROMPTS),
        "parity_matches": parity_matches,
        "parity_pct": parity_pct,
        "parity_pass": parity_pass,
        "base_model": args.base_model_path,
        "adapter_path": args.adapter_path,
        "merged_path": args.merged_output_path,
    }

    parity_file = Path(args.parity_output)
    parity_file.parent.mkdir(parents=True, exist_ok=True)
    with open(parity_file, "w", encoding="utf-8") as f:
        json.dump(parity_summary, f, indent=2, ensure_ascii=False)

    print("\n=======================================================")
    print(f"Merge Parity Verification Result: {parity_matches}/20 ({parity_pct}%)")
    print(f"Verdict: {'PASS' if parity_pass else 'FAIL'}")
    print(f"Saved: {parity_file}")
    print("=======================================================\n")

    del standalone_model
    torch.cuda.empty_cache()
    gc.collect()


if __name__ == "__main__":
    main()
