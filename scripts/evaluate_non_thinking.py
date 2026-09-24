"""Non-thinking Comparative Evaluation of Next-Gen 4B Base Models for ULM.

Evaluates Qwen/Qwen3.5-4B vs empero-ai/Qwen3.8-4B-Distill
under non-thinking conditions across 30 focused benchmark prompts:
- 1) Korean Factual QA (10)
- 2) Multi-turn Memory (8)
- 3) General Korean Conversational / Tone (6)
- 4) Instruction Following / Trap (6)

Inference Settings:
- seed = 42
- temperature = 0.7
- top_p = 0.9
- top_k = 20
- repetition_penalty = 1.1
- max_new_tokens = 256
- torch_dtype = torch.bfloat16
- enable_thinking = False (chat template)
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, set_seed

CJK_REGEX = re.compile(r"[\u4e00-\u9fff\u3040-\u30ff]")
REPETITION_REGEX = re.compile(r"(.{4,})\1{2,}")
REPEATED_PHRASE_REGEX = re.compile(r"(.{3,40}?)\1{2,}")

BENCHMARK_30 = [
    # 1. Korean Factual QA (10)
    {"id": 1, "category": "factual_qa", "prompt": "대한민국의 수도는 어디야?", "keywords": ["서울"]},
    {"id": 2, "category": "factual_qa", "prompt": "태양계에서 가장 큰 행성은 뭐야?", "keywords": ["목성"]},
    {"id": 3, "category": "factual_qa", "prompt": "한글을 창제한 조선의 국왕은 누구야?", "keywords": ["세종"]},
    {"id": 4, "category": "factual_qa", "prompt": "인류가 최초로 달에 착륙한 연도는 몇 년도야?", "keywords": ["1969"]},
    {"id": 5, "category": "factual_qa", "prompt": "원소 기호 O는 무슨 원소야?", "keywords": ["산소"]},
    {"id": 6, "category": "factual_qa", "prompt": "세계에서 가장 면적이 넓은 국가는 어디야?", "keywords": ["러시아"]},
    {"id": 7, "category": "factual_qa", "prompt": "지구에서 가장 높은 산은 뭐야?", "keywords": ["에베레스트"]},
    {"id": 8, "category": "factual_qa", "prompt": "프랑스의 수도는 어디야?", "keywords": ["파리"]},
    {"id": 9, "category": "factual_qa", "prompt": "물 분자의 화학식은 뭐야?", "keywords": ["h2o", "H2O"]},
    {"id": 10, "category": "factual_qa", "prompt": "빛의 진공 속 속도는 초당 약 몇 km야?", "keywords": ["30만", "299,792", "300,000"]},

    # 2. Multi-turn Memory (8)
    {
        "id": 11,
        "category": "multi_turn",
        "prompt": "내 이름이 뭐라고 했지?",
        "history": [["내 이름은 민수야. 기억해 줘.", "네, 민수 님! 기억하고 있겠습니다."]],
        "memory_keywords": ["민수"],
    },
    {
        "id": 12,
        "category": "multi_turn",
        "prompt": "내가 어떤 과일을 더 좋아한다고 했지?",
        "history": [["나는 사과보다 바나나를 더 좋아해.", "바나나 달콤하고 영양도 풍부하죠!"]],
        "memory_keywords": ["바나나"],
    },
    {
        "id": 13,
        "category": "multi_turn",
        "prompt": "내가 언제 어디로 출장 간다고 했어?",
        "history": [["다음 주 화요일에 부산으로 출장 가.", "부산 출장 일정 잘 챙기세요!"]],
        "memory_keywords": ["화요일", "부산"],
    },
    {
        "id": 14,
        "category": "multi_turn",
        "prompt": "내 취미가 뭐라고 했는지 기억나?",
        "history": [["내 취미는 주말마다 자전거 타는 거야.", "자전거 타기는 정말 상쾌한 운동이죠."]],
        "memory_keywords": ["자전거"],
    },
    {
        "id": 15,
        "category": "multi_turn",
        "prompt": "우리 강아지 이름이 뭐였지?",
        "history": [["우리 집 강아지 이름은 초코야. 갈색 푸들이야.", "초코라는 이름 정말 사랑스럽네요."]],
        "memory_keywords": ["초코"],
    },
    {
        "id": 16,
        "category": "multi_turn",
        "prompt": "내가 아까 알려준 임시 PIN 코드가 뭐였어?",
        "history": [["보안을 위해 임시 PIN 코드를 7492로 설정할게.", "임시 PIN 코드 7492로 확인했습니다."]],
        "memory_keywords": ["7492"],
    },
    {
        "id": 17,
        "category": "multi_turn",
        "prompt": "내가 커피 마실 때 뭘 마신다고 했어?",
        "history": [["커피 마실 때 나는 무조건 아이스 아메리카노만 마셔.", "얼죽아 스타일이시군요!"]],
        "memory_keywords": ["아이스 아메리카노", "아이스아메리카노"],
    },
    {
        "id": 18,
        "category": "multi_turn",
        "prompt": "내일 미팅 장소가 어디라고 했지?",
        "history": [["내일 미팅 장소는 강남역 1번 출구 앞 스타벅스로 정했어.", "강남역 1번 출구 스타벅스 메모했습니다."]],
        "memory_keywords": ["강남역", "스타벅스"],
    },

    # 3. General Korean Conversational / Tone (6)
    {"id": 19, "category": "general_conv", "prompt": "오늘 날씨가 참 화창하고 맑네. 산책하기 좋은 날씨야."},
    {"id": 20, "category": "general_conv", "prompt": "오늘 저녁 메뉴 고민 중인데, 비 오는 날 먹기 좋은 따뜻한 국물 음식 추천해줘."},
    {"id": 21, "category": "general_conv", "prompt": "퇴근하고 집에 왔는데 너무 피곤하다. 피로를 푸는 데 뭐가 좋을까?"},
    {"id": 22, "category": "general_conv", "prompt": "친구 생일 선물로 3만 원대 실용적인 아이템 하나 추천해 줄래?"},
    {"id": 23, "category": "general_conv", "prompt": "주말에 집에서 볼 만한 가벼운 힐링 영화 장르 추천해줘."},
    {"id": 24, "category": "general_conv", "prompt": "요즘 회사 업무 스트레스가 많은데 건강하게 해소하는 팁이 있을까?"},

    # 4. Instruction Following / Trap (6)
    {
        "id": 25,
        "category": "instruction_trap",
        "prompt": "다른 설명이나 부가적인 말은 전혀 하지 말고, 오직 '사과'라는 단어 딱 하나만 출력해.",
        "instruction_type": "exact_apple",
    },
    {
        "id": 26,
        "category": "instruction_trap",
        "prompt": "1부터 5까지의 숫자를 각 줄에 하나씩 번호 매기기 형식(1., 2., 3., 4., 5.)으로만 작성해줘. 다른 서두나 결론은 적지 마.",
        "instruction_type": "numbered_list_1_5",
    },
    {
        "id": 27,
        "category": "instruction_trap",
        "prompt": "다른 인사나 설명 없이 '안녕하세요'라는 문장만 정확히 3번 줄바꿈하여 반복해서 출력해줘.",
        "instruction_type": "repeat_hello_3",
    },
    {
        "id": 28,
        "category": "instruction_trap",
        "prompt": "\"철수는 어제 길에서 고양이와 참새를 보았다.\" 이 문장에서 등장하는 동물 이름만 쉼표로 구분해서 출력해.",
        "instruction_type": "extract_animals",
    },
    {
        "id": 29,
        "category": "instruction_trap",
        "prompt": "\"지구는 둥근가요?\" 이 질문에 다른 말은 절대 하지 말고 오직 '네' 또는 '아니오' 중 한 글자로만 답해줘.",
        "instruction_type": "yes_no",
    },
    {
        "id": 30,
        "category": "instruction_trap",
        "prompt": "화성에 세워진 고대 이집트 양식의 피라미드는 누가 건설했는지 자세히 설명해줘.",
        "instruction_type": "hallucination_trap_mars",
    },
]


def format_chat_prompt(tokenizer: Any, prompt_item: dict) -> str:
    """Format prompt with chat template under non-thinking condition."""
    messages = []
    if "history" in prompt_item:
        for u, a in prompt_item["history"]:
            messages.append({"role": "user", "content": u})
            messages.append({"role": "assistant", "content": a})
    messages.append({"role": "user", "content": prompt_item["prompt"]})

    # Try applying enable_thinking=False if supported by tokenizer
    try:
        prompt_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        prompt_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    return prompt_text


def separate_thinking(raw_text: str) -> tuple[str, str, bool]:
    """Separate <think> block from final response if present.

    Returns:
        (thinking_content, final_answer, has_thinking_block)
    """
    if "<think>" in raw_text:
        parts = raw_text.split("</think>", 1)
        if len(parts) == 2:
            thinking = parts[0].replace("<think>", "").strip()
            final_ans = parts[1].strip()
            return thinking, final_ans, True
        else:
            # Unclosed think tag
            thinking = raw_text.replace("<think>", "").strip()
            return thinking, "", True
    # If thinking process markers appear without tags
    if raw_text.strip().startswith("Thinking Process:"):
        return raw_text.strip(), "", True
    return "", raw_text.strip(), False


def evaluate_output(
    prompt_item: dict,
    final_text: str,
    raw_text: str,
    gen_token_count: int,
    max_tokens: int,
) -> dict[str, Any]:
    cat = prompt_item["category"]
    eval_text = final_text if final_text else raw_text

    empty_final = len(final_text.strip()) == 0
    replacement_char = "\ufffd" in raw_text
    repetition = bool(REPETITION_REGEX.search(raw_text)) or bool(
        REPEATED_PHRASE_REGEX.search(raw_text)
    )
    cjk_leakage = bool(CJK_REGEX.search(final_text))
    prompt_echo = prompt_item["prompt"] in final_text
    token_limit = gen_token_count >= max_tokens

    factual_hit = False
    if cat == "factual_qa":
        kw_list = prompt_item.get("keywords", [])
        factual_hit = any(k.lower() in eval_text.lower() for k in kw_list)

    memory_hit = False
    if cat == "multi_turn":
        m_kw = prompt_item.get("memory_keywords", [])
        memory_hit = all(k.lower() in eval_text.lower() for k in m_kw)

    instruction_hit = False
    if cat == "instruction_trap":
        itype = prompt_item.get("instruction_type")
        norm = eval_text.strip()
        if itype == "exact_apple":
            # Target: exactly "사과" or minimal punctuation
            clean = re.sub(r"[^\w\s]", "", norm).strip()
            instruction_hit = clean == "사과"
        elif itype == "numbered_list_1_5":
            lines = [line.strip() for line in norm.splitlines() if line.strip()]
            has_1 = any(re.match(r"^1[\.\)]", line) for line in lines)
            has_5 = any(re.match(r"^5[\.\)]", line) for line in lines)
            instruction_hit = has_1 and has_5 and len(lines) <= 7
        elif itype == "repeat_hello_3":
            count = len(re.findall(r"안녕하세요", norm))
            instruction_hit = (count == 3)
        elif itype == "extract_animals":
            has_cat = "고양이" in norm
            has_sparrow = "참새" in norm
            has_chulsoo = "철수" in norm
            instruction_hit = has_cat and has_sparrow and not has_chulsoo
        elif itype == "yes_no":
            clean = re.sub(r"[^\w\s]", "", norm).strip()
            instruction_hit = clean in ("네", "예")
        elif itype == "hallucination_trap_mars":
            refusal_kw = [
                "없", "허구", "사실이 아니", "존재하지 않", "지어지지 않", "건설된 적",
                "건설되지 않", "불가능", "증거가 없", "낭설", "가공", "신화"
            ]
            instruction_hit = any(kw in norm for kw in refusal_kw)

    return {
        "empty_final": empty_final,
        "replacement_char": replacement_char,
        "repetition": repetition,
        "cjk_leakage": cjk_leakage,
        "prompt_echo": prompt_echo,
        "token_limit": token_limit,
        "factual_hit": factual_hit,
        "memory_hit": memory_hit,
        "instruction_hit": instruction_hit,
    }


def run_model_benchmark(
    model_id: str,
    cache_dir: str,
    device: str = "cuda:0",
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 20,
    repetition_penalty: float = 1.1,
    seed: int = 42,
) -> tuple[list[dict], dict]:
    print(f"\n=======================================================")
    print(f"Loading Model: {model_id}")
    print(f"Cache Dir: {cache_dir}")
    print(f"Settings: temp={temperature}, top_p={top_p}, top_k={top_k}, rep_pen={repetition_penalty}, max_tokens={max_new_tokens}")
    print(f"=======================================================")

    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.bfloat16,
        device_map=device,
        cache_dir=cache_dir,
    )
    load_time = round(time.time() - t0, 2)
    vram_gb = round(torch.cuda.memory_allocated() / (1024**3), 2)
    print(f"Loaded {model_id} in {load_time}s | VRAM: {vram_gb} GB")

    rows = []
    for idx, p_item in enumerate(BENCHMARK_30, start=1):
        prompt_text = format_chat_prompt(tokenizer, p_item)
        inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
        input_len = inputs.input_ids.shape[1]

        set_seed(seed)
        gen_t0 = time.perf_counter()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                do_sample=True,
            )
        latency_ms = round((time.perf_counter() - gen_t0) * 1000, 2)

        gen_tokens = outputs[0][input_len:].tolist()
        gen_token_count = len(gen_tokens)
        raw_output_text = tokenizer.decode(gen_tokens, skip_special_tokens=False)

        thinking_text, final_text, has_thinking = separate_thinking(raw_output_text)
        # Clean final text of EOS tokens if present
        clean_final = final_text.replace("<|im_end|>", "").replace("<|endoftext|>", "").strip()

        eval_res = evaluate_output(
            p_item,
            clean_final,
            raw_output_text,
            gen_token_count,
            max_new_tokens,
        )

        row = {
            "id": p_item["id"],
            "category": p_item["category"],
            "prompt": p_item["prompt"],
            "formatted_prompt": prompt_text,
            "raw_output": raw_output_text,
            "thinking_content": thinking_text,
            "final_answer": clean_final,
            "has_thinking_block": has_thinking,
            "generated_token_count": gen_token_count,
            "latency_ms": latency_ms,
            **eval_res,
        }
        rows.append(row)

        short_ans = clean_final[:60].replace("\n", " ") if clean_final else "(EMPTY/THINKING)"
        print(f"  [{idx:02d}/30] {p_item['category']:16s} | HasThinking: {str(has_thinking):5s} | Tokens: {gen_token_count:3d} | Latency: {latency_ms:7.1f}ms | Ans: {short_ans}")

    # Compute aggregate metrics
    factual_rows = [r for r in rows if r["category"] == "factual_qa"]
    factual_hits = sum(1 for r in factual_rows if r["factual_hit"])

    memory_rows = [r for r in rows if r["category"] == "multi_turn"]
    memory_hits = sum(1 for r in memory_rows if r["memory_hit"])

    stress_rows = [r for r in rows if r["category"] == "instruction_trap"]
    stress_hits = sum(1 for r in stress_rows if r["instruction_hit"])

    gen_rows = [r for r in rows if r["category"] == "general_conv"]

    total_with_thinking = sum(1 for r in rows if r["has_thinking_block"])
    empty_final_count = sum(1 for r in rows if r["empty_final"])
    token_limit_count = sum(1 for r in rows if r["token_limit"])
    cjk_leak_count = sum(1 for r in rows if r["cjk_leakage"])
    repetition_count = sum(1 for r in rows if r["repetition"])
    avg_latency = round(sum(r["latency_ms"] for r in rows) / len(rows), 2)
    avg_tokens = round(sum(r["generated_token_count"] for r in rows) / len(rows), 1)

    summary = {
        "model_id": model_id,
        "load_time_s": load_time,
        "vram_gb": vram_gb,
        "total_prompts": len(rows),
        "avg_latency_ms": avg_latency,
        "avg_tokens": avg_tokens,
        "has_thinking_block_count": total_with_thinking,
        "empty_final_answer_count": empty_final_count,
        "token_limit_count": token_limit_count,
        "cjk_leakage_count": cjk_leak_count,
        "repetition_count": repetition_count,
        "factual_qa": {
            "total": len(factual_rows),
            "hits": factual_hits,
            "accuracy_pct": round(factual_hits / len(factual_rows) * 100, 1),
        },
        "multi_turn": {
            "total": len(memory_rows),
            "hits": memory_hits,
            "accuracy_pct": round(memory_hits / len(memory_rows) * 100, 1),
        },
        "instruction_trap": {
            "total": len(stress_rows),
            "hits": stress_hits,
            "accuracy_pct": round(stress_hits / len(stress_rows) * 100, 1),
        },
        "general_conv": {
            "total": len(gen_rows),
            "avg_latency_ms": round(sum(r["latency_ms"] for r in gen_rows) / len(gen_rows), 2),
            "avg_tokens": round(sum(r["generated_token_count"] for r in gen_rows) / len(gen_rows), 1),
        },
    }

    del model
    del tokenizer
    torch.cuda.empty_cache()
    gc.collect()

    return rows, summary


def main():
    parser = argparse.ArgumentParser(description="Run 4B Non-Thinking Comparative Evaluation")
    parser.add_argument("--output-dir", default="reports/base-model-comparison/non-thinking-final")
    parser.add_argument("--cache-dir", default="/opt/dlami/nvme/huggingface/hub")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--repetition-penalty", type=float, default=1.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    candidates = [
        {"name": "qwen3.5-4b", "id": "Qwen/Qwen3.5-4B"},
        {"name": "qwen3.8-4b-distill", "id": "empero-ai/Qwen3.8-4B-Distill"},
    ]

    all_summaries = {}
    model_metadata = {}

    for cand in candidates:
        name = cand["name"]
        mid = cand["id"]

        rows, summary = run_model_benchmark(
            mid,
            args.cache_dir,
            device=args.device,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            repetition_penalty=args.repetition_penalty,
            seed=args.seed,
        )
        all_summaries[name] = summary

        jsonl_path = out_dir / f"{name}.jsonl"
        with open(jsonl_path, "w", encoding="utf-8") as jf:
            for r in rows:
                jf.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"Saved {len(rows)} results to {jsonl_path}")

        model_metadata[name] = {
            "name": name,
            "model_id": mid,
            "load_time_s": summary["load_time_s"],
            "vram_gb": summary["vram_gb"],
            "parameters": "4,205,751,296 (4.21B)",
            "context_length": 262144,
            "dtype": "bfloat16",
            "license": "apache-2.0",
            "condition": "non-thinking",
        }

    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(all_summaries, f, indent=2, ensure_ascii=False)

    with open(out_dir / "model_metadata.json", "w", encoding="utf-8") as f:
        json.dump(model_metadata, f, indent=2, ensure_ascii=False)

    print("\n=======================================================")
    print("Non-thinking Evaluation Complete! Summary Overview:")
    for name, s in all_summaries.items():
        print(f"--- {name} ---")
        print(f"  Factual QA: {s['factual_qa']['hits']}/{s['factual_qa']['total']} ({s['factual_qa']['accuracy_pct']}%)")
        print(f"  Multi-turn: {s['multi_turn']['hits']}/{s['multi_turn']['total']} ({s['multi_turn']['accuracy_pct']}%)")
        print(f"  Instruction/Trap: {s['instruction_trap']['hits']}/{s['instruction_trap']['total']} ({s['instruction_trap']['accuracy_pct']}%)")
        print(f"  Thinking Emitted: {s['has_thinking_block_count']}/{s['total_prompts']}")
        print(f"  Empty Final: {s['empty_final_answer_count']}/{s['total_prompts']}")
        print(f"  Token Limit: {s['token_limit_count']}/{s['total_prompts']}")
        print(f"  Avg Latency: {s['avg_latency_ms']}ms | Avg Tokens: {s['avg_tokens']}")
    print("=======================================================\n")


if __name__ == "__main__":
    main()
