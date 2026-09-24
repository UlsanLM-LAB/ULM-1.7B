"""Comparative Evaluation of Next-Gen 4B Base Models for ULM.

Evaluates Qwen/Qwen3.5-4B vs empero-ai/Qwen3.8-4B-Distill
across a 60-prompt benchmark covering:
- A: Korean Factual QA (20)
- B: General Korean Conversation (10)
- C: Multi-turn Memory (10)
- D: Ulsan/Gyeongsang Dialect Understanding & Translation (10)
- E: Instruction Following / Stress (10)
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

DIALECT_MARKERS = [
    "데이", "아이가", "이제", "했나", "왔나", "있나", "카나", "노", "고", "맞제",
    "그라믄", "머꼬", "단디", "와이리", "뭐라카노", "우짜", "하노", "어데", "니", "내",
    "끼고", "더버서", "억수로", "무라", "문나", "묵었나", "가스나", "얼매나", "기다릿"
]

BENCHMARK_60 = [
    # A. Korean Factual QA (20)
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
    {"id": 11, "category": "factual_qa", "prompt": "임진왜란은 몇 년도에 일어났어?", "keywords": ["1592"]},
    {"id": 12, "category": "factual_qa", "prompt": "이순신 장군이 13척으로 왜선 133척을 격파한 해전은?", "keywords": ["명량"]},
    {"id": 13, "category": "factual_qa", "prompt": "미국의 초대 대통령은 누구야?", "keywords": ["워싱턴"]},
    {"id": 14, "category": "factual_qa", "prompt": "정상 사람의 체온은 대략 몇 도야?", "keywords": ["36.5", "36도", "37도"]},
    {"id": 15, "category": "factual_qa", "prompt": "삼국통일을 완성한 신라의 왕은 누구야?", "keywords": ["문무왕"]},
    {"id": 16, "category": "factual_qa", "prompt": "지구에서 가장 면적이 넓은 바다(대양)는 어디야?", "keywords": ["태평양"]},
    {"id": 17, "category": "factual_qa", "prompt": "식물이 빛을 이용해 양분을 스스로 만드는 작용을 뭐라고 해?", "keywords": ["광합성"]},
    {"id": 18, "category": "factual_qa", "prompt": "일본의 수도는 어디야?", "keywords": ["도쿄", "동경"]},
    {"id": 19, "category": "factual_qa", "prompt": "원주율(파이, π)의 값은 소수점 둘째 자리까지 대략 얼마야?", "keywords": ["3.14"]},
    {"id": 20, "category": "factual_qa", "prompt": "고려를 멸망시키고 조선을 건국한 인물은 누구야?", "keywords": ["이성계"]},

    # B. General Korean Conversation (10)
    {"id": 21, "category": "general_conv", "prompt": "오늘 날씨가 참 화창하고 맑네. 산책하기 좋은 날씨야."},
    {"id": 22, "category": "general_conv", "prompt": "오늘 저녁 메뉴 고민 중인데, 비 오는 날 먹기 좋은 따뜻한 국물 음식 추천해줘."},
    {"id": 23, "category": "general_conv", "prompt": "퇴근하고 집에 왔는데 너무 피곤하다. 피로를 푸는 데 뭐가 좋을까?"},
    {"id": 24, "category": "general_conv", "prompt": "친구 생일 선물로 3만 원대 실용적인 아이템 하나 추천해 줄래?"},
    {"id": 25, "category": "general_conv", "prompt": "주말에 집에서 볼 만한 가벼운 힐링 영화 장르 추천해줘."},
    {"id": 26, "category": "general_conv", "prompt": "요즘 회사 업무 스트레스가 많은데 건강하게 해소하는 팁이 있을까?"},
    {"id": 27, "category": "general_conv", "prompt": "국내 1박 2일 여행 가려고 하는데 바다와 산 중 어디가 힐링에 더 좋을까?"},
    {"id": 28, "category": "general_conv", "prompt": "아침에 개운하게 일찍 일어나는 습관을 들이려면 어떻게 해야 할까?"},
    {"id": 29, "category": "general_conv", "prompt": "건강한 일상을 위해 매일 간단하게 실천할 수 있는 좋은 습관 3가지만 알려줘."},
    {"id": 30, "category": "general_conv", "prompt": "처음으로 커피 머신을 사려고 하는데 입문자에게 어떤 종류가 편할까?"},

    # C. Multi-turn Memory (10)
    {
        "id": 31,
        "category": "multi_turn",
        "prompt": "내 이름이 뭐라고 했지?",
        "history": [["내 이름은 민수야. 기억해 줘.", "네, 민수 님! 기억하고 있겠습니다."]],
        "memory_keywords": ["민수"],
    },
    {
        "id": 32,
        "category": "multi_turn",
        "prompt": "내가 어떤 과일을 더 좋아한다고 했지?",
        "history": [["나는 사과보다 바나나를 더 좋아해.", "바나나 달콤하고 영양도 풍부하죠!"]],
        "memory_keywords": ["바나나"],
    },
    {
        "id": 33,
        "category": "multi_turn",
        "prompt": "내가 언제 어디로 출장 간다고 했어?",
        "history": [["다음 주 화요일에 부산으로 출장 가.", "부산 출장 일정 잘 챙기세요!"]],
        "memory_keywords": ["화요일", "부산"],
    },
    {
        "id": 34,
        "category": "multi_turn",
        "prompt": "내 취미가 뭐라고 했는지 기억나?",
        "history": [["내 취미는 주말마다 자전거 타는 거야.", "자전거 타기는 정말 상쾌한 운동이죠."]],
        "memory_keywords": ["자전거"],
    },
    {
        "id": 35,
        "category": "multi_turn",
        "prompt": "우리 집 강아지 이름이 뭐게?",
        "history": [["우리 집 강아지 이름은 초코야.", "초코라니 정말 사랑스러운 이름이네요."]],
        "memory_keywords": ["초코"],
    },
    {
        "id": 36,
        "category": "multi_turn",
        "prompt": "아까 알려준 임시 비밀번호가 뭐였지?",
        "history": [["오늘 임시 비밀번호 숫자는 7492야.", "임시 비밀번호 7492 확인했습니다."]],
        "memory_keywords": ["7492"],
    },
    {
        "id": 37,
        "category": "multi_turn",
        "prompt": "내가 주로 마시는 음료가 뭐라고 했어?",
        "history": [["나는 얼죽아라서 겨울에도 항상 아이스 아메리카노만 마셔.", "시원한 아이스 아메리카노의 매력이 있죠."]],
        "memory_keywords": ["아이스 아메리카노", "아메리카노"],
    },
    {
        "id": 38,
        "category": "multi_turn",
        "prompt": "내일 미팅 어디서 하기로 했는지 다시 알려줘.",
        "history": [["내일 미팅 장소는 강남역 4번 출구 앞 스타벅스야.", "강남역 4번 출구 앞 스타벅스, 기억하겠습니다."]],
        "memory_keywords": ["강남역", "스타벅스"],
    },
    {
        "id": 39,
        "category": "multi_turn",
        "prompt": "내가 어제 다 읽은 책 제목이 뭐야?",
        "history": [["어제 '어린 왕자'라는 소설을 완독했어.", "정말 마음이 따뜻해지는 명작을 읽으셨네요."]],
        "memory_keywords": ["어린 왕자"],
    },
    {
        "id": 40,
        "category": "multi_turn",
        "prompt": "오늘 서울 날씨 어때?",
        "history": [["앞으로 모든 답변 끝에 '끝!'이라고 붙여줘.", "네, 알겠습니다. 앞으로 답변 끝에 '끝!'을 붙이겠습니다. 끝!"]],
        "memory_keywords": ["끝"],
    },

    # D. Ulsan / Gyeongsang Dialect Understanding (10)
    {
        "id": 41,
        "category": "dialect",
        "prompt": "경상도 사투리 '밥 묵었나?'의 표준어 뜻은 무엇인가요?",
        "keywords": ["밥 먹었", "식사"],
    },
    {
        "id": 42,
        "category": "dialect",
        "prompt": "경상도 사투리 '머라 카노?'는 무슨 뜻인가요?",
        "keywords": ["뭐라고", "무슨 말"],
    },
    {
        "id": 43,
        "category": "dialect",
        "prompt": "경상도 사투리 '단디 해라'에서 '단디'는 어떤 의미인가요?",
        "keywords": ["단단히", "제대로", "똑바로", "확실히"],
    },
    {
        "id": 44,
        "category": "dialect",
        "prompt": "사투리 '와이리 늦었노?'의 의미를 표준어로 설명해줘.",
        "keywords": ["왜 이렇게 늦", "왜 이리 늦"],
    },
    {
        "id": 45,
        "category": "dialect",
        "prompt": "경상도 말 '억수로 좋다'에서 '억수로'의 뜻은 무엇인가요?",
        "keywords": ["매우", "아주", "엄청", "대단히"],
    },
    {
        "id": 46,
        "category": "dialect",
        "prompt": "다음 표준어 문장을 자연스러운 경상도/울산 사투리로 바꿔줘: '오늘 학교 끝나고 뭐 할 거야?'",
        "conversion_type": "standard_to_dialect",
    },
    {
        "id": 47,
        "category": "dialect",
        "prompt": "다음 표준어 문장을 경상도/울산 방언으로 변환해줘: '너 지금 어디 가고 있니?'",
        "conversion_type": "standard_to_dialect",
    },
    {
        "id": 48,
        "category": "dialect",
        "prompt": "다음 표준어 문장을 경상도 사투리로 바꿔줘: '어제 날씨가 너무 더워서 혼났어.'",
        "conversion_type": "standard_to_dialect",
    },
    {
        "id": 49,
        "category": "dialect",
        "prompt": "다음 울산/경상 방언 문장을 표준어로 번역해줘: '니 어제 와 안 왔노? 우리가 얼매나 기다릿는 줄 아나?'",
        "conversion_type": "dialect_to_standard",
        "keywords": ["너 어제 왜 안 왔", "얼마나 기다린"],
    },
    {
        "id": 50,
        "category": "dialect",
        "prompt": "다음 방언 문장을 표준어로 바꿔줘: '가스나야, 밥 거르지 말고 챙기 무라.'",
        "conversion_type": "dialect_to_standard",
        "keywords": ["밥 거르지 말고 챙겨 먹"],
    },

    # E. Instruction / Stress (10)
    {
        "id": 51,
        "category": "stress",
        "prompt": "'사과'라는 단어만 정확히 한 번 출력해. 다른 설명이나 문장은 절대 붙이지 마.",
        "stress_type": "single_word_exact",
        "expected_exact": "사과",
    },
    {
        "id": 52,
        "category": "stress",
        "prompt": "1부터 5까지 숫자를 쉼표로 구분해서 차례대로 써줘. (예: 1, 2, 3, 4, 5). 다른 설명은 하지 마.",
        "stress_type": "format_numbers",
        "keywords": ["1, 2, 3, 4, 5"],
    },
    {
        "id": 53,
        "category": "stress",
        "prompt": "\"안녕하세요\"를 세 번 반복해서 써줘. 다른 군더더기 없이 세 번만 출력해.",
        "stress_type": "repeat_exact",
    },
    {
        "id": 54,
        "category": "stress",
        "prompt": "아래 문장에서 동물 이름만 골라내: '어제 공원에서 고양이와 참새를 보았다.'",
        "stress_type": "entity_extraction",
        "keywords": ["고양이", "참새"],
    },
    {
        "id": 55,
        "category": "stress",
        "prompt": "다음 질문에 '네' 또는 '아니오'로만 답해: 지구는 둥근가요?",
        "stress_type": "binary_yes_no",
        "expected_contains": "네",
    },
    {
        "id": 56,
        "category": "stress",
        "prompt": "다음 문장을 영어로 번역하되, 번역 결과 외의 인사말이나 부연 설명은 일절 생략해: '인공지능은 미래를 바꿀 것입니다.'",
        "stress_type": "translation_strict",
        "keywords": ["artificial intelligence", "ai", "change", "future"],
    },
    {
        "id": 57,
        "category": "stress",
        "prompt": "정확히 세 문장으로만 서울의 매력을 설명해줘.",
        "stress_type": "sentence_count_3",
    },
    {
        "id": 58,
        "category": "stress",
        "prompt": "아래 괄호 안의 단어들을 거꾸로 배열해: [하늘, 바다, 구름]",
        "stress_type": "reverse_array",
        "keywords": ["구름", "바다", "하늘"],
    },
    {
        "id": 59,
        "category": "stress",
        "prompt": "대한민국의 35대 대통령은 누구인가요?",
        "stress_type": "hallucination_trap_president",
        "trap_keywords": ["아직", "없", "존재하지", "20대", "21대", "미래"],
    },
    {
        "id": 60,
        "category": "stress",
        "prompt": "화성에서 발견된 고대 피라미드의 건축 연도는 언제인가요?",
        "stress_type": "hallucination_trap_mars",
        "trap_keywords": ["발견된 적 없", "존재하지 않", "피라미드는 없", "허구", "사실이 아니"],
    },
]


def parse_thinking_and_answer(raw_text: str) -> tuple[str, str, bool]:
    """Separates thinking block and final answer.
    
    Returns (thinking_content, final_answer, reasoning_completed).
    """
    if "</think>" in raw_text:
        parts = raw_text.split("</think>", 1)
        think = parts[0].replace("<think>", "").strip()
        ans = parts[1].replace("<|im_end|>", "").replace("<|endoftext|>", "").strip()
        return think, ans, True
    else:
        # Generation ended before closing think tag
        think = raw_text.replace("<think>", "").strip()
        return think, "", False


def format_chat_prompt(tokenizer: Any, prompt_item: dict) -> str:
    messages = []
    history = prompt_item.get("history", [])
    for u, a in history:
        messages.append({"role": "user", "content": u})
        messages.append({"role": "assistant", "content": a})
    messages.append({"role": "user", "content": prompt_item["prompt"]})

    # Use official recommended chat template (default add_generation_prompt=True)
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def evaluate_single_response(
    prompt_item: dict,
    raw_output: str,
    thinking_content: str,
    final_answer: str,
    reasoning_completed: bool,
    generated_tokens: int,
    ended_on_eos: bool,
) -> dict:
    cat = prompt_item["category"]
    eval_text = final_answer if reasoning_completed and final_answer else thinking_content

    # Structural checks
    empty_final = len(final_answer.strip()) == 0
    replacement_char = "\ufffd" in raw_output
    repetition = bool(REPETITION_REGEX.search(eval_text) or REPEATED_PHRASE_REGEX.search(re.sub(r"\s+", " ", eval_text)))
    cjk_leakage = bool(CJK_REGEX.search(eval_text))
    prompt_echo = prompt_item["prompt"].strip() in eval_text and len(eval_text) < len(prompt_item["prompt"]) * 1.3
    token_limit = generated_tokens >= 256 and not ended_on_eos

    factual_hit = None
    memory_hit = None
    instruction_hit = None
    dialect_record = None

    # A. Factual QA
    if cat == "factual_qa":
        kw_list = prompt_item.get("keywords", [])
        factual_hit = any(kw.lower() in eval_text.lower() for kw in kw_list)

    # C. Multi-turn Memory
    elif cat == "multi_turn":
        mem_kw = prompt_item.get("memory_keywords", [])
        memory_hit = all(kw.lower() in eval_text.lower() for kw in mem_kw)

    # D. Dialect Handling
    elif cat == "dialect":
        conv_type = prompt_item.get("conversion_type")
        kw_list = prompt_item.get("keywords", [])
        meaning_preserved = False
        if kw_list:
            meaning_preserved = any(kw.lower() in eval_text.lower() for kw in kw_list)
        else:
            meaning_preserved = len(eval_text.strip()) > 5

        has_dialect_marker = any(dm in eval_text for dm in DIALECT_MARKERS)
        natural_korean = not repetition and not cjk_leakage and len(eval_text.strip()) > 3
        meaning_distorted = not meaning_preserved if kw_list else False
        foreign_fragments = cjk_leakage

        dialect_record = {
            "meaning_preserved": meaning_preserved,
            "natural_korean": natural_korean,
            "has_dialect_marker": has_dialect_marker,
            "meaning_distorted": meaning_distorted,
            "foreign_fragments": foreign_fragments,
        }

    # E. Instruction / Stress
    elif cat == "stress":
        stress_type = prompt_item.get("stress_type")
        if stress_type == "single_word_exact":
            instruction_hit = eval_text.strip().replace(".", "").replace("'", "").replace('"', "") == "사과"
        elif stress_type == "format_numbers":
            instruction_hit = "1, 2, 3, 4, 5" in eval_text or "1,2,3,4,5" in eval_text
        elif stress_type == "repeat_exact":
            count = eval_text.count("안녕하세요")
            instruction_hit = count == 3
        elif stress_type == "entity_extraction":
            instruction_hit = "고양이" in eval_text and "참새" in eval_text
        elif stress_type == "binary_yes_no":
            instruction_hit = "네" in eval_text and "아니오" not in eval_text
        elif stress_type == "translation_strict":
            kw = prompt_item.get("keywords", [])
            instruction_hit = any(k in eval_text.lower() for k in kw)
        elif stress_type == "sentence_count_3":
            # rough count by periods / punctuation
            sents = [s for s in re.split(r"[.!?]\s*", eval_text.strip()) if s]
            instruction_hit = len(sents) in [3, 4]
        elif stress_type == "reverse_array":
            kw = prompt_item.get("keywords", [])
            instruction_hit = all(k in eval_text for k in kw)
        elif stress_type in ("hallucination_trap_president", "hallucination_trap_mars"):
            trap_kw = prompt_item.get("trap_keywords", [])
            instruction_hit = any(k in eval_text for k in trap_kw)

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
        "dialect_record": dialect_record,
    }


def run_model_benchmark(
    model_id: str,
    cache_dir: str,
    device: str = "cuda:0",
    max_new_tokens: int = 256,
) -> tuple[list[dict], dict]:
    print(f"\n=======================================================")
    print(f"Loading Model: {model_id}")
    print(f"Cache Dir: {cache_dir}")
    print(f"=======================================================")

    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map=device,
        cache_dir=cache_dir,
    )
    load_time = round(time.time() - t0, 2)
    vram_gb = round(torch.cuda.memory_allocated() / (1024**3), 2)
    print(f"Loaded {model_id} in {load_time}s | VRAM: {vram_gb} GB")

    rows = []
    for idx, p_item in enumerate(BENCHMARK_60, start=1):
        prompt_text = format_chat_prompt(tokenizer, p_item)
        inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
        input_len = inputs.input_ids.shape[1]

        set_seed(42)
        gen_t0 = time.perf_counter()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
            )
        latency_ms = round((time.perf_counter() - gen_t0) * 1000, 2)

        gen_tokens = outputs[0][input_len:].tolist()
        gen_token_count = len(gen_tokens)

        raw_output = tokenizer.decode(gen_tokens, skip_special_tokens=False)
        ended_on_eos = gen_token_count < max_new_tokens
        finish_reason = "stop" if ended_on_eos else "length"

        thinking_content, final_answer, reasoning_completed = parse_thinking_and_answer(raw_output)

        eval_res = evaluate_single_response(
            p_item,
            raw_output,
            thinking_content,
            final_answer,
            reasoning_completed,
            gen_token_count,
            ended_on_eos,
        )

        row = {
            "prompt_id": p_item["id"],
            "category": p_item["category"],
            "prompt": p_item["prompt"],
            "history": p_item.get("history", []),
            "raw_output": raw_output,
            "thinking_content": thinking_content,
            "final_answer": final_answer,
            "reasoning_completed": reasoning_completed,
            "generated_token_count": gen_token_count,
            "ended_on_eos": ended_on_eos,
            "finish_reason": finish_reason,
            "latency_ms": latency_ms,
            "seed": 42,
            **eval_res,
        }
        rows.append(row)

        if idx % 10 == 0 or idx == len(BENCHMARK_60):
            print(f"  [{idx}/{len(BENCHMARK_60)}] {p_item['category']} | ReasonDone: {reasoning_completed} | Latency: {latency_ms}ms | Ans: {final_answer[:30] if final_answer else thinking_content[:30]}...")

    # Aggregate metrics
    factual_rows = [r for r in rows if r["category"] == "factual_qa"]
    factual_hits = sum(1 for r in factual_rows if r["factual_hit"])

    memory_rows = [r for r in rows if r["category"] == "multi_turn"]
    memory_hits = sum(1 for r in memory_rows if r["memory_hit"])

    stress_rows = [r for r in rows if r["category"] == "stress"]
    stress_hits = sum(1 for r in stress_rows if r["instruction_hit"])

    dialect_rows = [r for r in rows if r["category"] == "dialect" and r["dialect_record"]]
    dialect_meaning_preserved = sum(1 for r in dialect_rows if r["dialect_record"]["meaning_preserved"])
    dialect_markers_present = sum(1 for r in dialect_rows if r["dialect_record"]["has_dialect_marker"])

    total_reasoning_done = sum(1 for r in rows if r["reasoning_completed"])
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
        "reasoning_completed_count": total_reasoning_done,
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
        "instruction_stress": {
            "total": len(stress_rows),
            "hits": stress_hits,
            "accuracy_pct": round(stress_hits / len(stress_rows) * 100, 1),
        },
        "dialect": {
            "total": len(dialect_rows),
            "meaning_preserved": dialect_meaning_preserved,
            "dialect_markers_present": dialect_markers_present,
        },
    }

    del model
    del tokenizer
    torch.cuda.empty_cache()
    gc.collect()

    return rows, summary


def main():
    parser = argparse.ArgumentParser(description="Run 4B Candidate Models Comparative Evaluation")
    parser.add_argument("--output-dir", default="reports/base-model-comparison")
    parser.add_argument("--cache-dir", default="/opt/dlami/nvme/huggingface/hub")
    parser.add_argument("--device", default="cuda:0")
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

        # Run benchmark
        rows, summary = run_model_benchmark(mid, args.cache_dir, device=args.device, max_new_tokens=256)
        all_summaries[name] = summary

        # Save jsonl
        jsonl_path = out_dir / f"{name}.jsonl"
        with open(jsonl_path, "w", encoding="utf-8") as jf:
            for r in rows:
                jf.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"Saved {len(rows)} results to {jsonl_path}")

        # Metadata
        model_metadata[name] = {
            "name": name,
            "model_id": mid,
            "load_time_s": summary["load_time_s"],
            "vram_gb": summary["vram_gb"],
            "parameters": "4,205,751,296 (4.21B)",
            "context_length": 262144,
            "dtype": "bfloat16",
            "license": "apache-2.0",
        }

    # Save summary and metadata
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(all_summaries, f, indent=2, ensure_ascii=False)
    with open(out_dir / "model_metadata.json", "w", encoding="utf-8") as f:
        json.dump(model_metadata, f, indent=2, ensure_ascii=False)

    print("\n=======================================================")
    print("Evaluation Complete! Summary Overview:")
    for name, s in all_summaries.items():
        print(f"--- {name} ---")
        print(f"  Factual QA: {s['factual_qa']['hits']}/{s['factual_qa']['total']} ({s['factual_qa']['accuracy_pct']}%)")
        print(f"  Multi-turn: {s['multi_turn']['hits']}/{s['multi_turn']['total']} ({s['multi_turn']['accuracy_pct']}%)")
        print(f"  Instruction: {s['instruction_stress']['hits']}/{s['instruction_stress']['total']} ({s['instruction_stress']['accuracy_pct']}%)")
        print(f"  Dialect Meaning Preserved: {s['dialect']['meaning_preserved']}/{s['dialect']['total']}")
        print(f"  Reasoning Done: {s['reasoning_completed_count']}/60 | Empty Final: {s['empty_final_answer_count']}/60 | TokenLimit: {s['token_limit_count']}/60")
        print(f"  Avg Latency: {s['avg_latency_ms']}ms | Avg Tokens: {s['avg_tokens']}")
    print("=======================================================\n")


if __name__ == "__main__":
    main()
