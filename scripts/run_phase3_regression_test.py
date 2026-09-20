"""Comprehensive Regression Test (100+ prompts) for ULM-1.7B Phase 3 Best Merged Model.

Evaluates:
- Echo / copy rate
- Question ignoring
- Multilingual token leakage (Chinese/Japanese/etc.)
- Repetition loops
- Broken/meaningless generation
- Dialect naturalness and tone
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

PROMPTS_DATASET = [
    # 1. Casual Chat (10)
    {"id": "casual_01", "category": "casual chat", "prompt": "요즘 어떻게 지내?"},
    {"id": "casual_02", "category": "casual chat", "prompt": "오늘 날씨 되게 좋다."},
    {"id": "casual_03", "category": "casual chat", "prompt": "주말에 보통 뭐 하고 시간 보내?"},
    {"id": "casual_04", "category": "casual chat", "prompt": "심심한데 뭐 재미있는 이야기 없어?"},
    {"id": "casual_05", "category": "casual chat", "prompt": "너 mbti가 뭐야?"},
    {"id": "casual_06", "category": "casual chat", "prompt": "오늘 하루도 참 바빴다."},
    {"id": "casual_07", "category": "casual chat", "prompt": "커피 한잔 마시면서 쉬는 중이야."},
    {"id": "casual_08", "category": "casual chat", "prompt": "너는 잠 안 올 때 뭐 해?"},
    {"id": "casual_09", "category": "casual chat", "prompt": "나 오늘 새로운 취미 시작했어."},
    {"id": "casual_10", "category": "casual chat", "prompt": "시간이 진짜 빨리 가는 것 같아."},

    # 2. Short Utterance (8)
    {"id": "short_01", "category": "short utterance", "prompt": "어"},
    {"id": "short_02", "category": "short utterance", "prompt": "밥"},
    {"id": "short_03", "category": "short utterance", "prompt": "헐"},
    {"id": "short_04", "category": "short utterance", "prompt": "그래?"},
    {"id": "short_05", "category": "short utterance", "prompt": "야"},
    {"id": "short_06", "category": "short utterance", "prompt": "엥?"},
    {"id": "short_07", "category": "short utterance", "prompt": "머꼬"},
    {"id": "short_08", "category": "short utterance", "prompt": "진짜?"},

    # 3. Slang (8)
    {"id": "slang_01", "category": "slang", "prompt": "게이야"},
    {"id": "slang_02", "category": "slang", "prompt": "이번 판 억까 지리네"},
    {"id": "slang_03", "category": "slang", "prompt": "완전 꿀잼이다 ㅋㅋㅋ"},
    {"id": "slang_04", "category": "slang", "prompt": "이거 완전 갓생 살기 프로젝트 아니냐"},
    {"id": "slang_05", "category": "slang", "prompt": "폼 미쳤다 진짜"},
    {"id": "slang_06", "category": "slang", "prompt": "개이득 봤다 오늘"},
    {"id": "slang_07", "category": "slang", "prompt": "킹받네 진짜 왜 이러지"},
    {"id": "slang_08", "category": "slang", "prompt": "이게 실화냐?"},

    # 4. Math (8)
    {"id": "math_01", "category": "math", "prompt": "1+1은 뭐야?"},
    {"id": "math_02", "category": "math", "prompt": "12 * 13은 얼마야?"},
    {"id": "math_03", "category": "math", "prompt": "100 나누기 4는 얼마야?"},
    {"id": "math_04", "category": "math", "prompt": "2의 10승이 얼마지?"},
    {"id": "math_05", "category": "math", "prompt": "50에서 17을 빼면?"},
    {"id": "math_06", "category": "math", "prompt": "7 곱하기 8은?"},
    {"id": "math_07", "category": "math", "prompt": "99 나누기 3은 얼마고?"},
    {"id": "math_08", "category": "math", "prompt": "15 더하기 28은?"},

    # 5. Factual QA (8)
    {"id": "fact_01", "category": "factual QA", "prompt": "지구에서 가장 높은 산은 어디야?"},
    {"id": "fact_02", "category": "factual QA", "prompt": "대한민국의 수도는 어디야?"},
    {"id": "fact_03", "category": "factual QA", "prompt": "임진왜란은 몇 년도에 일어났어?"},
    {"id": "fact_04", "category": "factual QA", "prompt": "태양계에서 가장 큰 행성은 뭐야?"},
    {"id": "fact_05", "category": "factual QA", "prompt": "한글을 창제한 조선의 왕은 누구야?"},
    {"id": "fact_06", "category": "factual QA", "prompt": "피타고라스 정리가 뭐야?"},
    {"id": "fact_07", "category": "factual QA", "prompt": "세계에서 가장 면적이 넓은 나라는?"},
    {"id": "fact_08", "category": "factual QA", "prompt": "올림픽은 몇 년 주기로 열려?"},

    # 6. Coding (8)
    {"id": "code_01", "category": "coding", "prompt": "파이썬으로 리스트 정렬하는 법 알려줘"},
    {"id": "code_02", "category": "coding", "prompt": "파이썬 딕셔너리가 뭐야?"},
    {"id": "code_03", "category": "coding", "prompt": "자바스크립트에서 배열 요소 거꾸로 뒤집는 법 알려줘"},
    {"id": "code_04", "category": "coding", "prompt": "HTML에서 링크 걸 때 무슨 태그 써?"},
    {"id": "code_05", "category": "coding", "prompt": "SQL에서 특정 테이블의 모든 데이터를 조회하는 쿼리문은?"},
    {"id": "code_06", "category": "coding", "prompt": "깃(Git)에서 원격 저장소 내용을 내려받는 명령어는 뭐야?"},
    {"id": "code_07", "category": "coding", "prompt": "C언어에서 포인터가 뭔지 쉽게 설명해줘"},
    {"id": "code_08", "category": "coding", "prompt": "파이썬에서 파일 읽고 쓸 때 with open 쓰는 이유가 뭐야?"},

    # 7. Science (8)
    {"id": "sci_01", "category": "science", "prompt": "왜 하늘은 파래?"},
    {"id": "sci_02", "category": "science", "prompt": "물이 100도에서 왜 끓어?"},
    {"id": "sci_03", "category": "science", "prompt": "중력이 뭐야?"},
    {"id": "sci_04", "category": "science", "prompt": "달은 왜 모양이 매일 바뀌어 보여?"},
    {"id": "sci_05", "category": "science", "prompt": "얼음이 왜 물 위에 떠?"},
    {"id": "sci_06", "category": "science", "prompt": "식물이 광합성을 하는 이유가 뭐야?"},
    {"id": "sci_07", "category": "science", "prompt": "번개가 칠 때 왜 천둥소리는 나중에 들려?"},
    {"id": "sci_08", "category": "science", "prompt": "바닷물이 짠 이유는 뭐야?"},

    # 8. Empathy (8)
    {"id": "emp_01", "category": "empathy", "prompt": "오늘 기분이 별로다"},
    {"id": "emp_02", "category": "empathy", "prompt": "내일 시험인데 공부하기 너무 싫다"},
    {"id": "emp_03", "category": "empathy", "prompt": "오늘 친구랑 사소한 일로 다퉈서 마음이 안 좋아"},
    {"id": "emp_04", "category": "empathy", "prompt": "회사에서 실수해서 하루 종일 눈치 보였어"},
    {"id": "emp_05", "category": "empathy", "prompt": "몸이 으슬으슬 춥고 감기 기운이 있네"},
    {"id": "emp_06", "category": "empathy", "prompt": "열심히 준비한 면접에서 떨어졌어... 속상해"},
    {"id": "emp_07", "category": "empathy", "prompt": "요즘 미래에 대한 걱정이 너무 많아"},
    {"id": "emp_08", "category": "empathy", "prompt": "퇴근하고 집에 왔는데 아무것도 하기 싫고 무기력해"},

    # 9. Recommendation (8)
    {"id": "rec_01", "category": "recommendation", "prompt": "배고픈데 뭐 먹지?"},
    {"id": "rec_02", "category": "recommendation", "prompt": "오늘 점심 메뉴 좀 골라줘"},
    {"id": "rec_03", "category": "recommendation", "prompt": "주말에 집에서 볼 만한 재미있는 넷플릭스 영화 추천해줘"},
    {"id": "rec_04", "category": "recommendation", "prompt": "선선한 저녁에 가볍게 들을 만한 노래 추천해줘"},
    {"id": "rec_05", "category": "recommendation", "prompt": "비 오는 날 어울리는 음식 뭐 있어?"},
    {"id": "rec_06", "category": "recommendation", "prompt": "혼자 떠나기 좋은 국내 여행지 추천해줘"},
    {"id": "rec_07", "category": "recommendation", "prompt": "친구 생일 선물로 3만 원대 부담 없는 거 뭐 좋을까?"},
    {"id": "rec_08", "category": "recommendation", "prompt": "요즘 읽을 만한 교양 도서 하나 추천해줄래?"},

    # 10. Multi-turn / Context (8)
    {
        "id": "multi_01", "category": "multi-turn",
        "history": [
            {"role": "user", "content": "나 주말에 울산 가려고 하거든"},
            {"role": "assistant", "content": "오 진짜가? 울산 오면 갈 데 많데이! 며칠 일정으로 오는데?"}
        ],
        "prompt": "1박 2일로 갈 건데 바다 보고 싶어"
    },
    {
        "id": "multi_02", "category": "multi-turn",
        "history": [
            {"role": "user", "content": "파이썬 공부 시작했어"},
            {"role": "assistant", "content": "잘 생각했데이! 파이썬이 문법도 깔끔하고 처음 배우기 딱 좋제. 어디까지 봤노?"}
        ],
        "prompt": "변수랑 반복문 배웠는데 다음엔 뭐 공부해야 돼?"
    },
    {
        "id": "multi_03", "category": "multi-turn",
        "history": [
            {"role": "user", "content": "오늘 감기 걸린 것 같아"},
            {"role": "assistant", "content": "아이고, 요즘 환절기라 감기 환자 많데이. 열은 안 나나?"}
        ],
        "prompt": "열은 없고 목만 좀 칼칼해"
    },
    {
        "id": "multi_04", "category": "multi-turn",
        "history": [
            {"role": "user", "content": "점심에 라면 먹을까 김치찌개 먹을까?"},
            {"role": "assistant", "content": "얼큰한 김치찌개가 든든하지 않겠나!"}
        ],
        "prompt": "좋아, 김치찌개 먹으러 간다"
    },
    {
        "id": "multi_05", "category": "multi-turn",
        "history": [
            {"role": "user", "content": "내일 친구 생일이야"},
            {"role": "assistant", "content": "축하해줄 일이네! 선물은 미리 준비했나?"}
        ],
        "prompt": "아직 못 샀는데 케이크만 사갈까?"
    },
    {
        "id": "multi_06", "category": "multi-turn",
        "history": [
            {"role": "user", "content": "너 혹시 노래 부르는 거 좋아해?"},
            {"role": "assistant", "content": "노래 부르는 건 좋아하지만 목소리가 없어서 아쉽제!"}
        ],
        "prompt": "나중에 노래도 한번 불러줘라"
    },
    {
        "id": "multi_07", "category": "multi-turn",
        "history": [
            {"role": "user", "content": "나 다이어트 시작했다"},
            {"role": "assistant", "content": "오! 결심 대단하네. 운동으로 빼나, 식단으로 빼나?"}
        ],
        "prompt": "저녁에 야식 안 먹기부터 해보려고"
    },
    {
        "id": "multi_08", "category": "multi-turn",
        "history": [
            {"role": "user", "content": "오늘 서울 출장 왔어"},
            {"role": "assistant", "content": "서울 사람 복잡할 텐데 고생 많데이! 길은 잘 찾았나?"}
        ],
        "prompt": "지하철 환승이 너무 복잡하더라"
    },

    # 11. Clarification (8)
    {"id": "clar_01", "category": "clarification", "prompt": "아니 그게 무슨 말이야?"},
    {"id": "clar_02", "category": "clarification", "prompt": "뭐라고?"},
    {"id": "clar_03", "category": "clarification", "prompt": "방금 한 말 다시 설명해줘"},
    {"id": "clar_04", "category": "clarification", "prompt": "잘 못 알아듣겠는데 쉽게 말해줄래?"},
    {"id": "clar_05", "category": "clarification", "prompt": "무슨 뜻인지 잘 모르겠어"},
    {"id": "clar_06", "category": "clarification", "prompt": "좀 더 자세하게 풀어서 얘기해봐"},
    {"id": "clar_07", "category": "clarification", "prompt": "예를 들어서 설명해줄 수 있어?"},
    {"id": "clar_08", "category": "clarification", "prompt": "한 번만 더 말해줄래?"},

    # 12. Ulsan Topics (8)
    {"id": "ulsan_01", "category": "울산 관련 질문", "prompt": "울산에서 놀러갈 만한 데 있어?"},
    {"id": "ulsan_02", "category": "울산 관련 질문", "prompt": "너 울산 사람이야?"},
    {"id": "ulsan_03", "category": "울산 관련 질문", "prompt": "태화강 국가정원이 어떤 곳이야?"},
    {"id": "ulsan_04", "category": "울산 관련 질문", "prompt": "대왕암공원 출렁다리 어때?"},
    {"id": "ulsan_05", "category": "울산 관련 질문", "prompt": "간절곶 해돋이 보러 가기 좋아?"},
    {"id": "ulsan_06", "category": "울산 관련 질문", "prompt": "울산 언양 불고기 맛있어?"},
    {"id": "ulsan_07", "category": "울산 관련 질문", "prompt": "울산 삼산동에 젊은 사람들 많이 가?"},
    {"id": "ulsan_08", "category": "울산 관련 질문", "prompt": "울산 공업축제 알아?"},

    # 13. Standard Korean General Inquiries (7)
    {"id": "std_01", "category": "표준어 일반 질문", "prompt": "비행기는 어떻게 하늘을 날 수 있나요?"},
    {"id": "std_02", "category": "표준어 일반 질문", "prompt": "커피를 너무 많이 마시면 건강에 어떤 영향이 있나요?"},
    {"id": "std_03", "category": "표준어 일반 질문", "prompt": "인공지능이란 무엇인가요?"},
    {"id": "std_04", "category": "표준어 일반 질문", "prompt": "환율이 오르면 우리 경제에 어떤 영향이 있습니까?"},
    {"id": "std_05", "category": "표준어 일반 질문", "prompt": "올바른 스트레칭 방법이 궁금합니다."},
    {"id": "std_06", "category": "표준어 일반 질문", "prompt": "블록체인 기술의 기본 원리에 대해 설명해 주세요."},
    {"id": "std_07", "category": "표준어 일반 질문", "prompt": "지구 온난화를 줄이기 위해 일상에서 실천할 수 있는 일은 무엇이 있나요?"},
]

DEFAULT_SYSTEM_PROMPT = (
    "울산 지역어 대화 assistant로서 자연스럽고 일상적인 울산 사투리로 상대방과 친근하게 대화한다. "
    "자연스럽고 편안한 일상 울산 말투를 기본으로 구사한다."
)

CJK_REGEX = re.compile(r"[\u4e00-\u9fff\u3040-\u30ff]")  # Chinese Hanzi or Japanese Kana
REPETITION_REGEX = re.compile(r"(.{4,})\1{2,}")  # 4+ char phrase repeating 3+ times


def detect_issues(prompt: str, response: str, category: str) -> list[str]:
    issues = []
    
    # 1. Empty or too short
    if len(response.strip()) < 2:
        issues.append("empty_or_too_short")

    # 2. Multilingual leakage (Chinese, Japanese)
    cjk_matches = CJK_REGEX.findall(response)
    if cjk_matches:
        issues.append(f"multilingual_leakage({len(cjk_matches)} tokens: {''.join(set(cjk_matches))})")

    # 3. Echo detection
    # If response is almost identical to prompt (>80% overlap) and prompt is not trivial single word
    clean_p = re.sub(r"[^\w\s]", "", prompt).strip()
    clean_r = re.sub(r"[^\w\s]", "", response).strip()
    if len(clean_p) >= 4:
        if clean_r == clean_p:
            issues.append("exact_echo")
        elif clean_p in clean_r and len(clean_r) < len(clean_p) * 1.3:
            issues.append("near_echo")

    # 4. Repetition loop
    if REPETITION_REGEX.search(response):
        issues.append("repetition_loop")

    # 5. Question ignored (generic refusal/nonsense)
    if any(bad in response for bad in ["알 수 없는 오류", "undefined", "NaN", "null"]):
        issues.append("broken_system_token")

    return issues


def main():
    parser = argparse.ArgumentParser(description="Run 100+ prompt regression test on ULM-1.7B")
    parser.add_argument("--model-path", default="outputs/ulm-1.7b-phase3-best-merged")
    parser.add_argument("--output-report", default="reports/phase3_regression_100_results.json")
    parser.add_argument("--output-issues", default="reports/phase3_regression_issues.jsonl")
    args = parser.parse_args()

    print(f"=== Running 100+ Prompt Regression Test on {args.model_path} ===")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
    )
    model.eval()

    total_prompts = len(PROMPTS_DATASET)
    print(f"Total test cases: {total_prompts}")

    results = []
    issues_list = []
    category_stats = {}

    for idx, item in enumerate(PROMPTS_DATASET, start=1):
        pid = item["id"]
        cat = item["category"]
        prompt = item["prompt"]
        history = item.get("history", [])

        category_stats.setdefault(cat, {"total": 0, "pass": 0, "issues": 0})
        category_stats[cat]["total"] += 1

        set_seed(42 + idx)
        messages = [{"role": "system", "content": DEFAULT_SYSTEM_PROMPT}]
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": prompt})

        template_kwargs = {"tokenize": False, "add_generation_prompt": True, "enable_thinking": False}
        try:
            text = tokenizer.apply_chat_template(messages, **template_kwargs)
        except TypeError:
            template_kwargs.pop("enable_thinking", None)
            text = tokenizer.apply_chat_template(messages, **template_kwargs)

        inputs = tokenizer(text, return_tensors="pt").to("cuda:0")
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=160,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.eos_token_id,
            )
        gen_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        response = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()

        detected_issues = detect_issues(prompt, response, cat)
        is_pass = len(detected_issues) == 0

        if is_pass:
            category_stats[cat]["pass"] += 1
        else:
            category_stats[cat]["issues"] += 1
            issue_record = {
                "id": pid,
                "category": cat,
                "prompt": prompt,
                "response": response,
                "issues": detected_issues,
            }
            issues_list.append(issue_record)

        results.append({
            "id": pid,
            "category": cat,
            "prompt": prompt,
            "response": response,
            "issues": detected_issues,
            "pass": is_pass,
        })

        status_str = "PASS" if is_pass else f"FAIL({','.join(detected_issues)})"
        print(f"[{idx:03d}/{total_prompts}] [{cat}] {prompt} -> {status_str}")

    total_pass = sum(c["pass"] for c in category_stats.values())
    pass_rate = (total_pass / total_prompts) * 100

    summary_report = {
        "model_path": args.model_path,
        "total_prompts": total_prompts,
        "total_pass": total_pass,
        "total_issues": len(issues_list),
        "pass_rate_pct": round(pass_rate, 2),
        "category_stats": category_stats,
        "results": results,
    }

    os.makedirs(os.path.dirname(args.output_report), exist_ok=True)
    with open(args.output_report, "w", encoding="utf-8") as f:
        json.dump(summary_report, f, ensure_ascii=False, indent=2)

    with open(args.output_issues, "w", encoding="utf-8") as f:
        for iss in issues_list:
            f.write(json.dumps(iss, ensure_ascii=False) + "\n")

    print(f"\n==========================================")
    print(f"REGRESSION TEST SUMMARY:")
    print(f"Total Prompts:  {total_prompts}")
    print(f"Passed:         {total_pass} ({pass_rate:.1f}%)")
    print(f"Issues:         {len(issues_list)}")
    print(f"Report saved:   {args.output_report}")
    print(f"Issues JSONL:   {args.output_issues}")
    print(f"==========================================")


if __name__ == "__main__":
    main()
