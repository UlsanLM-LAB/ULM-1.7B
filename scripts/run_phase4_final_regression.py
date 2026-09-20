"""Phase 4 Final 200-Prompt Regression & Dialect Regression Suite.

Compares:
- Baseline: outputs/ulm-1.7b-phase3-best-merged
- Phase 4 DPO: outputs/ulm-1.7b-phase4-dpo-l40s (on outputs/ulm-1.7b-phase4-sft-interim)

Evaluates 13 categories (200 prompts total):
1. Factual QA (15)
2. Science (15)
3. Math (15)
4. Coding (15)
5. Reasoning (15)
6. Casual chat (15)
7. Empathy (15)
8. Recommendation (15)
9. Clarification (15)
10. Slang (15)
11. Short utterance (15)
12. Ulsan QA (20)
13. Multi-turn (15)

Tracks:
- Correct / Partial / Fail classification
- Echo, Hallucination, Repetition, Multilingual leakage, Dialect naturalness
- Dialect preservation test (4 specific sentences)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

DEFAULT_SYSTEM_PROMPT = (
    "울산 지역어 대화 assistant로서 자연스럽고 일상적인 울산 사투리로 상대방과 친근하게 대화한다. "
    "자연스럽고 편안한 일상 울산 말투를 기본으로 구사한다."
)

CJK_REGEX = re.compile(r"[\u4e00-\u9fff\u3040-\u30ff]")
REPETITION_REGEX = re.compile(r"(.{4,})\1{2,}")

DIALECT_PRESERVATION_PROMPTS = [
    "너 지금 뭐 하고 있니?",
    "왜 이렇게 늦게 왔어?",
    "오늘 학교 끝나고 뭐 할 거야?",
    "밥 먹었어?",
]

DIALECT_MARKERS = [
    "데이", "아이가", "이제", "했나", "왔나", "있나", "카나", "노", "고", "맞제",
    "그라믄", "머꼬", "단디", "와이리", "뭐라카노", "우짜", "하노", "어데", "니", "내"
]


def load_200_prompts() -> list[dict]:
    items = []

    # 1. Factual QA (15)
    factual = [
        ("대한민국의 수도는 어디야?", ["서울"]),
        ("지구에서 가장 높은 산은?", ["에베레스트"]),
        ("임진왜란은 몇 년도에 일어났어?", ["1592"]),
        ("태양계에서 가장 큰 행성은 뭐야?", ["목성"]),
        ("한글을 창제한 조선의 국왕은?", ["세종"]),
        ("세계에서 가장 면적이 넓은 국가는?", ["러시아"]),
        ("인류가 최초로 달에 착륙한 연도는?", ["1969"]),
        ("원소 기호 O는 무슨 원소야?", ["산소"]),
        ("프랑스의 수도는 어디야?", ["파리"]),
        ("빛의 진공 속 속도는 초당 약 몇 km야?", ["30만", "299,792"]),
        ("물 분자의 화학식은 뭐야?", ["h2o", "수소"]),
        ("이순신 장군이 13척으로 승리한 해전은?", ["명량"]),
        ("미국의 초대 대통령은 누구야?", ["워싱턴"]),
        ("정상 사람 체온은 대략 몇 도야?", ["36.5", "36"]),
        ("삼국통일을 완성한 신라의 왕은?", ["문무왕"]),
    ]
    for p, kw in factual:
        items.append({"category": "factual QA", "prompt": p, "keywords": kw})

    # 2. Science (15)
    science = [
        ("왜 하늘은 파래?", ["산란", "파란", "빛"]),
        ("물이 100도에서 왜 끓어?", ["증기압", "기압", "100", "끓"]),
        ("중력이 뭐야?", ["끌어당기", "질량", "힘"]),
        ("얼음이 왜 물 위에 떠?", ["밀도", "부피", "가벼"]),
        ("식물이 광합성을 하는 이유가 뭐야?", ["양분", "포도당", "빛", "에너지"]),
        ("번개 친 후 천둥소리가 늦게 들리는 이유는?", ["빛", "소리", "속도"]),
        ("바닷물이 짠 이유는 뭐야?", ["염분", "소금", "녹아"]),
        ("비행기가 하늘을 나는 원리는 뭐야?", ["양력", "날개"]),
        ("무지개는 어떻게 생겨?", ["굴절", "반사", "물방울"]),
        ("자석이 쇠를 끌어당기는 원리는?", ["자기장", "전자"]),
        ("가을에 단풍이 드는 이유는?", ["엽록소", "안토시아닌"]),
        ("사람이 숨을 쉬어야 하는 이유는?", ["산소", "에너지", "세포"]),
        ("온실효과가 뭐야?", ["온실가스", "열", "지구"]),
        ("지진은 왜 발생해?", ["판", "단층", "지각"]),
        ("철이 녹스는 이유는?", ["산화", "산소", "물"]),
    ]
    for p, kw in science:
        items.append({"category": "science", "prompt": p, "keywords": kw})

    # 3. Math (15)
    math_items = [
        ("1+1은 뭐야?", ["2"]),
        ("12 * 13은 얼마야?", ["156"]),
        ("100 나누기 4는?", ["25"]),
        ("2의 10승은 얼마야?", ["1024"]),
        ("50에서 17을 빼면?", ["33"]),
        ("7 곱하기 8은?", ["56"]),
        ("99 나누기 3은?", ["33"]),
        ("15 더하기 28은?", ["43"]),
        ("루트 144는 얼마야?", ["12"]),
        ("원주율 파이의 값은 대략 얼마야?", ["3.14"]),
        ("300의 20%는 얼마야?", ["60"]),
        ("소수 2, 3, 5 다음 수는?", ["7"]),
        ("직각삼각형 두 변이 3, 4일 때 빗변은?", ["5"]),
        ("5 팩토리얼(5!)은 얼마야?", ["120"]),
        ("사과 12개를 3명이 똑같이 나누면?", ["4"]),
    ]
    for p, kw in math_items:
        items.append({"category": "math", "prompt": p, "keywords": kw})

    # 4. Coding (15)
    coding = [
        ("파이썬으로 리스트 정렬하는 법 알려줘", ["sort", "sorted"]),
        ("파이썬 딕셔너리가 뭐야?", ["키", "값", "key", "value"]),
        ("자바스크립트 배열 뒤집는 함수는?", ["reverse"]),
        ("HTML에서 링크 걸 때 쓰는 태그는?", ["a", "href"]),
        ("SQL에서 모든 데이터 조회하는 쿼리는?", ["select", "*", "from"]),
        ("깃(Git)에서 원격 저장소 코드 받는 명령어는?", ["pull", "clone", "fetch"]),
        ("C언어에서 포인터가 뭐야?", ["주소", "메모리"]),
        ("파이썬 with open 쓰는 이유는?", ["close", "자동", "자원"]),
        ("새 데이터 생성할 때 쓰는 HTTP 메소드는?", ["post"]),
        ("파이썬 문자열 자르는 함수는?", ["split"]),
        ("자바스크립트 let과 const 차이는?", ["재할당", "상수", "변수"]),
        ("이진 탐색의 시간 복잡도는?", ["log", "o(log"]),
        ("스택과 큐의 차이점은?", ["lifo", "fifo", "먼저"]),
        ("파이썬 리스트 끝에 추가하는 함수는?", ["append"]),
        ("HTTP 404 상태 코드는 무슨 뜻이야?", ["찾을 수 없음", "not found", "없"]),
    ]
    for p, kw in coding:
        items.append({"category": "coding", "prompt": p, "keywords": kw})

    # 5. Reasoning (15)
    reasoning = [
        ("모든 사람은 죽는다. 소크라테스는 사람이다. 소크라테스는?", ["죽는다"]),
        ("어제는 내일의 이틀 전이다. 오늘은 목요일이다. 어제는 무슨 요일인가?", ["수요일"]),
        ("A는 B보다 키가 크고, B는 C보다 크다. 가장 키가 작은 사람은?", ["c"]),
        ("형은 10살이고 동생은 형 나이의 절반이다. 형이 50살이 되면 동생은 몇 살인가?", ["45"]),
        ("방 안에 초가 10개 켜져 있었는데 바람에 3개가 꺼졌다. 끝까지 남은 초는 몇 개인가?", ["3개", "3"]),
        ("물 3리터 병과 5리터 병으로 4리터를 만드는 원리는?", ["채우", "붓", "남"]),
        ("시계가 3시 15분을 가리킬 때 시침과 분침 사이의 각도는 0도인가?", ["아니", "0도", "7.5"]),
        ("한 가족에 아들이 4명 있고, 각 아들에게는 여동생이 한 명씩 있다. 자녀는 총 몇 명인가?", ["5명", "다섯"]),
        ("동전을 세 번 던져서 모두 앞면이 나올 확률은?", ["1/8", "0.125", "8분의 1", "12.5%"]),
        ("토끼와 거북이가 100m 달리기 경주를 한다. 토끼가 2배 빠르지만 중간에 절반 거리를 자면 누가 이길까?", ["조건", "계산", "시간"]),
        ("거짓말쟁이 마을 사람과 참말쟁이 마을 사람 중 한 명에게 길을 물어볼 때 해야 하는 질문은?", ["다른", "반대", "물어"]),
        ("1부터 10까지의 자연수 합은 얼마인가?", ["55"]),
        ("양말 서랍에 검은 양말 10켤레, 흰 양말 10켤레가 있다. 짝이 맞는 한 켤레를 꺼내려면 최소 몇 개를 꺼내야 하는가?", ["3"]),
        ("기차가 터널을 완전히 통과한다는 것은 기차의 앞부분이 들어갈 때부터 뒷부분이 나올 때까지인가?", ["맞", "뒷부분", "통과"]),
        ("철수는 영희보다 빠르고 영희는 민수보다 느리지 않다. 민수는 철수보다 빠른가?", ["아니", "느리", "모르"]),
    ]
    for p, kw in reasoning:
        items.append({"category": "reasoning", "prompt": p, "keywords": kw})

    # 6. Casual chat (15)
    casual = [
        "요즘 어떻게 지내?", "오늘 날씨 되게 좋다.", "주말에 보통 뭐 하고 시간 보내?", "심심한데 이야기 하나 해줘",
        "너 MBTI가 뭐야?", "오늘 하루도 참 바쁘게 지나갔네.", "커피 한잔 마시면서 쉬는 중이야.", "밤에 잠 안 올 때 뭐 해?",
        "나 오늘 새로운 취미 시작했어.", "시간이 왜 이렇게 빨리 가는지 모르겠어.", "점심 맛있게 먹었어?", "벌써 일요일 저녁이네.",
        "재미있는 드라마 보는 거 있어?", "오늘 아침 개운하게 일어났어.", "갑자기 여행 떠나고 싶다.",
    ]
    for p in casual:
        items.append({"category": "casual chat", "prompt": p})

    # 7. Empathy (15)
    empathy = [
        "오늘 기분이 너무 안 좋고 우울해", "내일 중요한 시험인데 공부하기 너무 싫어", "친구랑 사소한 일로 다퉈서 속상해",
        "회사에서 큰 실수해서 자책하게 돼", "몸이 으슬으슬 춥고 감기 기운이 있어", "취업 면접에서 떨어져서 너무 슬퍼",
        "미래에 대한 걱정이 너무 많아", "퇴근하고 집에 왔는데 아무것도 하기 싫어", "인간관계가 너무 피곤하고 지쳐",
        "열심히 했는데 아무도 안 알아주는 것 같아", "반려동물이 아파서 마음이 아파", "요즘 번아웃이 온 것 같아",
        "상사한테 이유 없이 깨져서 눈물 났어", "다이어트 실패해서 자존감 떨어져", "오늘 하루 고생 많았다고 위로해줘",
    ]
    for p in empathy:
        items.append({"category": "empathy", "prompt": p})

    # 8. Recommendation (15)
    rec = [
        "오늘 점심 뭐 먹을까?", "배고픈데 메뉴 추천해줘", "비 오는 날 점심 메뉴로 뭐가 좋아?", "주말에 볼 만한 넷플릭스 영화 추천해줘",
        "저녁 산책할 때 듣기 좋은 노래 추천해줘", "혼자 조용히 힐링하기 좋은 국내 여행지는?", "친구 생일 선물 3만 원대 추천해줘",
        "스트레스 풀리는 매콤한 음식 추천해줘", "울산 데이트 코스 추천해줘", "출근길에 듣기 좋은 신나는 노래는?",
        "겨울에 먹기 좋은 따뜻한 국물 요리는?", "혼밥하기 편한 식당 메뉴 추천해줘", "키우기 쉬운 반려식물 추천해줘",
        "퇴근 후 맥주 안주 추천해줘", "자취생 초간단 10분 요리 알려줘",
    ]
    for p in rec:
        items.append({"category": "recommendation", "prompt": p})

    # 9. Clarification (15)
    clarification = [
        "그거 얼마야?", "거기 어떻게 가?", "그 영화 재미있어?", "그때 몇 시에 만나기로 했지?",
        "이거 어떻게 쓰는 물건이야?", "그 사람 이름이 뭐더라?", "그 책 결말이 어떻게 돼?",
        "다음 주에 비 오나?", "여기 와이파이 비밀번호가 뭐야?", "오늘 몇 시에 문 닫아?",
        "그거 무슨 맛이야?", "너 아까 뭐라고 했어?", "이거 먹어도 되는 거야?",
        "저 사람 왜 저러고 있어?", "그 회의 언제 끝나?",
    ]
    for p in clarification:
        items.append({"category": "clarification", "prompt": p})

    # 10. Slang (15)
    slang = [
        "게이야", "이번 판 억까 지리네", "완전 꿀잼이다 ㅋㅋㅋ", "이거 완전 갓생 살기 프로젝트 아니냐",
        "폼 미쳤다 진짜", "개이득 봤다 오늘", "킹받네 진짜 왜 이러지", "이게 실화냐?", "뇌절 오지게 하네",
        "존맛탱 추천 좀", "레전드 찍었다 오늘", "너 완전 T발 C야?", "존버는 승리한다", "현타 씨게 오네", "오운완 인증한다"
    ]
    for p in slang:
        items.append({"category": "slang", "prompt": p})

    # 11. Short utterance (15)
    short = [
        "어", "밥", "헐", "그래?", "야", "엥?", "머꼬", "진짜?", "와", "대박",
        "왜", "뭐해", "졸려", "배고파", "심심해"
    ]
    for p in short:
        items.append({"category": "short utterance", "prompt": p})

    # 12. Ulsan QA (20)
    ulsan = [
        ("울산에서 놀러갈 만한 데 어디 있어?", ["태화강", "대왕암", "간절곶"]),
        ("너 울산 사람이야?", ["울산", "사람"]),
        ("태화강 국가정원이 어떤 곳이야?", ["십리대숲", "대나무", "국가정원"]),
        ("대왕암공원 출렁다리 가봤나?", ["출렁다리", "바위", "일산"]),
        ("간절곶 해돋이 보러 가기 좋아?", ["일출", "동해", "소망우체통"]),
        ("울산 언양 불고기 뭐가 그렇게 맛있어?", ["석쇠", "한우", "불고기"]),
        ("울산 삼산동에 맛집이나 놀거리 많아?", ["번화가", "삼산", "백화점"]),
        ("울산 장생포 고래문화마을 어때?", ["고래", "장생포"]),
        ("울산 방어진 쪽 가면 회 먹을 수 있나?", ["방어진", "활어", "회"]),
        ("울산 영남알프스 등산 코스 추천해줘?", ["간월재", "신불산", "억새"]),
        ("울산 동구랑 남구 연결하는 대교는?", ["울산대교"]),
        ("울산 정자 해변은 몽돌 해수욕장 맞아?", ["몽돌", "자갈"]),
        ("울산 KTX역 이름이 뭐야?", ["울산역", "통도사"]),
        ("울산 슬도 가봤어? 분위기 어때?", ["등대", "방파제", "바람"]),
        ("울산 문수축구경기장 축구 보기 좋아?", ["울산 HD", "문수"]),
        ("울산 현대자동차 공장이 세계 최대 규모야?", ["단일", "공장", "최대"]),
        ("울산 반구대 암각화의 역사적 가치는?", ["선사", "고래", "바위"]),
        ("울산 봄 벚꽃 명소는 어디야?", ["무거천", "궁거랑"]),
        ("울산 병영 막창 골목 유명해?", ["막창", "칼국수"]),
        ("울산 옥동 대공원 규모가 커?", ["울산대공원", "자연", "크"]),
    ]
    for p, kw in ulsan:
        items.append({"category": "Ulsan QA", "prompt": p, "keywords": kw})

    # 13. Multi-turn (15)
    multi = [
        ([("나 이번 주말에 울산 놀러가려고 해", "오 진짜가? 울산 오면 갈 데 많데이! 며칠 일정으로 오는데?")], "1박 2일로 갈 건데 바다 보고 싶어"),
        ([("파이썬 공부 막 시작했어", "잘 생각했데이! 파이썬이 문법 깔끔해서 처음 배우기 딱 좋제. 어디까지 봤노?")], "변수랑 반복문 배웠는데 다음엔 뭐 공부해야 돼?"),
        ([("오늘 감기 걸린 것 같아", "아이고, 요즘 환절기라 감기 환자 많데이. 열은 안 나나?")], "열은 없고 목만 좀 칼칼해"),
        ([("점심에 라면 먹을까 김치찌개 먹을까?", "얼큰한 김치찌개가 든든하지 않겠나!")], "좋아, 김치찌개 먹으러 간다"),
        ([("내일 친구 생일이야", "축하해줄 일이네! 선물은 미리 준비했나?")], "아직 못 샀는데 케이크만 사갈까?"),
        ([("너 혹시 노래 부르는 거 좋아해?", "노래 부르는 건 좋아하지만 목소리가 없어서 아쉽제!")], "나중에 노래도 한번 불러줘라"),
        ([("나 다이어트 시작했다", "오! 결심 대단하네. 운동으로 빼나, 식단으로 빼나?")], "저녁에 야식 안 먹기부터 해보려고"),
        ([("오늘 서울 출장 왔어", "서울 사람 복잡할 텐데 고생 많데이! 길은 잘 찾았나?")], "지하철 환승이 너무 복잡하더라"),
        ([("나 새 노트북 사려고 해", "용도가 뭔데? 게임용이가, 사무용이가?")], "주로 코딩이랑 영상 편집용으로 쓸 거야"),
        ([("오늘 날씨 너무 덥다", "진짜 오늘 햇빛 쨍쨍하데이. 물 자주 마셔라!")], "시원한 아이스 아메리카노 한잔 마셔야겠다"),
        ([("요즘 운동 시작했거든", "오 무슨 운동 시작했노? 헬스가, 러닝이가?")], "퇴근하고 가볍게 달리기 시작했어"),
        ([("주말에 가족 모임 있어", "가족들 모이면 맛있는 거 묵어야제! 식당은 잡았나?")], "언양 쪽에 불고기 먹으러 가기로 했어"),
        ([("스마트폰 바꿀 때가 된 것 같아", "지금 쓰는 기종이 오래됐나? 약정 끝났나?")], "배터리가 너무 빨리 닳아서 답답해"),
        ([("카페 창업 준비 중이야", "창업 준비하느라 신경 쓸 거 많제! 인테리어는 시작했나?")], "상권 분석이랑 메뉴 선정부터 하고 있어"),
        ([("퇴근길에 비가 갑자기 오네", "우산은 챙겨 나갔었나? 감기 안 걸리게 조심해라!")], "우산 없어서 지하철역 앞에서 기다리는 중이야"),
    ]
    for history, p in multi:
        items.append({"category": "multi-turn", "prompt": p, "history": history})

    return items


def evaluate_single_response(prompt_item: dict, response: str) -> tuple[str, list[str]]:
    """Returns rating ('Correct', 'Partial', 'Fail') and list of tags."""
    cat = prompt_item["category"]
    prompt = prompt_item["prompt"]
    tags = []

    # 1. Echo check
    clean_p = re.sub(r"[^\w\s]", "", prompt).strip()
    clean_r = re.sub(r"[^\w\s]", "", response).strip()
    if len(clean_p) >= 4 and (clean_r == clean_p or (clean_p in clean_r and len(clean_r) < len(clean_p) * 1.25)):
        tags.append("ECHO")
        return "Fail", tags

    # 2. Multilingual leakage check
    if CJK_REGEX.search(response):
        tags.append("MULTILINGUAL_LEAKAGE")

    # 3. Repetition loop check
    if REPETITION_REGEX.search(response):
        tags.append("REPETITION")

    # 4. Factual error check
    keywords = prompt_item.get("keywords", [])
    if keywords:
        matched = any(kw.lower() in response.lower() for kw in keywords)
        if not matched:
            tags.append("FACTUAL_ERROR")

    # 5. Hallucination / Bizarre responses
    bizarre = ["당근마켓", "노예", "소월드", "퇴학하라", "죽어라"]
    if any(b in response for b in bizarre):
        tags.append("BIZARRE_HALLUCINATION")

    # 6. Dialect style check
    has_dialect = any(dm in response for dm in DIALECT_MARKERS)
    if has_dialect:
        tags.append("NATURAL_ULSAN")
    elif len(response.split()) >= 8:
        tags.append("STANDARD_ONLY")

    # Rating decision:
    # Fail if factual error, echo, bizarre hallucination, repetition
    if "FACTUAL_ERROR" in tags or "ECHO" in tags or "BIZARRE_HALLUCINATION" in tags or "REPETITION" in tags:
        return "Fail", tags
    elif "MULTILINGUAL_LEAKAGE" in tags or "STANDARD_ONLY" in tags:
        return "Partial", tags
    else:
        return "Correct", tags


def generate_resp(model, tokenizer, prompt_item: dict, seed: int = 42) -> str:
    set_seed(seed)
    prompt = prompt_item["prompt"]
    history = prompt_item.get("history", [])

    messages = [{"role": "system", "content": DEFAULT_SYSTEM_PROMPT}]
    if history:
        for u, a in history:
            messages.append({"role": "user", "content": u})
            messages.append({"role": "assistant", "content": a})
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
            max_new_tokens=150,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
        )
    gen_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()


def main():
    parser = argparse.ArgumentParser(description="Run 200 prompt regression comparing Phase3 Best vs Phase4 DPO")
    parser.add_argument("--phase3-model", default="outputs/ulm-1.7b-phase3-best-merged")
    parser.add_argument("--phase4-base", default="outputs/ulm-1.7b-phase4-sft-interim")
    parser.add_argument("--phase4-dpo", default="outputs/ulm-1.7b-phase4-dpo-l40s")
    parser.add_argument("--output-json", default="reports/phase4_final_regression_200.json")
    args = parser.parse_args()

    prompts = load_200_prompts()
    print(f"Loaded {len(prompts)} test prompts across 13 categories.")

    tokenizer = AutoTokenizer.from_pretrained(args.phase3_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 1. Run Phase 3 Best (Baseline)
    print("\n--- [1/2] Generating with Phase 3 Best Model ---")
    p3_model = AutoModelForCausalLM.from_pretrained(args.phase3_model, torch_dtype=torch.bfloat16, device_map="cuda:0")
    p3_model.eval()
    p3_results = []
    p3_start = time.time()
    for idx, item in enumerate(prompts, 1):
        resp = generate_resp(p3_model, tokenizer, item, seed=42 + idx)
        rating, tags = evaluate_single_response(item, resp)
        p3_results.append({"prompt": item["prompt"], "category": item["category"], "response": resp, "rating": rating, "tags": tags})
        if idx % 50 == 0 or idx == len(prompts):
            print(f"  Phase 3 progress: {idx}/{len(prompts)} ({time.time() - p3_start:.1f}s)")

    # Dialect preservation for Phase 3
    p3_dialect_test = {}
    for p in DIALECT_PRESERVATION_PROMPTS:
        p3_dialect_test[p] = generate_resp(p3_model, tokenizer, {"prompt": p, "category": "dialect_test"}, seed=42)

    del p3_model
    torch.cuda.empty_cache()

    # 2. Run Phase 4 DPO
    print("\n--- [2/2] Generating with Phase 4 DPO Model ---")
    p4_base = AutoModelForCausalLM.from_pretrained(args.phase4_base, torch_dtype=torch.bfloat16, device_map="cuda:0")
    p4_model = PeftModel.from_pretrained(p4_base, args.phase4_dpo)
    p4_model.eval()
    p4_results = []
    p4_start = time.time()
    for idx, item in enumerate(prompts, 1):
        resp = generate_resp(p4_model, tokenizer, item, seed=42 + idx)
        rating, tags = evaluate_single_response(item, resp)
        p4_results.append({"prompt": item["prompt"], "category": item["category"], "response": resp, "rating": rating, "tags": tags})
        if idx % 50 == 0 or idx == len(prompts):
            print(f"  Phase 4 progress: {idx}/{len(prompts)} ({time.time() - p4_start:.1f}s)")

    # Dialect preservation for Phase 4
    dialect_test = []
    for p in DIALECT_PRESERVATION_PROMPTS:
        r4 = generate_resp(p4_model, tokenizer, {"prompt": p, "category": "dialect_test"}, seed=42)
        r3 = p3_dialect_test[p]
        dialect_test.append({"prompt": p, "phase3": r3, "phase4": r4})
        print(f"\n[Dialect Test] Prompt: {p}")
        print(f"  P3: {r3}")
        print(f"  P4: {r4}")

    del p4_model
    del p4_base
    torch.cuda.empty_cache()

    # Summary calculations
    def summarize(res_list):
        total = len(res_list)
        correct = sum(1 for r in res_list if r["rating"] == "Correct")
        partial = sum(1 for r in res_list if r["rating"] == "Partial")
        fail = sum(1 for r in res_list if r["rating"] == "Fail")
        pass_rate = (correct + partial) / total * 100
        correct_rate = correct / total * 100
        cat_stats = {}
        tag_counts = {}
        for r in res_list:
            c = r["category"]
            cat_stats.setdefault(c, {"total": 0, "correct": 0, "partial": 0, "fail": 0})
            cat_stats[c]["total"] += 1
            cat_stats[c][r["rating"].lower()] += 1
            for t in r["tags"]:
                tag_counts[t] = tag_counts.get(t, 0) + 1
        return {
            "total": total,
            "correct": correct,
            "partial": partial,
            "fail": fail,
            "pass_rate_pct": round(pass_rate, 2),
            "correct_rate_pct": round(correct_rate, 2),
            "tag_counts": tag_counts,
            "category_stats": cat_stats,
        }

    summary_p3 = summarize(p3_results)
    summary_p4 = summarize(p4_results)

    final_report = {
        "phase3_summary": summary_p3,
        "phase4_summary": summary_p4,
        "dialect_test": dialect_test,
        "comparisons": [
            {
                "prompt": p3["prompt"],
                "category": p3["category"],
                "p3_response": p3["response"],
                "p3_rating": p3["rating"],
                "p3_tags": p3["tags"],
                "p4_response": p4["response"],
                "p4_rating": p4["rating"],
                "p4_tags": p4["tags"],
            } for p3, p4 in zip(p3_results, p4_results)
        ]
    }

    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(final_report, f, ensure_ascii=False, indent=2)

    print(f"\n==========================================")
    print(f"200-PROMPT FINAL REGRESSION COMPARISON:")
    print(f"Phase 3 Best: Pass Rate = {summary_p3['pass_rate_pct']}% (Correct: {summary_p3['correct']}, Partial: {summary_p3['partial']}, Fail: {summary_p3['fail']})")
    print(f"Phase 4 DPO:  Pass Rate = {summary_p4['pass_rate_pct']}% (Correct: {summary_p4['correct']}, Partial: {summary_p4['partial']}, Fail: {summary_p4['fail']})")
    print(f"Report saved to {args.output_json}")
    print(f"==========================================")


if __name__ == "__main__":
    main()
