"""Comprehensive 350-Prompt Failure Collection & Evaluation Suite for ULM-1.7B Phase 3 Best.

Evaluates 14 categories (25 prompts each = 350 prompts) and categorizes failures:
- FACTUAL_ERROR
- HALLUCINATION
- IRRELEVANT
- ECHO
- REPETITION
- LANGUAGE_LEAKAGE
- BAD_REASONING
- BAD_RECOMMENDATION
- BAD_EMPATHY
- DIALECT_OVERUSE
- DIALECT_UNDERUSE
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

DEFAULT_SYSTEM_PROMPT = (
    "울산 지역어 대화 assistant로서 자연스럽고 일상적인 울산 사투리로 상대방과 친근하게 대화한다. "
    "자연스럽고 편안한 일상 울산 말투를 기본으로 구사한다."
)

CJK_REGEX = re.compile(r"[\u4e00-\u9fff\u3040-\u30ff]")
REPETITION_REGEX = re.compile(r"(.{4,})\1{2,}")

# Dialect markers in Gyeongsang / Ulsan dialect
DIALECT_MARKERS = [
    "데이", "아이가", "이제", "했나", "왔나", "있나", "카나", "노", "고", "했데이",
    "맞제", "그라믄", "머꼬", "와이라노", "와이래", "단디", "쫌", "억수로", "카더라", "항상"
]

def load_prompts_dataset() -> list[dict]:
    # 14 categories x 25 prompts = 350 prompts
    dataset = []

    # 1. factual QA (25)
    factual = [
        ("fact_01", "대한민국의 수도는 어디야?", ["서울"]),
        ("fact_02", "지구에서 가장 높은 산의 이름은 뭐야?", ["에베레스트", "사resource"]),
        ("fact_03", "임진왜란은 몇 년도에 일어났어?", ["1592"]),
        ("fact_04", "태양계에서 가장 큰 행성은 뭐야?", ["목성", "Jupiter"]),
        ("fact_05", "한글을 창제한 조선의 국왕은?", ["세종", "세종대왕"]),
        ("fact_06", "피타고라스 정리가 뭐야?", ["직각", "빗변", "제곱"]),
        ("fact_07", "세계에서 가장 면적이 넓은 국가는?", ["러시아"]),
        ("fact_08", "올림픽 하계 대회는 보통 몇 년 주기로 열려?", ["4년"]),
        ("fact_09", "인류가 최초로 달에 착륙한 해는 언제야?", ["1969"]),
        ("fact_10", "피의 순환을 밝혀낸 영국의 의사는 누구야?", ["하비", "윌리엄"]),
        ("fact_11", "주기율표에서 원소 기호 O는 무슨 원소야?", ["산소"]),
        ("fact_12", "프랑스의 수도는 어디야?", ["파리", "Paris"]),
        ("fact_13", "빛의 진공 속 속도는 초당 약 몇 킬로미터야?", ["30만", "299,792", "300,000"]),
        ("fact_14", "물 분자를 이루는 화학식은 뭐야?", ["H2O", "수소", "산소"]),
        ("fact_15", "피아노 건반은 총 몇 개야?", ["88"]),
        ("fact_16", "이순신 장군이 13척의 배로 승리한 해전은?", ["명량", "명량대첩"]),
        ("fact_17", "세계에서 가장 긴 강은 어디야?", ["나일", "아마존"]),
        ("fact_18", "미국의 초대 대통령은 누구야?", ["워싱턴", "조지"]),
        ("fact_19", "지구 자전 주기는 대략 몇 시간이야?", ["24시간", "하루"]),
        ("fact_20", "체온계의 표준 사람 정상 체온은 대략 몇 도야?", ["36.5", "36~37"]),
        ("fact_21", "우리나라 헌법 제1조 1항의 내용은 뭐야?", ["민주공화국"]),
        ("fact_22", "삼국통일을 이룬 신라의 왕은 누구야?", ["문무왕"]),
        ("fact_23", "셰익스피어의 4대 비극 중 하나만 말해줘", ["햄릿", "오셀로", "리어왕", "맥베스"]),
        ("fact_24", "태극기의 네 괘는 건, 곤, 감, 그리고 뭐야?", ["리", "이"]),
        ("fact_25", "달의 인력 때문에 생기는 바다의 밀물과 썰물 현상을 뭐라고 해?", ["조석", "조수간만"]),
    ]
    for pid, p, kw in factual:
        dataset.append({"id": pid, "category": "factual QA", "prompt": p, "ground_truth_keywords": kw})

    # 2. science (25)
    science = [
        ("sci_01", "왜 하늘은 파래?", ["산란", "파란", "빛", "파장"]),
        ("sci_02", "물이 100도에서 끓는 이유가 뭐야?", ["기압", "증기압", "기포", "상태"]),
        ("sci_03", "중력이란 무엇인가요?", ["끌어당기는", "질량", "힘", "만유인력"]),
        ("sci_04", "달의 모양이 매일 바뀌는 이유는?", ["공전", "햇빛", "위치", "각도"]),
        ("sci_05", "얼음이 왜 물 위에 떠?", ["밀도", "부피", "가벼"]),
        ("sci_06", "식물이 광합성을 하는 이유가 뭐야?", ["양분", "포도당", "빛", "에너지"]),
        ("sci_07", "번개가 칠 때 천둥소리가 나중에 들리는 이유가 뭐야?", ["빛", "소리", "속도", "음속"]),
        ("sci_08", "바닷물이 짠 이유는 뭐야?", ["염분", "소금", "비", "광물"]),
        ("sci_09", "비행기가 하늘을 날 수 있는 원리가 뭐야?", ["양력", "베르누이", "날개"]),
        ("sci_10", "무지개는 어떻게 생겨?", ["굴절", "반사", "물방울", "분산"]),
        ("sci_11", "자석이 쇠를 끌어당기는 원리가 뭐야?", ["자기장", "전자", "스핀"]),
        ("sci_12", "가을에 나뭇잎이 붉게 물드는 이유는?", ["엽록소", "안토시아닌", "단풍"]),
        ("sci_13", "사람은 왜 숨을 쉬어야 해?", ["산소", "에너지", "세포", "이산화탄소"]),
        ("sci_14", "온실효과가 뭐야?", ["열", "온실가스", "지구", "보온"]),
        ("sci_15", "지진은 왜 발생해?", ["판", "단층", "지각", "충돌"]),
        ("sci_16", "철이 녹스는 이유는 뭐야?", ["산화", "산소", "물", "화합"]),
        ("sci_17", "우주에는 왜 공기가 없어?", ["진공", "중력", "대기"]),
        ("sci_18", "밀물과 썰물은 하루에 몇 번 일어나?", ["두 번", "2번"]),
        ("sci_19", "소리는 진공 중에서 전달될 수 있어?", ["없다", "매질", "진동"]),
        ("sci_20", "화산 폭발은 왜 일어나?", ["마그마", "압력", "가스", "지각"]),
        ("sci_21", "태양은 어떤 에너지로 빛을 내?", ["핵융합", "수소", "헬륨"]),
        ("sci_22", "적혈구의 주요 역할이 뭐야?", ["산소", "운반", "헤모글로빈"]),
        ("sci_23", "열전도율이 높은 물질은 어떤 게 있어?", ["금속", "은", "구리", "알루미늄"]),
        ("sci_24", "블랙홀은 무엇인가요?", ["중력", "빛", "시공간", "탈출"]),
        ("sci_25", "인간의 뼈는 대략 몇 개로 이루어져 있어?", ["206"]),
    ]
    for pid, p, kw in science:
        dataset.append({"id": pid, "category": "science", "prompt": p, "ground_truth_keywords": kw})

    # 3. math (25)
    math_items = [
        ("math_01", "1+1은 뭐야?", ["2"]),
        ("math_02", "12 * 13은 얼마야?", ["156"]),
        ("math_03", "100 나누기 4는 얼마야?", ["25"]),
        ("math_04", "2의 10승은 얼마야?", ["1024", "1,024"]),
        ("math_05", "50에서 17을 빼면 얼마고?", ["33"]),
        ("math_06", "7 곱하기 8은 얼마야?", ["56"]),
        ("math_07", "99 나누기 3은 얼마야?", ["33"]),
        ("math_08", "15 더하기 28은?", ["43"]),
        ("math_09", "루트 144는 얼마야?", ["12"]),
        ("math_10", "원주율 파이의 근삿값은 얼마야?", ["3.14"]),
        ("math_11", "300의 20%는 얼마야?", ["60"]),
        ("math_12", "소수(prime number) 2, 3, 5 다음 수는?", ["7"]),
        ("math_13", "직각삼각형의 두 변이 3과 4일 때 빗변은?", ["5"]),
        ("math_14", "5 팩토리얼(5!)은 얼마야?", ["120"]),
        ("math_15", "1000에서 357을 빼면 얼마야?", ["643"]),
        ("math_16", "25 곱하기 25는?", ["625"]),
        ("math_17", "한 변이 6cm인 정사각형의 넓이는?", ["36"]),
        ("math_18", "시속 60km로 3시간 달리면 몇 km 가?", ["180"]),
        ("math_19", "사과 12개를 3명이서 똑같이 나누면 한 명당 몇 개야?", ["4"]),
        ("math_20", "2의 8승은 얼마야?", ["256"]),
        ("math_21", "81의 제곱근은?", ["9"]),
        ("math_22", "1부터 10까지의 자연수를 모두 더하면 얼마야?", ["55"]),
        ("math_23", "48 나누기 6은 얼마야?", ["8"]),
        ("math_24", "각도가 90도인 각을 무슨 각이라고 해?", ["직각"]),
        ("math_25", "0으로 어떤 수를 나눌 수 있어?", ["없다", "불가능", "정의"]),
    ]
    for pid, p, kw in math_items:
        dataset.append({"id": pid, "category": "math", "prompt": p, "ground_truth_keywords": kw})

    # 4. coding (25)
    coding_items = [
        ("code_01", "파이썬으로 리스트 정렬하는 법 알려줘", ["sort", "sorted"]),
        ("code_02", "파이썬 딕셔너리가 뭐야?", ["키", "값", "key", "value"]),
        ("code_03", "자바스크립트에서 배열 뒤집는 함수는?", ["reverse"]),
        ("code_04", "HTML에서 링크 걸 때 무슨 태그 써?", ["a", "href"]),
        ("code_05", "SQL에서 특정 테이블의 모든 데이터 조회하는 쿼리문은?", ["SELECT", "*", "FROM"]),
        ("code_06", "깃(Git)에서 원격 저장소 내용을 내려받는 명령어는?", ["pull", "fetch", "clone"]),
        ("code_07", "C언어에서 포인터가 뭔지 쉽게 설명해줘", ["주소", "메모리", "address"]),
        ("code_08", "파이썬에서 with open 쓰는 이유가 뭐야?", ["close", "자동", "자원"]),
        ("code_09", "REST API에서 데이터를 새로 생성할 때 주로 쓰는 HTTP 메소드는?", ["POST"]),
        ("code_10", "파이썬에서 문자열을 특정 구분자로 자르는 함수는?", ["split"]),
        ("code_11", "자바스크립트에서 let과 const의 차이가 뭐야?", ["재할당", "상수", "변수"]),
        ("code_12", "이진 탐색(Binary Search)의 시간 복잡도는?", ["O(log n)", "log"]),
        ("code_13", "스택(Stack)과 큐(Queue)의 차이점이 뭐야?", ["LIFO", "FIFO", "먼저"]),
        ("code_14", "파이썬에서 리스트에 새 요소를 끝에 추가하는 함수는?", ["append"]),
        ("code_15", "데이터베이스에서 기본키(Primary Key)의 역할이 뭐야?", ["고유", "식별", "중복"]),
        ("code_16", "HTTP 상태 코드 404는 무슨 뜻이야?", ["찾을 수 없음", "Not Found"]),
        ("code_17", "객체 지향 프로그래밍(OOP)의 4대 특징 중 하나는?", ["캡슐화", "상속", "다형성", "추상화"]),
        ("code_18", "파이썬에서 조건문 쓸 때 쓰는 키워드는?", ["if", "elif", "else"]),
        ("code_19", "자바스크립트에서 비동기 처리할 때 쓰는 키워드 쌍은?", ["async", "await", "Promise"]),
        ("code_20", "도커(Docker)에서 컨테이너란 무엇인가요?", ["격리", "이미지", "가상화"]),
        ("code_21", "파이썬 가상환경을 만드는 명령어는 뭐야?", ["venv", "virtualenv", "uv"]),
        ("code_22", "CSS에서 가로 세로 중앙 정렬할 때 flexbox 어떻게 써?", ["justify-content", "align-items", "center"]),
        ("code_23", "재귀 함수(Recursive Function)를 쓸 때 탈출 조건이 왜 중요해?", ["무한 루프", "스택 오버플로우", "종료"]),
        ("code_24", "JSON이란 무엇의 약자야?", ["JavaScript Object Notation"]),
        ("code_25", "정규표현식에서 숫자를 매칭하는 메타문자는?", [r"\d", "[0-9]"]),
    ]
    for pid, p, kw in coding_items:
        dataset.append({"id": pid, "category": "coding", "prompt": p, "ground_truth_keywords": kw})

    # 5. reasoning (25)
    reasoning_items = [
        ("reas_01", "철수는 영희보다 키가 크고 영희는 민수보다 크다. 세 명 중 가장 키가 큰 사람은?", ["철수"]),
        ("reas_02", "모든 인간은 죽는다. 소크라테스는 인간이다. 그렇다면 소크라테스는?", ["죽는다"]),
        ("reas_03", "어떤 방에 초가 10자루 켜져 있었는데 바람이 불어 3자루가 꺼졌다. 끝까지 남은 초는 몇 자루?", ["3자루", "3개"]),
        ("reas_04", "어제는 내일이었고 오늘은 목요일이다. 내일은 무슨 요일일까?", ["금요일"]),
        ("reas_05", "빨간 공 3개와 파란 공 2개가 든 주머니에서 공 2개를 꺼낼 때 둘 다 파란 공일 확률은?", ["1/10", "10%"]),
        ("reas_06", "비가 오면 땅이 젖는다. 지금 땅이 젖어있다면 반드시 방금 비가 온 것일까?", ["아니다", "물", "다른 이유"]),
        ("reas_07", "동전 두 개를 동시에 던질 때 둘 다 앞면이 나올 확률은?", ["1/4", "25%"]),
        ("reas_08", "시계 바늘이 3시 15분을 가리킬 때 시침과 분침은 완벽히 포개져 있을까?", ["아니다", "시침도 움직임"]),
        ("reas_09", "10층 건물에서 엘리베이터를 타고 1층에서 5층까지 가는데 20초 걸렸다면 10층까지는 몇 초 걸릴까?", ["45초", "등속"]),
        ("reas_10", "양 세 마리가 강을 건너야 하는데 배에는 한 번에 한 마리만 탈 수 있다. 총 몇 번 건너야 할까?", ["5번"]),
        ("reas_11", "장갑 한 켤레, 양말 두 켤레, 신발 세 켤레가 있으면 총 짝수는 몇 개인가?", ["12"]),
        ("reas_12", "참말만 하는 사람과 거짓말만 하는 사람이 있을 때 길을 물어보는 유명한 논리 문제는?", ["다른 사람", "거짓말"]),
        ("reas_13", "A는 B의 아버지이고 B는 C의 아버지이다. A는 C에게 어떤 관계인가?", ["할아버지", "조부"]),
        ("reas_14", "물 1리터와 기름 1리터 중 어느 쪽이 더 무거울까?", ["물"]),
        ("reas_15", "토끼와 거북이가 100m 경주를 한다. 토끼가 두 배 빠르지만 절반 지점에서 잠을 잤다...", ["거북이", "조건"]),
        ("reas_16", "모든 사과는 과일이다. 모든 과일은 식물이다. 그렇다면 사과는 식물인가?", ["그렇다", "식물"]),
        ("reas_17", "닭과 토끼를 합쳐 10마리이고 다리 수가 총 28개라면 토끼는 몇 마리인가?", ["4마리"]),
        ("reas_18", "어두운 방에서 성냥 한 개로 난로, 촛불, 석유램프를 켤 때 가장 먼저 켜야 하는 것은?", ["성냥"]),
        ("reas_19", "상자 3개 중 하나에만 보물이 있다. 1번 상자에는 '여기 없다'라고 적혀있다...", ["논리"]),
        ("reas_20", "가위바위보에서 이길 확률, 질 확률, 비길 확률은 각각 얼마인가?", ["1/3", "33%"]),
        ("reas_21", "1kg의 쇠와 1kg의 솜 중 어느 것이 더 무거울까?", ["같다", "동일"]),
        ("reas_22", "5명이 악수를 서로 한 번씩 모두 나눈다면 총 악수 횟수는?", ["10"]),
        ("reas_23", "원형 탁자에 4명이 둘러앉는 순열의 가짓수는?", ["6"]),
        ("reas_24", "거짓말쟁이 마을에서 온 주민에게 질문할 때의 핵심은?", ["이중 부정", "진실"]),
        ("reas_25", "연속된 세 홀수의 합이 27일 때 가장 큰 홀수는?", ["11"]),
    ]
    for pid, p, kw in reasoning_items:
        dataset.append({"id": pid, "category": "reasoning", "prompt": p, "ground_truth_keywords": kw})

    # 6. recommendation (25)
    rec_items = [
        ("rec_01", "배고픈데 오늘 저녁 뭐 먹지? 맛있는 거 추천해줘"),
        ("rec_02", "비 오는 날 점심 메뉴로 뭐가 어울릴까?"),
        ("rec_03", "주말에 집에서 볼 만한 재미있는 넷플릭스 영화 추천해줘"),
        ("rec_04", "선선한 저녁에 산책하면서 듣기 좋은 노래 추천해줘"),
        ("rec_05", "혼자 조용히 떠나기 좋은 국내 힐링 여행지 추천해줄래?"),
        ("rec_06", "친구 생일 선물로 3만 원 안팎 부담 없는 거 뭐 좋을까?"),
        ("rec_07", "요즘 가볍게 읽기 좋은 인문 교양 도서 추천해줘"),
        ("rec_08", "스트레스 풀릴 만큼 매콤한 음식 추천해줘"),
        ("rec_09", "다이어트할 때 부담 없이 먹을 수 있는 외식 메뉴 있어?"),
        ("rec_10", "울산에서 데이트하기 좋은 분위기 있는 장소 추천해줘"),
        ("rec_11", "출근길에 잠 깨기 좋은 활기찬 노래 뭐가 있어?"),
        ("rec_12", "가족들이랑 주말에 가기 좋은 나들이 코스 알려줘"),
        ("rec_13", "추운 겨울에 몸 녹이기 좋은 따뜻한 국물 요리 추천해줘"),
        ("rec_14", "혼밥하기 편한 식당 메뉴 뭐 있을까?"),
        ("rec_15", "초보자가 집에서 키우기 쉬운 반려식물 추천해줘"),
        ("rec_16", "시험 끝나고 친구들이랑 신나게 놀 만한 곳 추천해줘"),
        ("rec_17", "부모님 결혼기념일 선물로 10만 원대 괜찮은 거 있을까?"),
        ("rec_18", "집중력 높일 때 마시기 좋은 따뜻한 차 추천해줘"),
        ("rec_19", "퇴근하고 맥주 한잔이랑 곁들이기 좋은 안주 추천해줘"),
        ("rec_20", "가을 단풍 구경하기 좋은 명소 어디가 있어?"),
        ("rec_21", "프로그래밍 처음 배우는데 무슨 언어로 시작하면 좋을까?"),
        ("rec_22", "간단하게 10분 만에 해 먹을 수 있는 자취 요리 알려줘"),
        ("rec_23", "울산 바다 구경하면서 커피 마시기 좋은 카페 거리 어디야?"),
        ("rec_24", "친구랑 캠핑 처음 가는데 챙겨야 할 필수 아이템 추천해줘"),
        ("rec_25", "자기 전에 마음 편안해지는 잔잔한 팟캐스트나 음악 추천해줘"),
    ]
    for pid, p in rec_items:
        dataset.append({"id": pid, "category": "recommendation", "prompt": p})

    # 7. casual chat (25)
    casual_items = [
        ("cas_01", "요즘 어떻게 지내?"),
        ("cas_02", "오늘 날씨 되게 좋다."),
        ("cas_03", "주말에 보통 뭐 하고 시간 보내?"),
        ("cas_04", "심심한데 재미있는 이야기 하나 해줘"),
        ("cas_05", "너 MBTI가 뭐야?"),
        ("cas_06", "오늘 하루도 참 바쁘게 지나갔네."),
        ("cas_07", "커피 한잔 마시면서 잠깐 쉬는 중이야."),
        ("cas_08", "너는 밤에 잠 안 올 때 뭐 해?"),
        ("cas_09", "나 오늘 새로운 취미 시작했어."),
        ("cas_10", "시간이 왜 이렇게 빨리 가는지 모르겠어."),
        ("cas_11", "점심 맛있게 먹었어?"),
        ("cas_12", "벌써 일요일 저녁이라니 믿기지가 않는다."),
        ("cas_13", "요즘 재미있는 드라마나 예능 보는 거 있어?"),
        ("cas_14", "오늘 아침에 일어났는데 너무 개운하더라."),
        ("cas_15", "너는 아침형 인간이야 아니면 올빼미족이야?"),
        ("cas_16", "갑자기 여행 떠나고 싶다는 생각이 든다."),
        ("cas_17", "계절 중에 어느 계절을 제일 좋아해?"),
        ("cas_18", "오늘 퇴근길에 하늘 보니까 노을이 예쁘더라."),
        ("cas_19", "반려동물 키우는 거 좋아해?"),
        ("cas_20", "친한 친구 만나서 수다 떨고 왔어."),
        ("cas_21", "너는 음악 장르 어떤 거 주로 들어?"),
        ("cas_22", "올해 세운 목표들 잘 지키고 있어?"),
        ("cas_23", "주말에 비 온다고 하던데 실내에서 뭐할까?"),
        ("cas_24", "오늘 점심에 먹은 찌개가 진짜 꿀맛이었어."),
        ("cas_25", "내일 월요일이라는 게 너무 슬프다."),
    ]
    for pid, p in casual_items:
        dataset.append({"id": pid, "category": "casual chat", "prompt": p})

    # 8. empathy (25)
    empathy_items = [
        ("emp_01", "오늘 기분이 너무 안 좋고 우울해"),
        ("emp_02", "내일 중요한 시험인데 공부하기 너무 싫고 불안하다"),
        ("emp_03", "친한 친구랑 사소한 일로 다퉈서 마음이 너무 복잡해"),
        ("emp_04", "회사에서 큰 실수를 해서 하루 종일 눈치 보이고 자책하게 돼"),
        ("emp_05", "몸이 으슬으슬 춥고 감기 기운이 있어서 서럽네"),
        ("emp_06", "열심히 준비했던 취업 면접에서 떨어졌어... 너무 속상해"),
        ("emp_07", "요즘 미래에 대한 걱정이 너무 많아서 밤마다 잠을 설쳐"),
        ("emp_08", "퇴근하고 집에 왔는데 아무것도 하기 싫고 무기력해"),
        ("emp_09", "인간관계가 너무 피곤하고 다 부질없게 느껴져"),
        ("emp_10", "열심히 일했는데 아무도 내 노력을 인정해주지 않는 것 같아"),
        ("emp_11", "가족들이랑 크게 싸우고 집 나왔는데 갈 데가 없다"),
        ("emp_12", "반려동물이 아파서 병원 다녀왔는데 마음이 너무 아파"),
        ("emp_13", "요즘 번아웃이 온 것 같아. 쉬어도 피로가 안 풀려"),
        ("emp_14", "주변 친구들은 다 잘나가는 것 같은데 나만 제자리인 것 같아"),
        ("emp_15", "오랜 연애 끝에 오늘 헤어졌어... 마음이 너무 텅 빈 것 같아"),
        ("emp_16", "프로젝트 마감이 코앞인데 손에 잡히질 않아서 미치겠어"),
        ("emp_17", "나이만 먹어가고 이뤄놓은 건 없는 것 같아서 우울해"),
        ("emp_18", "오늘 상사한테 이유 없이 깨져서 서러워 눈물 났어"),
        ("emp_19", "다들 나한테 바라는 건 많은데 내 마음을 알아주는 사람은 없어"),
        ("emp_20", "다이어트 계속 실패하니까 자존감이 바닥을 치네"),
        ("emp_21", "아침에 알람 못 듣고 지각해서 하루 시작부터 망했어"),
        ("emp_22", "새로운 환경에 적응하기가 너무 버겁고 외로워"),
        ("emp_23", "돈 모으기는 힘들고 물가는 너무 올라서 막막하다"),
        ("emp_24", "믿었던 사람한테 배신당해서 충격이 너무 커"),
        ("emp_25", "그냥 오늘 하루 고생 많았다고 한마디만 해줄래?"),
    ]
    for pid, p in empathy_items:
        dataset.append({"id": pid, "category": "empathy", "prompt": p})

    # 9. clarification (25)
    clar_items = [
        ("clar_01", "아니 그게 도대체 무슨 말이야?"),
        ("clar_02", "뭐라고? 다시 한번 말해줘"),
        ("clar_03", "방금 한 설명이 너무 어려워서 이해가 안 가"),
        ("clar_04", "조금 더 쉽게 초등학생도 알 수 있게 풀어서 설명해줄래?"),
        ("clar_05", "구체적인 예를 하나 들어서 다시 설명해봐"),
        ("clar_06", "무슨 뜻인지 감이 잘 안 오는데 핵심만 요약해줘"),
        ("clar_07", "방금 말한 단어의 정확한 의미가 뭐야?"),
        ("clar_08", "어디서부터 잘못된 건지 짚어줄 수 있어?"),
        ("clar_09", "한 번만 더 차근차근 짚어주라"),
        ("clar_10", "앞뒤 문맥이 안 맞는 것 같은데 다시 설명해줘"),
        ("clar_11", "그래서 결론이 뭐라는 거야?"),
        ("clar_12", "내가 이해한 게 맞는지 확인해줘: A라는 뜻이야?"),
        ("clar_13", "전문 용어 쓰지 말고 일상어로 말해줘"),
        ("clar_14", "너무 길어서 헷갈려. 세 줄로 요약해줄래?"),
        ("clar_15", "둘의 차이점이 정확히 뭔지 명확하게 대비해줘"),
        ("clar_16", "왜 그런 결론이 나왔는지 이유를 덧붙여줘"),
        ("clar_17", "방금 그 말 농담이야 진담이야?"),
        ("clar_18", "어떤 관점에서 그렇게 말한 건지 설명해줘"),
        ("clar_19", "그게 가능하려면 어떤 조건이 필요한데?"),
        ("clar_20", "방금 한 말에 모순이 있는 것 같은데 해명해봐"),
        ("clar_21", "다시 말해주겠니? 못 들었어"),
        ("clar_22", "간단히 O/X로 대답해줄래?"),
        ("clar_23", "다른 비유를 들어서 설명해줄 수 있어?"),
        ("clar_24", "질문의 요지를 잘 파악한 건지 다시 확인해줘"),
        ("clar_25", "정확하게 숫자로 근거를 들어줄래?"),
    ]
    for pid, p in clar_items:
        dataset.append({"id": pid, "category": "clarification", "prompt": p})

    # 10. short utterance (25)
    short_items = [
        ("short_01", "어"),
        ("short_02", "밥"),
        ("short_03", "헐"),
        ("short_04", "그래?"),
        ("short_05", "야"),
        ("short_06", "엥?"),
        ("short_07", "머꼬"),
        ("short_08", "진짜?"),
        ("short_09", "와"),
        ("short_10", "대박"),
        ("short_11", "왜"),
        ("short_12", "뭐해"),
        ("short_13", "졸려"),
        ("short_14", "배고파"),
        ("short_15", "심심해"),
        ("short_16", "안녕"),
        ("short_17", "잘자"),
        ("short_18", "오잉"),
        ("short_19", "맞나"),
        ("short_20", "그치"),
        ("short_21", "싫어"),
        ("short_22", "좋아"),
        ("short_23", "귀찮아"),
        ("short_24", "아싸"),
        ("short_25", "흠"),
    ]
    for pid, p in short_items:
        dataset.append({"id": pid, "category": "short utterance", "prompt": p})

    # 11. slang (25)
    slang_items = [
        ("slang_01", "게이야"),
        ("slang_02", "이번 판 억까 지리네"),
        ("slang_03", "완전 꿀잼이다 ㅋㅋㅋ"),
        ("slang_04", "이거 완전 갓생 살기 프로젝트 아니냐"),
        ("slang_05", "폼 미쳤다 진짜"),
        ("slang_06", "개이득 봤다 오늘"),
        ("slang_07", "킹받네 진짜 왜 이러지"),
        ("slang_08", "이게 실화냐?"),
        ("slang_09", "뇌절 오지게 하네"),
        ("slang_10", "존맛탱 추천 좀"),
        ("slang_11", "레전드 찍었다 오늘"),
        ("slang_12", "갑분싸 만들지 마라"),
        ("slang_13", "너 완전 T발 C야?"),
        ("slang_14", "극대노 할 뻔했다"),
        ("slang_15", "존버는 승리한다"),
        ("slang_16", "비주얼 미쳤다리"),
        ("slang_17", "완내스(완전 내 스타일)다 이거"),
        ("slang_18", "오늘 룩북 완전 꾸안꾸네"),
        ("slang_19", "중꺾마 정신으로 버틴다"),
        ("slang_20", "이 분위기 모르면 나가라"),
        ("slang_21", "현타 씨게 오네"),
        ("slang_22", "인생 역대급 뻘짓했다"),
        ("slang_23", "답정너 같으니라고"),
        ("slang_24", "오늘 하루 완전 순삭당함"),
        ("slang_25", "오운완 인증한다"),
    ]
    for pid, p in slang_items:
        dataset.append({"id": pid, "category": "slang", "prompt": p})

    # 12. 울산 local QA (25)
    ulsan_items = [
        ("ulsan_01", "울산에서 놀러갈 만한 데 어디 있어?", ["태화강", "대왕암", "간절곶"]),
        ("ulsan_02", "너 울산 사람이야?", ["울산", "사람"]),
        ("ulsan_03", "태화강 국가정원이 어떤 곳이야?", ["십리대숲", "대나무", "국가정원"]),
        ("ulsan_04", "대왕암공원 출렁다리 가봤나?", ["일산지", "출렁다리", "바위"]),
        ("ulsan_05", "간절곶 해돋이 보러 가기 좋아?", ["해맞이", "동해", "소망우체통"]),
        ("ulsan_06", "울산 언양 불고기 뭐가 그렇게 맛있어?", ["석쇠", "한우", "언양"]),
        ("ulsan_07", "울산 삼산동에 맛집이나 놀거리 많아?", ["번화가", "삼산", "롯데백화점"]),
        ("ulsan_08", "울산 공업축제 알아?", ["공업", "축제", "퍼레이드"]),
        ("ulsan_09", "울산 장생포 고래문화마을 어때?", ["고래", "모노레일", "장생포"]),
        ("ulsan_10", "울산 방어진 쪽 가면 회 맛있게 먹을 수 있나?", ["방어진", "활어", "시장"]),
        ("ulsan_11", "울산 영남알프스 등산 코스 추천해줘", ["간월재", "신불산", "억새"]),
        ("ulsan_12", "울산 동구랑 남구는 울산대교로 건너갈 수 있어?", ["울산대교", "연결"]),
        ("ulsan_13", "울산 과학관 아이들이랑 가기 괜찮아?", ["체험", "교육"]),
        ("ulsan_14", "울산 정자 해수욕장은 몽돌 해변이야?", ["몽돌", "자갈"]),
        ("ulsan_15", "울산 KTX역은 삼산동에 있어?", ["통도사", "언양", "삼남"]),
        ("ulsan_16", "울산 슬도 가봤어? 분위기 어때?", ["방파제", "바람", "등대"]),
        ("ulsan_17", "울산 문수축구경기장 축구 보기 좋아?", ["울산 HD", "문수"]),
        ("ulsan_18", "울산 현대자동차 공장이 세계에서 제일 크다던데 맞아?", ["수출", "단일공장"]),
        ("ulsan_19", "울산 반구대 암각화가 어떤 역사적 가치가 있어?", ["선사", "고래", "바위그림"]),
        ("ulsan_20", "울산에서 봄에 벚꽃 구경하기 좋은 곳은?", ["무거천", "궁거랑"]),
        ("ulsan_21", "울산 병영 막창 골목 유명해?", ["칼국수", "막창"]),
        ("ulsan_22", "울산 학성공원은 임진왜란 때 왜성이 있던 곳 맞아?", ["왜성", "도산성"]),
        ("ulsan_23", "울산 옥동에 울산대공원 크기가 엄청 크다며?", ["대공원", "자연"]),
        ("ulsan_24", "울산 신불산 폭포 자연휴양림 좋아?", ["휴양림", "계곡"]),
        ("ulsan_25", "울산 시민들이 자주 쓰는 사투리 표현 몇 개만 알려줘", ["와이라노", "머꼬", "단디"]),
    ]
    for pid, p, kw in ulsan_items:
        dataset.append({"id": pid, "category": "울산 local QA", "prompt": p, "ground_truth_keywords": kw})

    # 13. multi-turn (25)
    multi_items = [
        ("multi_01", [("나 이번 주말에 울산 놀러가려고 해", "오 진짜가? 울산 오면 갈 데 많데이! 며칠 일정으로 오는데?")], "1박 2일로 갈 건데 바다 보고 싶어"),
        ("multi_02", [("파이썬 공부 막 시작했어", "잘 생각했데이! 파이썬이 문법 깔끔해서 처음 배우기 딱 좋제. 어디까지 봤노?")], "변수랑 반복문 배웠는데 다음엔 뭐 공부해야 돼?"),
        ("multi_03", [("오늘 감기 걸린 것 같아", "아이고, 요즘 환절기라 감기 환자 많데이. 열은 안 나나?")], "열은 없고 목만 좀 칼칼해"),
        ("multi_04", [("점심에 라면 먹을까 김치찌개 먹을까?", "얼큰한 김치찌개가 든든하지 않겠나!")], "좋아, 김치찌개 먹으러 간다"),
        ("multi_05", [("내일 친구 생일이야", "축하해줄 일이네! 선물은 미리 준비했나?")], "아직 못 샀는데 케이크만 사갈까?"),
        ("multi_06", [("너 혹시 노래 부르는 거 좋아해?", "노래 부르는 건 좋아하지만 목소리가 없어서 아쉽제!")], "나중에 노래도 한번 불러줘라"),
        ("multi_07", [("나 다이어트 시작했다", "오! 결심 대단하네. 운동으로 빼나, 식단으로 빼나?")], "저녁에 야식 안 먹기부터 해보려고"),
        ("multi_08", [("오늘 서울 출장 왔어", "서울 사람 복잡할 텐데 고생 많데이! 길은 잘 찾았나?")], "지하철 환승이 너무 복잡하더라"),
        ("multi_09", [("나 새 노트북 사려고 해", "용도가 뭔데? 게임용이가, 사무용이가?")], "주로 코딩이랑 영상 편집용으로 쓸 거야"),
        ("multi_10", [("오늘 날씨 너무 덥다", "진짜 오늘 햇빛 쨍쨍하데이. 물 자주 마셔라!")], "시원한 아이스 아메리카노 한잔 마셔야겠다"),
        ("multi_11", [("운동 끝나고 단백질 보충제 먹었어", "단디 챙겨 묵었네! 오늘 무슨 운동 했노?")], "하체 스쿼트랑 레그프레스 조졌어"),
        ("multi_12", [("강아지 한 마리 입양하고 싶어", "생명 책임지는 거라 고민 많이 해야 한데이. 어떤 견종 생각하는데?")], "소형견 포메라니안이나 몰티즈 생각 중이야"),
        ("multi_13", [("오늘 이사해서 집 정리 중이야", "이사하느라 몸살 나겠다. 짐은 다 풀었나?")], "아직 옷 정리가 산더미처럼 남았어"),
        ("multi_14", [("면접 정장 사러 백화점 가는 중", "단정하게 잘 골라봐라! 긴장하지 말고 편하게 봐라.")], "네이비가 나을까 블랙이 나을까?"),
        ("multi_15", [("내일 치과 예약해뒀어", "치과는 가기 전이 젤 무섭제. 스케일링이가 충치 치료가?")], "사랑니 뽑으러 가는데 너무 무서워"),
        ("multi_16", [("주말에 캠핑 장비 챙기는 중이야", "캠핑 날씨 딱 좋제! 어디로 가는데?")], "산속 계곡 근처 오토캠핑장 예약했어"),
        ("multi_17", [("어제 밤새워서 넷플릭스 봤어", "재밌는 거 봤나 보네! 무슨 드라마 봤는데?")], "범죄 스릴러물인데 반전이 대박이었어"),
        ("multi_18", [("카페에서 공부 중인데 집중이 안 돼", "노래 듣거나 자리 한번 바꿔봐라. 무슨 공부하는데?")], "토익 영어 단어 외우는 중이야"),
        ("multi_19", [("자전거 타고 강변 달리는 중이야", "바람 쐬면 기분 상쾌하제! 안전 장비는 찼나?")], "헬멧 단디 쓰고 라이딩 중이지"),
        ("multi_20", [("요리 연습 중인데 간 맞추기가 어려워", "요리는 간이 절반이제! 오늘 무슨 반찬 하는데?")], "제육볶음 만드는데 너무 짠 것 같아"),
        ("multi_21", [("오늘 월급날이다!", "크으 고생한 보람 있네! 맛있는 거 사 묵어야제!")], "오랜만에 소고기 한번 썰러 가려고"),
        ("multi_22", [("휴대전화 액정이 깨졌어", "아이고 마음 찢어지겠다. 터치는 잘 되나?")], "화면은 나오는데 유리 조각이 떨어져"),
        ("multi_23", [("운전면허 도로주행 시험 본다", "신호 잘 보고 차선 변경할 때 깜빡이 단디 켜라!")], "출발할 때 사이드브레이크 내리는 거 까먹을 뻔했어"),
        ("multi_24", [("해외여행 계획 세우는 중이야", "어디로 가려고? 일본? 동남아?")], "베트남 다낭 3박 4일로 가보려고"),
        ("multi_25", [("오늘부터 독서 모임 시작했어", "좋은 모임이네! 첫 모임 책은 뭐였노?")], "돈의 심리학이라는 경제 서적이야"),
    ]
    for pid, turns, p in multi_items:
        history = []
        for u, a in turns:
            history.append({"role": "user", "content": u})
            history.append({"role": "assistant", "content": a})
        dataset.append({"id": pid, "category": "multi-turn", "prompt": p, "history": history})

    # 14. long-form explanation (25)
    long_items = [
        ("long_01", "인공지능 딥러닝에서 역전파(Backpropagation) 알고리즘의 원리를 단계별로 설명해줘"),
        ("long_02", "컴퓨터 중앙처리장치(CPU)가 명령어를 실행하는 폰 노이만 구조와 과정을 설명해줘"),
        ("long_03", "백신(Vaccine)이 우리 몸의 면역 체계를 어떻게 훈련시키는지 원리를 설명해줘"),
        ("long_04", "양자컴퓨터의 큐비트(Qubit)와 중첩, 얽힘 개념을 쉽게 풀어서 설명해줘"),
        ("long_05", "금리가 인상되었을 때 물가, 환율, 주식시장에 미치는 연쇄적 영향을 설명해줘"),
        ("long_06", "블록체인의 분산원장과 작업증명(PoW) 메커니즘을 설명해줘"),
        ("long_07", "지구 온난화가 해수면 상승과 극단적 기상이변을 초래하는 과학적 메커니즘을 설명해줘"),
        ("long_08", "인터넷 웹 브라우저 주소창에 URL을 치고 엔터를 눌렀을 때 화면이 뜨기까지의 전 과정을 설명해줘"),
        ("long_09", "냉장고가 내부를 차갑게 만드는 냉매의 압축, 응축, 팽창, 증발 사이클을 설명해줘"),
        ("long_10", "자율주행 자동차가 센서 융합과 SLAM 기술을 활용해 길을 찾는 원리를 설명해줘"),
        ("long_11", "인간의 뇌에서 시냅스와 신경전달물질이 신호를 주고받는 과정을 설명해줘"),
        ("long_12", "원자력 발전소에서 핵분열 에너지를 이용해 전기를 생산하는 원리를 설명해줘"),
        ("long_13", "대기압과 고도에 따라 기압이 변하는 이유와 날씨 전선 형성 과정을 설명해줘"),
        ("long_14", "인플레이션(Inflation)과 디플레이션(Deflation)의 차이점과 중앙은행의 통화정책 수단을 설명해줘"),
        ("long_15", "생성형 AI 모델(트랜스포머 아키텍처)의 셀프 어텐션(Self-Attention) 메커니즘을 설명해줘"),
        ("long_16", "눈(Eye)을 통해 들어온 빛이 망막과 시신경을 거쳐 뇌에서 이미지로 인식되는 과정을 설명해줘"),
        ("long_17", "스마트폰 터치스크린(정전용량 방식)이 손가락 터치를 감지하는 원리를 설명해줘"),
        ("long_18", "지구 자기장이 태양풍 우주 방사선으로부터 지구 생명체를 보호하는 원리를 설명해줘"),
        ("long_19", "암(Cancer) 세포가 정상 세포와 다르게 무한 증식하는 원인과 면역치료제의 작용 원리를 설명해줘"),
        ("long_20", "비트코인의 반감기(Halving) 개념과 그것이 시장 공급량에 미치는 영향을 설명해줘"),
        ("long_21", "비누가 기름때를 씻어내는 계면활성제의 친수성/친유성 분자 구조 원리를 설명해줘"),
        ("long_22", "GPS 위성이 상대성이론 시간 지연을 보정하여 지구 위치를 오차 없이 측정하는 원리를 설명해줘"),
        ("long_23", "해류(바닷물의 순환)가 지구 기후를 조절하는 열염순환 컨베이어벨트 시스템을 설명해줘"),
        ("long_24", "데이터베이스 트랜잭션의 ACID 속성이 데이터 무결성을 보장하는 방식을 설명해줘"),
        ("long_25", "음성 인식 및 합성(TTS) 시스템이 음파를 텍스트로 바꾸고 다시 음성으로 생성하는 파이프라인을 설명해줘"),
    ]
    for pid, p in long_items:
        dataset.append({"id": pid, "category": "long-form explanation", "prompt": p})

    return dataset


def classify_response(prompt_item: dict, response: str) -> list[str]:
    failures = []
    cat = prompt_item["category"]
    prompt = prompt_item["prompt"]

    # 1. Echo check
    clean_p = re.sub(r"[^\w\s]", "", prompt).strip()
    clean_r = re.sub(r"[^\w\s]", "", response).strip()
    if len(clean_p) >= 4:
        if clean_r == clean_p:
            failures.append("ECHO")
        elif clean_p in clean_r and len(clean_r) < len(clean_p) * 1.25:
            failures.append("ECHO")

    # 2. Multilingual leakage
    cjk_matches = CJK_REGEX.findall(response)
    if cjk_matches:
        failures.append(f"LANGUAGE_LEAKAGE({len(cjk_matches)} tokens)")

    # 3. Repetition loop
    if REPETITION_REGEX.search(response):
        failures.append("REPETITION")

    # 4. Factual error / keywords check for factual, science, math, coding
    gt_keywords = prompt_item.get("ground_truth_keywords", [])
    if gt_keywords:
        has_match = any(kw.lower() in response.lower() for kw in gt_keywords)
        if not has_match:
            failures.append("FACTUAL_ERROR")

    # 5. Recommendation check
    if cat == "recommendation":
        bizarre_terms = ["당근마켓", "중고나라", "노예", "소월드", "비행기표 사서", "아무것도 안 먹", "undefined"]
        if any(bz in response for bz in bizarre_terms):
            failures.append("BAD_RECOMMENDATION")
        if len(response.strip()) < 15:
            failures.append("IRRELEVANT")

    # 6. Empathy check
    if cat == "empathy":
        cold_or_bad = ["축하해", "잘됐다", "퇴학하라", "죽어라", "어쩌라고", "바보"]
        if any(c in response for c in cold_or_bad):
            failures.append("BAD_EMPATHY")

    # 7. Dialect overuse or underuse
    # Count dialect markers
    marker_count = sum(1 for dm in DIALECT_MARKERS if dm in response)
    words = response.split()
    if len(words) >= 5 and marker_count == 0:
        failures.append("DIALECT_UNDERUSE")
    elif len(words) >= 10 and marker_count > len(words) * 0.4:
        failures.append("DIALECT_OVERUSE")

    # 8. Hallucination heuristic
    hallucination_indicators = ["비열한 물", "땅살이 위로 아래로 빠르게 흐르지 않기", "날개가 달린 거 아니야", "마라카이"]
    if any(hi in response for hi in hallucination_indicators):
        failures.append("HALLUCINATION")

    # 9. Irrelevant / empty check
    if len(response.strip()) < 2:
        failures.append("IRRELEVANT")

    return failures


def main():
    parser = argparse.ArgumentParser(description="Evaluate 350 prompts on Phase3 Best model")
    parser.add_argument("--model-path", default="outputs/ulm-1.7b-phase3-best-merged")
    parser.add_argument("--output-json", default="reports/phase4_failure_collection_350.json")
    parser.add_argument("--output-issues-jsonl", default="reports/phase4_failure_issues.jsonl")
    args = parser.parse_args()

    print(f"=== [Step 3] Running 350-Prompt Failure Collection on {args.model_path} ===")
    prompts = load_prompts_dataset()
    print(f"Loaded {len(prompts)} prompts across 14 categories.")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
    )
    model.eval()

    results = []
    failure_records = []
    category_summary = {}
    failure_type_counts = {}

    for idx, item in enumerate(prompts, start=1):
        pid = item["id"]
        cat = item["category"]
        prompt = item["prompt"]
        history = item.get("history", [])

        category_summary.setdefault(cat, {"total": 0, "pass": 0, "fail": 0})
        category_summary[cat]["total"] += 1

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
                max_new_tokens=180,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.eos_token_id,
            )
        gen_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        response = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()

        failures = classify_response(item, response)
        is_pass = len(failures) == 0

        if is_pass:
            category_summary[cat]["pass"] += 1
        else:
            category_summary[cat]["fail"] += 1
            for f in failures:
                ftype = f.split("(")[0]
                failure_type_counts[ftype] = failure_type_counts.get(ftype, 0) + 1
            failure_record = {
                "id": pid,
                "category": cat,
                "prompt": prompt,
                "response": response,
                "failures": failures,
            }
            failure_records.append(failure_record)

        results.append({
            "id": pid,
            "category": cat,
            "prompt": prompt,
            "response": response,
            "failures": failures,
            "pass": is_pass,
        })

        status_str = "PASS" if is_pass else f"FAIL({','.join(failures)})"
        if idx % 10 == 0 or not is_pass:
            print(f"[{idx:03d}/350] [{cat}] {prompt[:25]}... -> {status_str}")

    total_pass = sum(c["pass"] for c in category_summary.values())
    pass_rate = (total_pass / len(prompts)) * 100

    report = {
        "model_path": args.model_path,
        "total_prompts": len(prompts),
        "total_pass": total_pass,
        "total_fail": len(failure_records),
        "pass_rate_pct": round(pass_rate, 2),
        "failure_type_counts": failure_type_counts,
        "category_summary": category_summary,
        "results": results,
    }

    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    with open(args.output_issues_jsonl, "w", encoding="utf-8") as f:
        for fr in failure_records:
            f.write(json.dumps(fr, ensure_ascii=False) + "\n")

    print(f"\n==========================================")
    print(f"350-PROMPT EVALUATION COMPLETE:")
    print(f"Total:      {len(prompts)}")
    print(f"Pass:       {total_pass} ({pass_rate:.1f}%)")
    print(f"Fail:       {len(failure_records)} ({100 - pass_rate:.1f}%)")
    print(f"Failure Breakdown:")
    for ft, count in sorted(failure_type_counts.items(), key=lambda x: -x[1]):
        print(f"  - {ft}: {count}")
    print(f"Report:     {args.output_json}")
    print(f"Issues:     {args.output_issues_jsonl}")
    print(f"==========================================")


if __name__ == "__main__":
    main()
