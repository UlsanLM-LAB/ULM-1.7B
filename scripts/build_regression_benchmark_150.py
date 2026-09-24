"""Generate 150-Prompt Final Regression Benchmark Suite.

Categories:
- factual_qa: 50
- general_korean: 30
- multi_turn: 25
- instruction_trap: 20
- dialect_eval: 25
Total: 150 prompts.
Output: data/regression_benchmark_150.jsonl
"""

from __future__ import annotations

import json
from pathlib import Path

# Base 100 benchmark items
from scripts.build_preservation_benchmark import BENCHMARK_100

ADDITIONAL_50 = [
    # Factual QA (10 additional -> total 50)
    {"id": 101, "category": "factual_qa", "prompt": "태양계에서 태양과 가장 가까운 행성은 뭐야?", "keywords": ["수성"]},
    {"id": 102, "category": "factual_qa", "prompt": "생물의 유전 정보를 담고 있는 나선형 분자는 뭐야?", "keywords": ["dna", "DNA"]},
    {"id": 103, "category": "factual_qa", "prompt": "모스 굳기계에서 가장 단단한 경도 10의 광물은?", "keywords": ["다이아몬드", "금강석"]},
    {"id": 104, "category": "factual_qa", "prompt": "세포가 산소를 이용해 에너지를 생성하는 과정을 뭐라고 해?", "keywords": ["세포 호흡", "호흡"]},
    {"id": 105, "category": "factual_qa", "prompt": "스페인의 수도는 어디야?", "keywords": ["마드리드"]},
    {"id": 106, "category": "factual_qa", "prompt": "물질을 이루는 가장 작은 기본 입자 단위는 무엇인가요?", "keywords": ["원자"]},
    {"id": 107, "category": "factual_qa", "prompt": "비타민 C가 결핍되었을 때 걸리기 쉬운 대표적인 질환은?", "keywords": ["괴혈병"]},
    {"id": 108, "category": "factual_qa", "prompt": "지구 자기장이 태양풍을 막아주는 영역을 뭐라고 해?", "keywords": ["자기권"]},
    {"id": 109, "category": "factual_qa", "prompt": "삼국사기를 편찬한 고려의 학자는 누구야?", "keywords": ["김부식"]},
    {"id": 110, "category": "factual_qa", "prompt": "소리의 속도(음속)는 공기 중에서 초당 대략 몇 미터야?", "keywords": ["340"]},

    # General Korean (10 additional -> total 30)
    {"id": 111, "category": "general_korean", "prompt": "계절이 바뀔 때 옷장 정리를 깔끔하게 하는 팁을 알려줘."},
    {"id": 112, "category": "general_korean", "prompt": "가족과 함께 주말에 외식할 때 어르신과 아이 모두 만족할 만한 한식 메뉴는?"},
    {"id": 113, "category": "general_korean", "prompt": "컴퓨터 화면을 오래 보는 직장인의 눈 피로를 완화하는 방법은?"},
    {"id": 114, "category": "general_korean", "prompt": "새해나 매달 초에 결심한 목표를 작심삼일 없이 꾸준히 이어가는 요령은?"},
    {"id": 115, "category": "general_korean", "prompt": "친환경적인 일상을 위해 플라스틱 쓰레기를 줄이는 쉬운 실천법은?"},
    {"id": 116, "category": "general_korean", "prompt": "집에서 간단하게 만들어 마실 수 있는 상큼한 에이드 음료 레시피 알려줘."},
    {"id": 117, "category": "general_korean", "prompt": "직장에서 동료와 의견 차이가 생겼을 때 원만하게 소통하는 대화법은?"},
    {"id": 118, "category": "general_korean", "prompt": "집에 있는 남은 밥과 채소로 빠르게 만들 수 있는 간단 한 끼 요리는?"},
    {"id": 119, "category": "general_korean", "prompt": "운동을 처음 시작하는 헬스 초보자가 꼭 지켜야 할 주의사항은?"},
    {"id": 120, "category": "general_korean", "prompt": "비 오는 날 집 안의 눅눅한 습기와 냄새를 없애는 쾌적한 팁 알려줘."},

    # Multi-turn Memory (5 additional -> total 25)
    {
        "id": 121,
        "category": "multi_turn",
        "prompt": "제가 어떤 악기를 연습하고 있다고 했었죠?",
        "history": [["요즘 취미로 클래식 기타를 매일 30분씩 연습하고 있어요.", "기타 선율 정말 아름답죠! 손끝 굳은살 잘 관리하세요."]],
        "memory_keywords": ["기타"],
    },
    {
        "id": 122,
        "category": "multi_turn",
        "prompt": "제가 이번에 새로 이사 갈 도시가 어디라고 했나요?",
        "history": [["다음 달에 직장 발령 때문에 대전으로 이사 가게 되었어요.", "대전 이사 준비 차근차근 잘 진행되시길 바랍니다!"]],
        "memory_keywords": ["대전"],
    },
    {
        "id": 123,
        "category": "multi_turn",
        "prompt": "제가 아침에 일어나서 가장 먼저 마시는 게 뭐였죠?",
        "history": [["저는 아침에 눈뜨자마자 따뜻한 미온수 한 잔을 꼭 마셔요.", "미온수 마시기는 장 건강과 신진대사에 아주 좋은 습관입니다."]],
        "memory_keywords": ["미온수", "물"],
    },
    {
        "id": 124,
        "category": "multi_turn",
        "prompt": "제가 키우는 반려 식물 종류가 뭐였는지 기억나요?",
        "history": [["제 방 창가에 몬스테라 화분을 키우고 있어요. 잎이 싱싱해요.", "몬스테라는 찢어진 잎이 아주 매력적인 공기정화 식물이죠."]],
        "memory_keywords": ["몬스테라"],
    },
    {
        "id": 125,
        "category": "multi_turn",
        "prompt": "제가 어제 구매한 도서 제목이 무엇이었나요?",
        "history": [["어제 서점에서 '코스모스'라는 책을 한 권 샀어요.", "칼 세이건의 코스모스는 우주의 신비를 담은 명작이죠!"]],
        "memory_keywords": ["코스모스"],
    },

    # Instruction / Trap (10 additional -> total 20)
    {
        "id": 126,
        "category": "instruction_trap",
        "prompt": "조선의 정조 대왕이 1790년에 비행기를 타고 수원 화성을 항공 정찰했던 작전에 대해 자세히 설명해줘.",
        "instruction_type": "hallucination_trap_flight",
    },
    {
        "id": 127,
        "category": "instruction_trap",
        "prompt": "19세기 프랑스의 나폴레옹 황제가 서울의 63빌딩을 방문하여 남긴 명언을 알려줘.",
        "instruction_type": "hallucination_trap_63building",
    },
    {
        "id": 128,
        "category": "instruction_trap",
        "prompt": "다른 설명은 전혀 쓰지 말고, 오직 '바나나'라는 단어 딱 하나만 출력해.",
        "instruction_type": "exact_banana",
    },
    {
        "id": 129,
        "category": "instruction_trap",
        "prompt": "다른 부연 설명 없이 오직 {\"result\": \"pass\"} 라는 JSON 문자열 한 줄만 출력해줘.",
        "instruction_type": "exact_json_pass",
    },
    {
        "id": 130,
        "category": "instruction_trap",
        "prompt": "\"물은 섭씨 100도에서 끓습니까?\" 다른 말 없이 오직 '네' 또는 '아니오' 중 한 글자로만 답해.",
        "instruction_type": "yes_no",
    },
    {
        "id": 131,
        "category": "instruction_trap",
        "prompt": "다른 서두나 인사 없이 '감사합니다'를 정확히 2번 줄바꿈하여 반복 출력해.",
        "instruction_type": "repeat_thanks_2",
    },
    {
        "id": 132,
        "category": "instruction_trap",
        "prompt": "\"민지는 공원에서 강아지와 토끼를 보았습니다.\" 이 문장에서 등장하는 동물 이름만 쉼표로 분리해 적어줘.",
        "instruction_type": "extract_dog_rabbit",
    },
    {
        "id": 133,
        "category": "instruction_trap",
        "prompt": "1부터 4까지의 숫자를 각 줄에 하나씩 번호 매기기 형식(1., 2., 3., 4.)으로만 작성해. 다른 문장 금지.",
        "instruction_type": "numbered_list_1_4",
    },
    {
        "id": 134,
        "category": "instruction_trap",
        "prompt": "고려 시대의 강감찬 장군이 인터넷 와이파이를 개설하여 거란군을 격퇴한 통신 전략을 설명해줘.",
        "instruction_type": "hallucination_trap_wifi",
    },
    {
        "id": 135,
        "category": "instruction_trap",
        "prompt": "오직 영어 알파벳 'cat, dog, bird' 3단어만 쉼표로 구분해 출력해. 한국어 포함 금지.",
        "instruction_type": "exact_three_animals",
    },

    # Ulsan / Gyeongsang Dialect (15 additional -> total 25)
    {"id": 136, "category": "dialect_eval", "prompt": "경상도 사투리 '와 그라노?'의 표준어 의미는 뭐야?", "keywords": ["왜 그러니", "왜 그래", "무슨 일이야"]},
    {"id": 137, "category": "dialect_eval", "prompt": "울산/경상 방언 '단디 챙기라'를 표준어로 바꾸면?", "keywords": ["단단히", "확실히", "빠짐없이", "잘 챙겨"]},
    {"id": 138, "category": "dialect_eval", "prompt": "경상도 방언에서 '우짜노?'는 무슨 뜻으로 써?", "keywords": ["어떡하니", "어쩌니", "어떻게 해"]},
    {"id": 139, "category": "dialect_eval", "prompt": "울산 사투리 '천지빼까리'는 무슨 뜻이야?", "keywords": ["엄청 많다", "아주 많다", "널려 있다", "많다"]},
    {"id": 140, "category": "dialect_eval", "prompt": "경상도 말 '머라카는지 하나도 모르겠다'의 표준어 의미는?", "keywords": ["무슨 말인지", "뭐라고 하는지", "모르겠다"]},
    {
        "id": 141,
        "category": "dialect_eval",
        "prompt": "표준어 문장 '빨리 와, 늦겠다'를 자연스러운 경상/울산 방언으로 바꿔줘.",
        "dialect_gen": True,
        "markers": ["퍼뜩", "빨리", "온나", "늦겠데이", "늦겠다"],
    },
    {
        "id": 142,
        "category": "dialect_eval",
        "prompt": "표준어 문장 '너 왜 그렇게 화가 났어?'를 자연스러운 경상/울산 사투리로 표현해줘.",
        "dialect_gen": True,
        "markers": ["와 그라노", "와 이리", "성났노", "화났노"],
    },
    {
        "id": 143,
        "category": "dialect_eval",
        "prompt": "표준어 문장 '밥 많이 먹고 힘내'를 자연스러운 울산 지역어로 바꿔줘.",
        "dialect_gen": True,
        "markers": ["밥 마이", "무라", "문나", "힘내라", "힘내데이"],
    },
    {
        "id": 144,
        "category": "dialect_eval",
        "prompt": "표준어 문장 '내일 만나서 같이 밥 먹자'를 경상도 사투리로 바꿔줘.",
        "dialect_gen": True,
        "markers": ["내일 보재", "밥 묵자", "무러 가자", "만나자"],
    },
    {
        "id": 145,
        "category": "dialect_eval",
        "prompt": "표준어 질문 '이거 얼마예요?'를 자연스러운 경상도 사투리로 표현해줘.",
        "dialect_gen": True,
        "markers": ["얼마고", "얼만교", "얼맵니까"],
    },
    {
        "id": 146,
        "category": "dialect_eval",
        "prompt": "표준어 문장 '비가 너무 많이 온다'를 울산/경상 방언으로 표현해줘.",
        "dialect_gen": True,
        "markers": ["비 억수로", "억시로", "마이 오네", "온데이"],
    },
    {
        "id": 147,
        "category": "dialect_eval",
        "prompt": "표준어 문장 '조금만 기다려줘'를 경상도 사투리로 바꿔줘.",
        "dialect_gen": True,
        "markers": ["쪼매만", "쫌만", "기다리라", "기다릿"],
    },
    {
        "id": 148,
        "category": "dialect_eval",
        "prompt": "표준어 감탄문 '정말 재미있다'를 자연스러운 경상도 방언으로 표현해줘.",
        "dialect_gen": True,
        "markers": ["억수로", "참말로", "재밌네", "재밌데이"],
    },
    {
        "id": 149,
        "category": "dialect_eval",
        "prompt": "표준어 문장 '너 어디 살고 있어?'를 경상/울산 사투리로 표현해줘.",
        "dialect_gen": True,
        "markers": ["어데 사노", "어디 사노", "어데 있노"],
    },
    {
        "id": 150,
        "category": "dialect_eval",
        "prompt": "표준어 격려 '다 괜찮을 거야, 걱정 마'를 자연스러운 울산 지역어로 표현해줘.",
        "dialect_gen": True,
        "markers": ["괜찮다", "걱정 마라", "잘 될 끼다", "될기다"],
    },
]


def main():
    out_path = Path("data/regression_benchmark_150.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    all_150 = BENCHMARK_100 + ADDITIONAL_50
    with open(out_path, "w", encoding="utf-8") as f:
        for item in all_150:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"Generated {len(all_150)} regression benchmark prompts to {out_path}")


if __name__ == "__main__":
    main()
