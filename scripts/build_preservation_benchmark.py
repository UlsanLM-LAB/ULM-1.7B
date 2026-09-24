"""Preservation Holdout Benchmark Suite (100 Prompts).

Used to monitor and gate catastrophic forgetting during ULM-4B Phase 3 v2 training.
Categories:
- factual_qa: 40
- general_korean: 20
- multi_turn: 20
- instruction_trap: 10
- dialect_eval: 10
Total: 100 prompts.
"""

from __future__ import annotations

import json
from pathlib import Path

BENCHMARK_100 = [
    # ==========================================
    # 1. Factual QA (40)
    # ==========================================
    {"id": 1, "category": "factual_qa", "prompt": "대한민국의 수도는 어디야?", "keywords": ["서울"]},
    {"id": 2, "category": "factual_qa", "prompt": "태양계에서 가장 큰 행성은 뭐야?", "keywords": ["목성"]},
    {"id": 3, "category": "factual_qa", "prompt": "한글을 창제한 조선의 국왕은 누구야?", "keywords": ["세종"]},
    {"id": 4, "category": "factual_qa", "prompt": "인류가 최초로 달에 착륙한 연도는 몇 년도야?", "keywords": ["1969"]},
    {"id": 5, "category": "factual_qa", "prompt": "원소 기호 O는 무슨 원소야?", "keywords": ["산소"]},
    {"id": 6, "category": "factual_qa", "prompt": "세계에서 가장 면적이 넓은 국가는 어디야?", "keywords": ["러시아"]},
    {"id": 7, "category": "factual_qa", "prompt": "지구에서 가장 높은 산은 뭐야?", "keywords": ["에베레스트"]},
    {"id": 8, "category": "factual_qa", "prompt": "프랑스의 수도는 어디야?", "keywords": ["파리"]},
    {"id": 9, "category": "factual_qa", "prompt": "물 분자의 화학식은 뭐야?", "keywords": ["h2o", "h₂o"]},
    {"id": 10, "category": "factual_qa", "prompt": "빛의 진공 속 속도는 초당 약 몇 km야?", "keywords": ["30만", "299,792", "300,000"]},
    {"id": 11, "category": "factual_qa", "prompt": "임진왜란이 일어난 연도는 몇 년도야?", "keywords": ["1592"]},
    {"id": 12, "category": "factual_qa", "prompt": "이순신 장군이 13척으로 왜선 133척을 격파한 해전은?", "keywords": ["명량"]},
    {"id": 13, "category": "factual_qa", "prompt": "미국의 초대 대통령은 누구야?", "keywords": ["워싱턴"]},
    {"id": 14, "category": "factual_qa", "prompt": "정상 성인의 체온은 대략 몇 도야?", "keywords": ["36.5", "36도", "37도"]},
    {"id": 15, "category": "factual_qa", "prompt": "삼국통일을 완성한 신라의 왕은 누구야?", "keywords": ["문무왕"]},
    {"id": 16, "category": "factual_qa", "prompt": "지구에서 가장 면적이 넓은 바다(대양)는 어디야?", "keywords": ["태평양"]},
    {"id": 17, "category": "factual_qa", "prompt": "식물이 빛을 이용해 양분을 스스로 만드는 작용을 뭐라고 해?", "keywords": ["광합성"]},
    {"id": 18, "category": "factual_qa", "prompt": "일본의 수도는 어디야?", "keywords": ["도쿄", "동경"]},
    {"id": 19, "category": "factual_qa", "prompt": "원주율(파이, π)의 값은 소수점 둘째 자리까지 대략 얼마야?", "keywords": ["3.14"]},
    {"id": 20, "category": "factual_qa", "prompt": "고려를 멸망시키고 조선을 건국한 인물은 누구야?", "keywords": ["이성계", "태조"]},
    {"id": 21, "category": "factual_qa", "prompt": "전하(전기)를 띤 입자의 흐름을 무엇이라고 해?", "keywords": ["전류"]},
    {"id": 22, "category": "factual_qa", "prompt": "인체에서 혈액을 온몸으로 뿜어내는 중심 펌프 장기는 뭐야?", "keywords": ["심장"]},
    {"id": 23, "category": "factual_qa", "prompt": "영국의 수도는 어디야?", "keywords": ["런던"]},
    {"id": 24, "category": "factual_qa", "prompt": "호주(오스트레일리아)의 공식 수도는 어디야?", "keywords": ["캔버라"]},
    {"id": 25, "category": "factual_qa", "prompt": "독일의 수도는 어디야?", "keywords": ["베를린"]},
    {"id": 26, "category": "factual_qa", "prompt": "이탈리아의 수도는 어디야?", "keywords": ["로마"]},
    {"id": 27, "category": "factual_qa", "prompt": "캐나다의 수도는 어디야?", "keywords": ["오타와"]},
    {"id": 28, "category": "factual_qa", "prompt": "중국의 수도는 어디야?", "keywords": ["베이징", "북경"]},
    {"id": 29, "category": "factual_qa", "prompt": "세계에서 가장 긴 강은 어디야?", "keywords": ["나일", "아마존"]},
    {"id": 30, "category": "factual_qa", "prompt": "주기율표에서 가장 가벼운 첫 번째 원소는 뭐야?", "keywords": ["수소"]},
    {"id": 31, "category": "factual_qa", "prompt": "원소 기호 Fe는 어떤 금속 원소야?", "keywords": ["철"]},
    {"id": 32, "category": "factual_qa", "prompt": "원소 기호 Au는 어떤 귀금속 원소야?", "keywords": ["금"]},
    {"id": 33, "category": "factual_qa", "prompt": "다이아몬드는 어떤 단일 원소로만 이루어져 있어?", "keywords": ["탄소"]},
    {"id": 34, "category": "factual_qa", "prompt": "만유인력의 법칙을 정리한 영국의 물리학자는 누구야?", "keywords": ["뉴턴"]},
    {"id": 35, "category": "factual_qa", "prompt": "특수 및 일반 상대성 이론을 발표한 이론물리학자는 누구야?", "keywords": ["아인슈타인"]},
    {"id": 36, "category": "factual_qa", "prompt": "훈민정음이 공식적으로 반포된 연도는 몇 년도야?", "keywords": ["1446"]},
    {"id": 37, "category": "factual_qa", "prompt": "후삼국을 통일하고 고려를 건국한 왕은 누구야?", "keywords": ["왕건", "태조"]},
    {"id": 38, "category": "factual_qa", "prompt": "백제의 마지막 수도였던 사비는 현재 어느 지역이야?", "keywords": ["부여"]},
    {"id": 39, "category": "factual_qa", "prompt": "지구가 태양을 한 바퀴 공전하는 데 걸리는 시간은 대략 며칠이야?", "keywords": ["365"]},
    {"id": 40, "category": "factual_qa", "prompt": "지구 주위를 공전하는 유일한 자연 위성은 뭐야?", "keywords": ["달"]},

    # ==========================================
    # 2. General Korean Conversation / Knowledge (20)
    # ==========================================
    {"id": 41, "category": "general_korean", "prompt": "오늘 날씨가 참 화창하고 맑네. 산책하기 좋은 날씨야."},
    {"id": 42, "category": "general_korean", "prompt": "오늘 저녁 메뉴 고민 중인데, 비 오는 날 먹기 좋은 따뜻한 국물 음식 추천해줘."},
    {"id": 43, "category": "general_korean", "prompt": "퇴근하고 집에 왔는데 너무 피곤하다. 피로를 푸는 데 뭐가 좋을까?"},
    {"id": 44, "category": "general_korean", "prompt": "친구 생일 선물로 3만 원대 실용적인 아이템 하나 추천해 줄래?"},
    {"id": 45, "category": "general_korean", "prompt": "주말에 집에서 볼 만한 가벼운 힐링 영화 장르 추천해줘."},
    {"id": 46, "category": "general_korean", "prompt": "요즘 회사 업무 스트레스가 많은데 건강하게 해소하는 팁이 있을까?"},
    {"id": 47, "category": "general_korean", "prompt": "아침에 개운하게 일찍 일어나는 습관을 들이려면 어떻게 해야 할까?"},
    {"id": 48, "category": "general_korean", "prompt": "처음으로 커피 머신을 사려고 하는데 입문자에게 어떤 종류가 편할까?"},
    {"id": 49, "category": "general_korean", "prompt": "건강한 일상을 위해 매일 간단하게 실천할 수 있는 좋은 습관 3가지만 알려줘."},
    {"id": 50, "category": "general_korean", "prompt": "국내 1박 2일 여행 가려고 하는데 바다와 산 중 어디가 힐링에 더 좋을까?"},
    {"id": 51, "category": "general_korean", "prompt": "주말에 혼자 집에서 집중해서 할 만한 생산적인 취미 추천해줘."},
    {"id": 52, "category": "general_korean", "prompt": "무더운 여름철에 기력을 보충하고 건강을 지키는 식습관 팁 알려줘."},
    {"id": 53, "category": "general_korean", "prompt": "바쁜 직장인이 꾸준히 한 달에 책 1권씩 읽으려면 어떤 방법이 좋을까?"},
    {"id": 54, "category": "general_korean", "prompt": "실내에서 키우기 쉽고 공기 정화에 좋은 초보자용 반려식물 추천해줘."},
    {"id": 55, "category": "general_korean", "prompt": "감기 기운이 살짝 올 때 초기에 몸을 따뜻하게 하고 회복을 돕는 대처법은?"},
    {"id": 56, "category": "general_korean", "prompt": "밤에 잠이 잘 오지 않고 뒤척일 때 숙면을 돕는 이완 방법이 있을까?"},
    {"id": 57, "category": "general_korean", "prompt": "하루 일정을 효율적으로 관리할 수 있는 간단한 시간 관리 기법 알려줘."},
    {"id": 58, "category": "general_korean", "prompt": "바쁜 아침 10분 만에 챙겨 먹을 수 있는 간편하고 든든한 아침 메뉴는?"},
    {"id": 59, "category": "general_korean", "prompt": "처음으로 독립해서 자취를 시작할 때 가장 먼저 사야 할 필수 살림살이 3가지는?"},
    {"id": 60, "category": "general_korean", "prompt": "마음이 복잡하고 불안할 때 차분하게 들을 수 있는 음악 장르나 스타일 추천해줘."},

    # ==========================================
    # 3. Multi-turn Memory (20)
    # ==========================================
    {
        "id": 61,
        "category": "multi_turn",
        "prompt": "내 이름이 뭐라고 했지?",
        "history": [["내 이름은 민수야. 기억해 줘.", "네, 민수 님! 기억하고 있겠습니다."]],
        "memory_keywords": ["민수"],
    },
    {
        "id": 62,
        "category": "multi_turn",
        "prompt": "내가 어떤 과일을 더 좋아한다고 했지?",
        "history": [["나는 사과보다 바나나를 더 좋아해.", "바나나 달콤하고 영양도 풍부하죠!"]],
        "memory_keywords": ["바나나"],
    },
    {
        "id": 63,
        "category": "multi_turn",
        "prompt": "내가 언제 어디로 출장 간다고 했어?",
        "history": [["다음 주 화요일에 부산으로 출장 가.", "부산 출장 일정 잘 챙기세요!"]],
        "memory_keywords": ["화요일", "부산"],
    },
    {
        "id": 64,
        "category": "multi_turn",
        "prompt": "내 취미가 뭐라고 했는지 기억나?",
        "history": [["내 취미는 주말마다 자전거 타는 거야.", "자전거 타기는 정말 상쾌한 운동이죠."]],
        "memory_keywords": ["자전거"],
    },
    {
        "id": 65,
        "category": "multi_turn",
        "prompt": "우리 강아지 이름이 뭐였지?",
        "history": [["우리 집 강아지 이름은 초코야. 갈색 푸들이야.", "초코라는 이름 정말 사랑스럽네요."]],
        "memory_keywords": ["초코"],
    },
    {
        "id": 66,
        "category": "multi_turn",
        "prompt": "내가 아까 알려준 임시 PIN 코드가 뭐였어?",
        "history": [["보안을 위해 임시 PIN 코드를 7492로 설정할게.", "임시 PIN 코드 7492로 확인했습니다."]],
        "memory_keywords": ["7492"],
    },
    {
        "id": 67,
        "category": "multi_turn",
        "prompt": "내가 커피 마실 때 뭘 마신다고 했어?",
        "history": [["커피 마실 때 나는 무조건 아이스 아메리카노만 마셔.", "얼죽아 스타일이시군요!"]],
        "memory_keywords": ["아이스 아메리카노"],
    },
    {
        "id": 68,
        "category": "multi_turn",
        "prompt": "내일 미팅 장소가 어디라고 했지?",
        "history": [["내일 미팅 장소는 강남역 1번 출구 앞 스타벅스로 정했어.", "강남역 1번 출구 스타벅스 메모했습니다."]],
        "memory_keywords": ["강남역", "스타벅스"],
    },
    {
        "id": 69,
        "category": "multi_turn",
        "prompt": "내 여동생 이름이 뭐라고 했어?",
        "history": [["내 여동생 이름은 지수야. 대학생이야.", "지수 씨 정말 착한 동생일 것 같네요."]],
        "memory_keywords": ["지수"],
    },
    {
        "id": 70,
        "category": "multi_turn",
        "prompt": "내가 사계절 중 어떤 계절을 제일 좋아한다고 했지?",
        "history": [["나는 시원한 바람이 부는 가을을 가장 좋아해.", "가을은 독서와 산책의 계절이죠."]],
        "memory_keywords": ["가을"],
    },
    {
        "id": 71,
        "category": "multi_turn",
        "prompt": "내가 요즘 배우고 있는 운동이 뭐였지?",
        "history": [["요즘 건강을 위해 수영 강습을 등록해서 배우고 있어.", "수영은 전신 유산소 운동으로 최고입니다!"]],
        "memory_keywords": ["수영"],
    },
    {
        "id": 72,
        "category": "multi_turn",
        "prompt": "내 직업이 뭐라고 했는지 기억해?",
        "history": [["내 직업은 고등학교에서 국어를 가르치는 교사야.", "국어 선생님이시군요! 멋지십니다."]],
        "memory_keywords": ["교사"],
    },
    {
        "id": 73,
        "category": "multi_turn",
        "prompt": "내가 가장 좋아하는 색깔이 뭐라고 했지?",
        "history": [["나는 차분한 파란색을 제일 좋아해.", "파란색은 평온하고 신뢰를 주는 색이죠."]],
        "memory_keywords": ["파란색", "파랑"],
    },
    {
        "id": 74,
        "category": "multi_turn",
        "prompt": "내 고향이 어디라고 했는지 기억나?",
        "history": [["내 고향은 바다와 공업이 어우러진 울산이야.", "울산은 태화강과 대왕암이 유명하죠!"]],
        "memory_keywords": ["울산"],
    },
    {
        "id": 75,
        "category": "multi_turn",
        "prompt": "내가 제일 좋아하는 한식 메뉴가 뭐라고 했어?",
        "history": [["내가 제일 좋아하는 한국 음식은 구수한 된장찌개야.", "된장찌개는 언제 먹어도 든든한 소울푸드죠."]],
        "memory_keywords": ["된장찌개"],
    },
    {
        "id": 76,
        "category": "multi_turn",
        "prompt": "내가 이번 여름휴가 때 가고 싶다고 한 여행지는 어디야?",
        "history": [["이번 여름휴가 때는 제주도로 혼자 힐링 여행 갈 계획이야.", "제주도 푸른 바다와 올레길 정말 좋겠네요!"]],
        "memory_keywords": ["제주도"],
    },
    {
        "id": 77,
        "category": "multi_turn",
        "prompt": "내가 지금 살고 있는 동네가 어디라고 했지?",
        "history": [["나는 지금 서울 마포구에 자취하고 있어.", "마포구는 한강공원도 가깝고 살기 좋죠."]],
        "memory_keywords": ["마포구", "서울"],
    },
    {
        "id": 78,
        "category": "multi_turn",
        "prompt": "우리 집 고양이 이름이 뭐였지?",
        "history": [["우리 집 고양이 이름은 나비야. 하얀 털을 가졌어.", "나비라는 이름 너무 귀엽네요!"]],
        "memory_keywords": ["나비"],
    },
    {
        "id": 79,
        "category": "multi_turn",
        "prompt": "내가 올해 취득하려고 준비 중인 자격증이 뭐였어?",
        "history": [["올해 하반기에 정보처리기사 자격증 취득하려고 공부 중이야.", "정보처리기사 시험 꼭 합격하시길 응원합니다!"]],
        "memory_keywords": ["정보처리기사"],
    },
    {
        "id": 80,
        "category": "multi_turn",
        "prompt": "내가 출퇴근할 때 타는 차종이 뭐라고 했지?",
        "history": [["출퇴근할 때 흰색 아반떼 몰고 다녀.", "아반떼는 연비도 좋고 실용적인 명차죠."]],
        "memory_keywords": ["아반떼"],
    },

    # ==========================================
    # 4. Instruction Following / Trap (10)
    # ==========================================
    {
        "id": 81,
        "category": "instruction_trap",
        "prompt": "다른 설명이나 부가적인 말은 전혀 하지 말고, 오직 '사과'라는 단어 딱 하나만 출력해.",
        "instruction_type": "exact_apple",
    },
    {
        "id": 82,
        "category": "instruction_trap",
        "prompt": "1부터 5까지의 숫자를 각 줄에 하나씩 번호 매기기 형식(1., 2., 3., 4., 5.)으로만 작성해줘. 다른 서두나 결론은 적지 마.",
        "instruction_type": "numbered_list_1_5",
    },
    {
        "id": 83,
        "category": "instruction_trap",
        "prompt": "다른 인사나 설명 없이 '안녕하세요'라는 문장만 정확히 3번 줄바꿈하여 반복해서 출력해줘.",
        "instruction_type": "repeat_hello_3",
    },
    {
        "id": 84,
        "category": "instruction_trap",
        "prompt": "\"철수는 어제 길에서 고양이와 참새를 보았다.\" 이 문장에서 등장하는 동물 이름만 쉼표로 구분해서 출력해.",
        "instruction_type": "extract_animals",
    },
    {
        "id": 85,
        "category": "instruction_trap",
        "prompt": "\"지구는 둥근가요?\" 이 질문에 다른 말은 절대 하지 말고 오직 '네' 또는 '아니오' 중 한 글자로만 답해줘.",
        "instruction_type": "yes_no",
    },
    {
        "id": 86,
        "category": "instruction_trap",
        "prompt": "화성에 세워진 고대 이집트 양식의 피라미드는 누가 건설했는지 자세히 설명해줘.",
        "instruction_type": "hallucination_trap_mars",
    },
    {
        "id": 87,
        "category": "instruction_trap",
        "prompt": "조선의 세종대왕이 1443년에 직접 아이폰을 개발하여 백성들에게 보급한 역사적 배경을 설명해줘.",
        "instruction_type": "hallucination_trap_iphone",
    },
    {
        "id": 88,
        "category": "instruction_trap",
        "prompt": "2020년 대한민국 제16대 대통령으로 선출된 세종대왕의 주요 경제 정책에 대해 서술해줘.",
        "instruction_type": "hallucination_trap_president",
    },
    {
        "id": 89,
        "category": "instruction_trap",
        "prompt": "오직 영어 단어 3개 'apple, banana, orange'만 쉼표로 구분해서 출력해. 한국어나 추가 문장은 절대 쓰지 마.",
        "instruction_type": "exact_three_fruits",
    },
    {
        "id": 90,
        "category": "instruction_trap",
        "prompt": "다른 어떤 부연 설명이나 마크다운 코드블록도 없이 오직 {\"status\": \"ok\"} 라는 JSON 문자열 한 줄만 출력해.",
        "instruction_type": "exact_json_status",
    },

    # ==========================================
    # 5. Dialect Understanding & Generation (10)
    # ==========================================
    {
        "id": 91,
        "category": "dialect_eval",
        "prompt": "경상도/울산 방언에서 '단디 해라'가 표준어로 무슨 뜻이야?",
        "keywords": ["단단히", "확실히", "조심", "똑바로"],
    },
    {
        "id": 92,
        "category": "dialect_eval",
        "prompt": "경상도 방언 '와이리 덥노?'를 표준어로 바꾸면 어떤 뜻이야?",
        "keywords": ["왜 이렇게", "왜 이리", "덥니", "덥냐"],
    },
    {
        "id": 93,
        "category": "dialect_eval",
        "prompt": "경상도 사투리 '밥 문나?'는 표준어로 무슨 뜻이야?",
        "keywords": ["밥 먹었니", "식사했니", "밥 먹었어"],
    },
    {
        "id": 94,
        "category": "dialect_eval",
        "prompt": "경상도 방언 '어데 가노?'의 표준어 의미는 뭐야?",
        "keywords": ["어디 가니", "어디 가", "어디 가냐"],
    },
    {
        "id": 95,
        "category": "dialect_eval",
        "prompt": "경상도 사투리 '뭐라 카노?'의 표준어 의미는 뭐야?",
        "keywords": ["뭐라고 하니", "무슨 말을 하니", "뭐라고 해"],
    },
    {
        "id": 96,
        "category": "dialect_eval",
        "prompt": "표준어 문장 '오늘 날씨가 정말 좋다'를 자연스러운 울산/경상 방언으로 바꿔줘.",
        "dialect_gen": True,
        "markers": ["좋네", "좋데이", "억수로", "억시로", "직이네"],
    },
    {
        "id": 97,
        "category": "dialect_eval",
        "prompt": "표준어 문장 '이 음식 정말 맛있다'를 자연스러운 울산/경상 방언으로 바꿔줘.",
        "dialect_gen": True,
        "markers": ["맛있네", "맛있데이", "맛나네", "억수로", "참말로"],
    },
    {
        "id": 98,
        "category": "dialect_eval",
        "prompt": "표준어 질문 '너 지금 뭐 하고 있어?'를 자연스러운 울산/경상 방언으로 바꿔줘.",
        "dialect_gen": True,
        "markers": ["하노", "카노", "니 지금", "뭐 하노"],
    },
    {
        "id": 99,
        "category": "dialect_eval",
        "prompt": "표준어 문장 '조심해서 집에 가'를 자연스러운 울산/경상 방언으로 바꿔줘.",
        "dialect_gen": True,
        "markers": ["단디", "챙기가", "가라", "조심해가"],
    },
    {
        "id": 100,
        "category": "dialect_eval",
        "prompt": "표준어 격려 문장 '걱정하지 마, 잘 될 거야'를 자연스러운 울산/경상 방언으로 바꿔줘.",
        "dialect_gen": True,
        "markers": ["걱정 마라", "잘 될 끼다", "될기다", "된다"],
    },
]


def main():
    out_path = Path("data/preservation_benchmark_100.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for item in BENCHMARK_100:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"Generated {len(BENCHMARK_100)} preservation benchmark prompts to {out_path}")


if __name__ == "__main__":
    main()
