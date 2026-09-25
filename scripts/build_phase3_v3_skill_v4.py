"""Construct five dialect skills and preservation replay from disjoint source rows."""

import gzip
import json
import random
import re
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path

ROOT = Path("data/ulsan_dialect_skill_v4")
RNG = random.Random(4304)
SKILL_QUOTAS = {"conversion": 1225, "meaning": 700, "grammar": 525,
                "correction": 525, "contextual": 525}
NOISE = re.compile(r"\{[^}]*\}|#[^#]*#|&[^&]*&|\(\(.*?\)\)|<[^>]*>|[A-Za-z]{5,}")
MARKER = re.compile(r"아이가|데이|맞나|어데|온나|묵|그라|마이|억수로|억시로|직이|퍼뜩|단디|[가-힣]카[가-힣]|뿌|[가-힣]노\b|[가-힣]제\b|[가-힣]나\b")
BENCHMARKS = ["reports/phase3-v3/gate-40.jsonl", "data/regression_benchmark_150.jsonl",
              "reports/phase3-v3/final-dialect-holdout-50.jsonl",
              "reports/phase3-v3/independent-dialect-20.jsonl"]
HELD_SENTENCES = ("오늘 날씨가 정말 좋다", "이 음식 정말 맛있다", "너 지금 뭐 하고 있어",
                  "조심해서 집에 가", "걱정하지 마, 잘 될 거야", "늦기 전에 서둘러 출발하자",
                  "물건이 너무 많아서 정리하기 힘들다")


def read(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


def norm(s):
    return re.sub(r"\s+", "", s).lower()


def msg(role, content):
    return {"role": role, "content": content}


def classify(text):
    for marker in ("퍼뜩", "단디", "천지빼까리", "파이다", "어데", "온나", "맞나", "뭇나", "묵", "그라", "데이", "노", "아이가", "제", "카", "뿌", "억수로", "억시로", "마이", "직이"):
        if marker in text:
            return marker
    return "other"


def complete(text):
    return bool(re.search(r"[.?!]$|(?:요|다|어|네|제|데이|노|나|아이가)$", text.strip()))


LEXICON = [
    ("퍼뜩", "빨리, 얼른", "퍼뜩 준비하고 나가자", "빨리 준비하고 나가자"),
    ("단디", "꼼꼼하게, 확실히", "가방 단디 챙기고 온나", "가방을 꼼꼼히 챙겨서 와"),
    ("천지빼까리", "아주 많음", "책이 천지빼까리라 정리도 못 하겠다", "책이 아주 많아서 정리도 못 하겠다"),
    ("파이다", "별로다, 좋지 않다", "그 방법은 내 보기엔 파이다", "그 방법은 내가 보기엔 별로다"),
    ("어데", "어디", "니 오늘은 어데서 만날래?", "너 오늘은 어디에서 만날래?"),
    ("온나", "와라", "시간 되면 이쪽으로 온나", "시간 되면 이쪽으로 와라"),
    ("맞나", "맞니?, 정말이니?", "내일 비 온다는 게 맞나?", "내일 비 온다는 게 맞니?"),
    ("뭇나", "먹었니?", "아침은 챙겨 뭇나?", "아침은 챙겨 먹었니?"),
    ("묵다", "먹다", "저녁은 집에서 묵자", "저녁은 집에서 먹자"),
    ("와", "왜", "와 벌써 가려 카노?", "왜 벌써 가려고 하니?"),
    ("그라노", "그러니?", "친구한테 와 그라노?", "친구한테 왜 그러니?"),
    ("아이가", "아니야?, 그렇잖아", "이게 니가 찾던 거 아이가?", "이게 네가 찾던 것이 아니야?"),
    ("맞제", "맞지?, 그렇지?", "이 길로 가는 게 맞제?", "이 길로 가는 것이 맞지?"),
    ("마이", "많이", "오늘 사람이 마이 왔네", "오늘 사람이 많이 왔네"),
    ("억수로", "매우, 굉장히", "바람이 억수로 세네", "바람이 매우 세네"),
    ("카노", "하니?, 말하니?", "지금 뭐라 카노?", "지금 뭐라고 말하니?"),
    ("와이리", "왜 이렇게", "와이리 바쁘노?", "왜 이렇게 바쁘니?"),
    ("문나", "먹었니?", "저녁은 챙겨 문나?", "저녁은 챙겨 먹었니?"),
    ("카노", "말하니?, 하니?", "니 지금 그 소리를 와 카노?", "너 지금 그 말을 왜 하니?"),
    ("그라노", "그러니?", "갑자기 니 와 그라노?", "갑자기 너 왜 그러니?"),
]


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    benchmark_prompts = {norm(x["prompt"]) for p in BENCHMARKS for x in read(p)}
    groups = defaultdict(list)
    seen_prompt = set()
    openings = Counter()
    target_counts = Counter()
    template_counts = Counter()
    source_used = set()

    def add(category, messages, template, source=None, max_template=150):
        if source is not None and source in source_used:
            return False
        prompt = messages[-2]["content"]
        answer = messages[-1]["content"]
        k = norm(prompt)
        opening = norm(answer)[:14]
        noisy = NOISE.search(prompt + answer) if category in SKILL_QUOTAS else re.search(r"&[^&]*&|\(\(.*?\)\)", prompt + answer)
        opening_limit = 5 if category in SKILL_QUOTAS else 30
        if (not prompt or not answer or k in benchmark_prompts or k in seen_prompt
                or any(norm(s) in k for s in HELD_SENTENCES) or noisy
                or template_counts[template] >= max_template or openings[(category, opening)] >= opening_limit
                or (category in SKILL_QUOTAS and target_counts[norm(answer)] >= 2)):
            return False
        groups[category].append({"task": category, "category": "dialect_skill" if category in SKILL_QUOTAS
            else category, "messages": messages, "template": template})
        seen_prompt.add(k)
        openings[(category, opening)] += 1
        target_counts[norm(answer)] += 1
        template_counts[template] += 1
        if source is not None:
            source_used.add(source)
        return True

    dense = []
    seen_pair = set()
    for x in read("data/ulsan_dialect_dense/train.jsonl.gz"):
        standard, dialect = x.get("standard_text", "").strip(), x.get("dialect_text", "").strip()
        if (x.get("task") != "standard_to_dialect" or x.get("quality_grade") != "A"
                or x.get("synthetic") is not False or x.get("metadata", {}).get("ulsan_tier") not in ("U0", "U1")
                or not (15 <= len(standard) <= 115 and 15 <= len(dialect) <= 115)
                or NOISE.search(standard + dialect) or not MARKER.search(dialect)
                or any(norm(s) in norm(standard) for s in HELD_SENTENCES)):
            continue
        ratio = SequenceMatcher(None, standard, dialect).ratio()
        pair = (norm(standard), norm(dialect))
        if not .45 <= ratio <= .965 or pair in seen_pair:
            continue
        seen_pair.add(pair)
        dense.append((standard, dialect))
    RNG.shuffle(dense)
    frequency = Counter(classify(d) for _, d in dense)
    dense.sort(key=lambda pair: (not (complete(pair[0]) and complete(pair[1])), frequency[classify(pair[1])]))
    marker_counts = Counter()
    conv_prompts = (
        "이 문장을 울산에서 들을 법한 자연스러운 말투로 바꿔줘: {}",
        "경상도 말투를 과장하지 말고 자연스럽게 옮겨줘: {}",
        "친한 사람에게 말하듯 울산/경상 말투로 고쳐줘: {}",
        "뜻은 유지하고 종결어미까지 자연스럽게 바꿔줘. 원문: {}",
        "다음 말을 일상적인 울산 말투로 표현해줘: {}",
        "표준어 문장을 경상도식 구어로 옮겨줘: {}",
        "울산에서도 자연스럽게 들릴 말투로 한 문장만 써줘: {}",
        "의미를 바꾸지 말고 경상도 방언으로 말해줘: {}",
        "이 문장을 친구에게 쓰는 울산 말투로 말해봐: {}",
    )
    for standard, dialect in dense:
        if len(groups["conversion"]) >= SKILL_QUOTAS["conversion"]:
            break
        marker = classify(dialect)
        if marker_counts[marker] >= (350 if marker in ("아이가", "제") else 270):
            continue
        idx = len(groups["conversion"]) % len(conv_prompts)
        if add("conversion", [msg("user", conv_prompts[idx].format(standard)), msg("assistant", dialect)],
               f"conversion_{idx}", (standard, dialect)):
            marker_counts[marker] += 1

    meaning_prompts = (
        "이 경상도 말의 뜻을 자연스러운 표준어로 풀어줘: {}",
        "다음 말을 듣고 표준어로는 어떻게 이해하면 될까? {}",
        "이 문장을 표준어로 옮겨 설명해줘: {}",
        "경상도식 발화를 일상적인 표준어로 바꿔줘: {}",
        "이 말이 무슨 뜻인지 표준어 한 문장으로 알려줘: {}",
        "울산에서도 들을 수 있는 이 말의 표준어 의미는? {}",
        "다음 방언 문장을 과장 없이 표준어로 해석해줘: {}",
    )
    for standard, dialect in dense:
        if len(groups["meaning"]) >= 660:
            break
        idx = len(groups["meaning"]) % len(meaning_prompts)
        add("meaning", [msg("user", meaning_prompts[idx].format(dialect)), msg("assistant", standard)],
            f"meaning_{idx}", (standard, dialect))
    contexts = ("친구에게 들었어.", "동네에서 들은 말이야.", "대화 중에 들었어.",
                "울산에서도 쓰이지만 경상도 전반에 있는 표현이야.", "가족이 한 말이야.")
    for marker, meaning, dialect, standard in LEXICON:
        for i in range(7):
            if len(groups["meaning"]) >= SKILL_QUOTAS["meaning"]:
                break
            prompt = f"{contexts[i % len(contexts)]} '{dialect}'에서 '{marker}'가 무슨 뜻인지 표준어로 풀어줘."
            answer = f"'{marker}'는 여기서 {meaning}라는 뜻이야. 문장 전체로는 '{standard}' 정도야."
            add("meaning", [msg("user", prompt), msg("assistant", answer)], f"lexicon_{i}", max_template=30)

    # A deliberately malformed stacked ending gives a clear correction target.
    def corrupt(dialect):
        for marker, fake in (("데이", "데이노"), ("맞나", "맞나제"),
                             ("온나", "온나노"), ("했제", "했제노"), ("맞제", "맞제노")):
            if marker in dialect:
                return dialect.replace(marker, fake, 1), marker
        match = re.search(r"(?:갔|왔|했|있|없|먹|묵|맞|가|보|되|하)나\?|[가-힣]노([?.! ]|$)|(?:하|맞|있|없|됐|했|좋|그러)제([?.! ]|$)", dialect)
        if match:
            marker = "나" if match.group().endswith("나?") else ("노" if "노" in match.group() else "제")
            pos = match.start() + match.group().rindex(marker)
            return dialect[:pos] + marker + "데이" + dialect[pos + len(marker):], marker
        if "아이가" in dialect:
            return dialect.replace("아이가", "아이가데이", 1), "아이가"
        return None, None

    grammar_prompts = (
        "어미를 겹쳐 쓰지 않는 자연스러운 경상도 표현을 골라줘. A: {a} / B: {b}",
        "울산에서도 자연스럽게 들릴 쪽은? A: {a} / B: {b}",
        "둘 중 종결어미가 덜 어색한 문장을 선택하고 이유를 짧게 말해줘. A: {a} / B: {b}",
        "경상도 말투를 과장하지 않은 문장은 어느 쪽이야? A: {a} / B: {b}",
        "말끝을 잘못 겹친 문장을 피해서 골라줘. A: {a} / B: {b}",
    )
    correction_prompts = (
        "어색하게 방언 어미를 겹쳐 쓴 문장을 자연스럽게 고쳐줘: {}",
        "가짜 경상도 말투처럼 들리는 부분을 바로잡아줘: {}",
        "이 울산/경상 말투의 어색한 말끝을 수정해줘: {}",
        "뜻은 그대로 두고 방언 표현을 자연스럽게 교정해줘: {}",
        "과장된 어미를 덜어 실제 대화처럼 고쳐줘: {}",
    )
    ending_counts = defaultdict(Counter)
    for category in ("grammar", "correction"):
        for standard, dialect in dense:
            if len(groups[category]) >= SKILL_QUOTAS[category]:
                break
            wrong, ending = corrupt(dialect)
            if not wrong or wrong == dialect or not complete(dialect) or not complete(standard):
                continue
            family = "아이가" if ending == "아이가" else ("나" if ending in ("나", "맞나") else ending)
            cap = {"아이가": 220, "나": 210, "노": 110, "제": 65, "데이": 35}.get(family, 20)
            if ending_counts[category][family] >= cap:
                continue
            idx = len(groups[category]) % 5
            if category == "grammar":
                a, b = (dialect, wrong) if len(groups[category]) % 2 == 0 else (wrong, dialect)
                label = "A" if a == dialect else "B"
                prompt = grammar_prompts[idx].format(a=a, b=b)
                answer = f"{dialect} ({label}) — '{ending}' 뒤에 다른 어미를 겹치지 않는 편이 자연스럽다."
            else:
                prompt = correction_prompts[idx].format(wrong)
                answer = dialect
            if add(category, [msg("user", prompt), msg("assistant", answer)], f"{category}_{idx}",
                   max_template=110):
                ending_counts[category][family] += 1

    context_prompts = (
        ("울산 말투로 연습하자. 먼저 '{}'는 어떻게 해?", "좋아. 같은 말투로 '{}'도 말해줘."),
        ("친구에게 경상도식으로 말해 봐. '{}'", "그 말투 유지해서 이것도 부탁해: '{}'"),
        ("뜻을 유지하면서 울산 말투로 해보자: '{}'", "방금처럼 자연스럽게 이어서 '{}'도 해봐."),
        ("경상도 말투로 대화하자. '{}'", "응, 계속 그 말투로 '{}'도 말해줘."),
        ("울산에서 들을 법한 말투로 바꿔줘: '{}'", "좋아. 다음 말도 같은 말투로: '{}'"),
    )
    available = [(s, d) for s, d in dense if (s, d) not in source_used] + [(s, d) for s, d in dense if (s, d) in source_used]
    for i in range(0, len(available) - 1, 2):
        if len(groups["contextual"]) >= SKILL_QUOTAS["contextual"]:
            break
        s1, d1 = available[i]
        s2, d2 = available[i + 1]
        idx = len(groups["contextual"]) % 5
        first, second = context_prompts[idx]
        messages = [msg("user", first.format(s1)), msg("assistant", d1),
                    msg("user", second.format(s2)), msg("assistant", d2)]
        if add("contextual", messages, f"contextual_{idx}", max_template=110):
            source_used.add((s1, d1))

    for category, count in SKILL_QUOTAS.items():
        if len(groups[category]) != count:
            raise RuntimeError(f"{category}: {len(groups[category])}/{count}; dense={len(dense)}")

    # Factual/general replay: distinct human-facing explanations and problem solving.
    v1 = list(read("data/ulsan_dialect_phase3/train.jsonl"))
    RNG.shuffle(v1)
    factual_cats = {"science", "math_logic", "coding", "ulsan_local", "daily_chat", "short_banter"}
    for row in v1:
        if len(groups["factual_preservation"]) >= 1909:
            break
        messages = row.get("messages", [])
        if row.get("category") in factual_cats and len(messages) >= 2 and 25 <= len(messages[-1]["content"]) <= 550:
            add("factual_preservation", messages, f"factual_{row['category']}", max_template=600)
    for path in ("data/ulsan_dialect_phase3_v2/train.jsonl", "data/ulsan_dialect_phase3_v2/validation.jsonl"):
        for row in read(path):
            if row.get("category") == "factual_preservation":
                add("factual_preservation", row["messages"], "existing_factual", max_template=120)
    for row in v1:
        if len(groups["factual_preservation"]) >= 650:
            break
        messages = row.get("messages", [])
        if row.get("category") == "aihub_casual_turn" and len(messages) == 2:
            answer = messages[-1]["content"].strip()
            if (20 <= len(answer) <= 250 and answer.endswith((".", "?", "!", "요", "다"))
                    and not re.search(r"~|&[^&]*&|\{[^}]*\}", answer)):
                add("factual_preservation", messages, "casual_general", max_template=550)
    # Arithmetic and unit questions have deterministic targets and diverse values.
    cases = [(a, b, op) for a in range(12, 95) for b in range(3, 42)
             for op in range(6) if (a * 7 + b * 11 + op) % 5 == 0]
    RNG.shuffle(cases)
    for a, b, op in cases:
        if len(groups["factual_preservation"]) >= 1909:
            break
        if op == 0:
            prompt, answer = f"{a}개와 {b}개를 합치면 모두 몇 개야? 계산도 설명해줘.", f"{a}와 {b}를 더하면 {a+b}개야. 두 묶음을 하나로 합치는 계산이니까 {a} + {b} = {a+b}로 구하면 돼."
        elif op == 1:
            prompt, answer = f"물건 {a+b}개 중 {b}개를 사용했어. 몇 개 남았어?", f"{a}개가 남아. 처음 {a+b}개에서 사용한 {b}개를 빼면 {a+b} - {b} = {a}가 되니까."
        elif op == 2:
            prompt, answer = f"한 상자에 {b}개씩 들어 있어. {a}상자면 몇 개야?", f"모두 {a*b}개야. 상자 수 {a}에 상자당 {b}개를 곱해서 {a} × {b} = {a*b}로 계산해."
        elif op == 3:
            prompt, answer = f"{a}분에 {b}분을 더하면 몇 분이야?", f"합계는 {a+b}분이야. 같은 시간 단위끼리 더하면 되므로 {a}분 + {b}분 = {a+b}분이야."
        elif op == 4:
            prompt, answer = f"가로 {a}cm, 세로 {b}cm인 직사각형 넓이를 구해줘.", f"넓이는 {a*b}제곱센티미터야. 직사각형 넓이는 가로 곱하기 세로이므로 {a} × {b} = {a*b}야."
        else:
            prompt, answer = f"{a*b}개를 {a}명에게 똑같이 나누면 한 명당 몇 개야?", f"한 사람당 {b}개씩 받아. 전체 {a*b}개를 {a}명으로 나누면 {a*b} ÷ {a} = {b}니까."
        add("factual_preservation", [msg("user", prompt), msg("assistant", answer)],
            f"arithmetic_{op}", max_template=260)

    # Existing real replay is retained, then augmented with varied grounded recall and exact instructions.
    for path in ("data/ulsan_dialect_phase3_v2/train.jsonl", "data/ulsan_dialect_phase3_v2/validation.jsonl"):
        for row in read(path):
            if row.get("category") == "instruction_preservation":
                add("instruction_preservation", row["messages"], f"existing_{row.get('task')}", max_template=200)

    names = ["민지", "수빈", "현우", "가은", "도윤", "하린", "유진", "지훈", "서연", "지민", "예린", "태민", "주원", "다은", "수아", "준서", "지우", "은서", "윤호", "소윤"]
    values = ["복숭아", "보리차", "녹차", "딸기우유", "크림빵", "파란 우산", "검은 가방", "도서관", "남문", "수영장", "시청역", "대공원", "화요일", "목요일", "토요일", "오후 세 시", "오전 열 시", "7번 버스", "장미꽃", "추리 소설", "자전거", "배드민턴", "사진 찍기", "국수", "김밥"]
    recall_types = [
        ("{name}가 좋아하는 건 {value}야. 나중에 물어보면 기억해 줘.", "알겠어. {name}가 좋아하는 건 {value}라고 기억할게.", "아까 {name}가 좋아한다고 한 게 뭐였지?", "{name}가 좋아하는 건 {value}야."),
        ("{name}와 약속한 장소는 {value}야. 기억해 둬.", "응, 약속 장소는 {value}로 기억할게.", "{name}와 어디서 만나기로 했지?", "{name}와 만날 장소는 {value}야."),
        ("내가 {name}한테 건넬 물건은 {value}야.", "{name}에게 {value}를 건넬 거구나.", "{name}에게 뭘 건네야 하지?", "{name}에게 {value}를 건네면 돼."),
        ("{name}의 메모에 적힌 내용은 {value}야.", "메모에는 {value}라고 적혀 있구나.", "{name}의 메모 내용 기억나?", "메모 내용은 {value}야."),
        ("이번 주 {name}와 관련된 키워드는 {value}야.", "좋아, {name}와 관련된 키워드를 기억할게.", "이번 주 {name}의 키워드는 뭐였어?", "키워드는 {value}야."),
    ]
    combos = [(name, value, idx) for name in names for value in values for idx in range(5)]
    RNG.shuffle(combos)
    occasions = ["도서관", "공원", "카페", "학교", "시장", "버스 정류장", "체육관", "식당", "집", "역 앞", "서점", "산책길", "전시회", "영화관", "사무실", "편의점", "놀이터", "강변", "회의실", "운동장", "박물관", "정원", "주차장", "광장", "공연장"]
    for name, value, idx in combos:
        if len(groups["instruction_preservation"]) >= 760:
            break
        u1, a1, u2, a2 = recall_types[idx]
        occasion = occasions[values.index(value)]
        messages = [msg("user", f"{occasion}에서 나눈 얘기야. " + u1.format(name=name, value=value)), msg("assistant", a1.format(name=name, value=value)),
                    msg("user", u2.format(name=name) + f" {occasion}에서 나눈 얘기를 떠올려줘."), msg("assistant", a2.format(name=name, value=value))]
        add("instruction_preservation", messages, f"memory_{idx}", max_template=150)
    for name in names:
        for value in values:
            if len(groups["instruction_preservation"]) >= 954:
                break
            for idx in range(4):
                if len(groups["instruction_preservation"]) >= 954:
                    break
                if idx == 0:
                    prompt, answer = f"설명 없이 '{name}: {value}'를 그대로 한 줄에 써줘.", f"{name}: {value}"
                elif idx == 1:
                    prompt, answer = f"{name}의 물건 {value}를 JSON 문자열 값으로 담아줘. key는 item.", json.dumps({"item": value}, ensure_ascii=False)
                elif idx == 2:
                    prompt, answer = f"'{name}'와 '{value}'를 순서대로 쉼표로 연결해 한 줄만 출력해줘.", f"{name}, {value}"
                else:
                    prompt, answer = f"목록에 {name}, {value}가 있다. 두 항목을 번호 1, 2로만 적어줘.", f"1. {name}\n2. {value}"
                add("instruction_preservation", [msg("user", prompt), msg("assistant", answer)],
                    f"strict_{idx}", max_template=120)
    if len(groups["factual_preservation"]) < 1909 or len(groups["instruction_preservation"]) < 954:
        raise RuntimeError(f"replay shortfall: factual={len(groups['factual_preservation'])}, instruction={len(groups['instruction_preservation'])}; templates={[(k,v) for k,v in template_counts.items() if k.startswith(('memory_', 'strict_', 'existing_'))]}")

    skills = [row for key in SKILL_QUOTAS for row in groups[key]]
    full = skills + groups["factual_preservation"][:1909] + groups["instruction_preservation"][:954]
    RNG.shuffle(full)
    f = groups["factual_preservation"][:1909].copy(); m = groups["instruction_preservation"][:954].copy()
    RNG.shuffle(f); RNG.shuffle(m)
    d1, d2 = [], []
    stage_quotas = {"conversion": (125, 90), "meaning": (110, 55), "grammar": (55, 40),
                    "correction": (55, 40), "contextual": (39, 39)}
    for category, (n1, n2) in stage_quotas.items():
        pool = groups[category].copy()
        RNG.shuffle(pool)
        if category == "meaning":
            lexicon = [row for row in pool if row["template"].startswith("lexicon_")]
            others = [row for row in pool if not row["template"].startswith("lexicon_")]
            pool = lexicon + others
            d1.extend(pool[:n1]); d2.extend(pool[:20] + pool[n1:n1+n2-20])
        else:
            d1.extend(pool[:n1]); d2.extend(pool[n1:n1+n2])
    stage1 = d1 + f[:60] + m[:36]
    stage2 = d2 + f[60:204] + m[36:108]
    RNG.shuffle(stage1); RNG.shuffle(stage2)
    pilot = stage1 + stage2
    curriculum_full = pilot + full
    for name, rows in (("full-train", full), ("pilot", pilot), ("full-curriculum", curriculum_full)):
        with (ROOT / f"{name}.jsonl").open("w", encoding="utf-8") as out:
            for row in rows:
                out.write(json.dumps({k: v for k, v in row.items() if k != "template"}, ensure_ascii=False) + "\n")
    summary = {"skill_counts": {k: len(groups[k]) for k in SKILL_QUOTAS},
               "mixture_counts": {"dialect_skill": len(skills), "factual_general": 1909, "memory_instruction": 954},
               "pilot_stage1": {"dialect": 384, "factual": 60, "memory_instruction": 36},
               "pilot_stage2": {"dialect": 264, "factual": 144, "memory_instruction": 72},
               "benchmark_exact_prompt_overlap": sum(norm(r["messages"][-2]["content"]) in benchmark_prompts for r in full),
               "max_response_opening_count": max(openings.values()), "max_template_count": max(template_counts.values()),
               "conversion_markers": dict(marker_counts), "source_pairs": len(dense)}
    summary["ending_selection_counts"] = {k: dict(v) for k, v in ending_counts.items()}
    (ROOT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
