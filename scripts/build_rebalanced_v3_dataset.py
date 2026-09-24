"""Build rebalanced Phase 3 v3 dataset with factual & instruction preservation replay.

Mixture Target (~6,000 samples):
- Dialect SFT: ~2,700 samples (45.0%)
- Factual & General Knowledge Replay: ~2,400 samples (40.0%)
- Multi-turn Memory & Instruction Following Replay: ~900 samples (15.0%)

Zero leakage: Strict filtering against preservation_benchmark_100 and gate-60.
"""

from __future__ import annotations

import difflib
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any

RANDOM_SEED = 42
random.seed(RANDOM_SEED)

DATA_DIR = Path("data")
OUT_DIR = DATA_DIR / "ulsan_dialect_phase3_v3_rebalanced"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BENCH_100_PATH = DATA_DIR / "preservation_benchmark_100.jsonl"
GATE_60_PATH = Path("reports/phase3-v3/gate-60.jsonl")


def load_benchmark_prompts() -> list[str]:
    prompts = []
    for path in (BENCH_100_PATH, GATE_60_PATH):
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        item = json.loads(line)
                        p = item.get("prompt", "").strip()
                        if p:
                            prompts.append(p)
    return list(set(prompts))


def is_benchmark_overlap(prompt: str, benchmark_prompts: list[str]) -> bool:
    p_norm = re.sub(r"[^\w]", "", prompt).lower()
    for bp in benchmark_prompts:
        bp_norm = re.sub(r"[^\w]", "", bp).lower()
        if p_norm == bp_norm:
            return True
        if len(p_norm) > 8 and len(bp_norm) > 8:
            if difflib.SequenceMatcher(None, p_norm, bp_norm).ratio() > 0.75:
                return True
    return False


def get_dialect_samples(target: int, bench_prompts: list[str]) -> list[dict[str, Any]]:
    print(f"Collecting Dialect SFT samples (target: {target})...")
    v3_path = DATA_DIR / "ulsan_dialect_phase3_v3" / "train.jsonl"
    if not v3_path.exists():
        raise FileNotFoundError(f"{v3_path} not found")

    samples = []
    opening_counter = Counter()

    with open(v3_path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            msgs = row.get("messages", [])
            if len(msgs) < 2:
                continue

            user_msg = msgs[0]["content"].strip()
            asst_msg = msgs[1]["content"].strip()

            if is_benchmark_overlap(user_msg, bench_prompts):
                continue

            # Cap repetitive openings to avoid severe template memorization
            prefix = asst_msg[:25]
            if opening_counter[prefix] >= 8:
                continue
            opening_counter[prefix] += 1

            samples.append({
                "id": row.get("id", f"dia_{len(samples)}"),
                "task": row.get("task", "dialect_conversation"),
                "category": "dialect_sft",
                "subcategory": row.get("category", "dialect"),
                "messages": msgs,
            })

    random.shuffle(samples)
    selected = samples[:target]
    print(f"Selected {len(selected)} dialect samples.")
    return selected


def get_factual_samples(target: int, bench_prompts: list[str]) -> list[dict[str, Any]]:
    print(f"Collecting Factual Replay samples (target: {target})...")
    samples = []

    # Source 1: phase3_v2 factual_preservation
    v2_path = DATA_DIR / "ulsan_dialect_phase3_v2" / "train.jsonl"
    if v2_path.exists():
        with open(v2_path, "r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                if row.get("category") == "factual_preservation":
                    u = row["messages"][0]["content"]
                    if not is_benchmark_overlap(u, bench_prompts):
                        samples.append({
                            "id": f"fact_{len(samples)}",
                            "task": row.get("task", "factual_replay"),
                            "category": "factual_preservation",
                            "messages": row["messages"],
                        })

    # Source 2: phase3 science, math_logic, coding
    v3_raw_path = DATA_DIR / "ulsan_dialect_phase3" / "train.jsonl"
    if v3_raw_path.exists() and len(samples) < target:
        with open(v3_raw_path, "r", encoding="utf-8") as f:
            for line in f:
                if len(samples) >= target * 2:
                    break
                row = json.loads(line)
                cat = row.get("category", "")
                if cat in ("math_logic", "science", "coding"):
                    msgs = row.get("messages", [])
                    if len(msgs) == 2:
                        u = msgs[0]["content"]
                        a = msgs[1]["content"]
                        if is_benchmark_overlap(u, bench_prompts):
                            continue
                        a_clean = (
                            a.replace("했제", "했습니다")
                            .replace("이제", "입니다")
                            .replace("한데이", "합니다")
                            .replace("기라", "것입니다")
                            .replace("맞제", "맞습니다")
                        )
                        samples.append({
                            "id": f"fact_p3_{len(samples)}",
                            "task": f"{cat}_replay",
                            "category": "factual_preservation",
                            "messages": [
                                {"role": "user", "content": u},
                                {"role": "assistant", "content": a_clean},
                            ],
                        })

    random.shuffle(samples)
    selected = samples[:target]
    print(f"Selected {len(selected)} factual preservation samples.")
    return selected


def get_multiturn_instruction_samples(target: int, bench_prompts: list[str]) -> list[dict[str, Any]]:
    print(f"Collecting Multi-turn & Instruction Replay samples (target: {target})...")
    samples = []

    # Source 1: phase3_v2 instruction_preservation
    v2_path = DATA_DIR / "ulsan_dialect_phase3_v2" / "train.jsonl"
    if v2_path.exists():
        with open(v2_path, "r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                if row.get("category") == "instruction_preservation":
                    u = row["messages"][-2]["content"]
                    if not is_benchmark_overlap(u, bench_prompts):
                        samples.append({
                            "id": f"inst_{len(samples)}",
                            "task": row.get("task", "instruction_replay"),
                            "category": "instruction_preservation",
                            "messages": row["messages"],
                        })

    # Expand stateful memory templates if needed to reach target
    MEMORY_TEMPLATES = [
        [
            ("제 이름은 {name}이고, 직업은 {job}입니다.", "만나서 반갑습니다, {name} 님! {job} 분야에서 일하고 계시군요. 무엇을 도와드릴까요?"),
            ("제가 무슨 일을 하고 있다고 말씀드렸었죠?", "{name} 님께서는 직업이 **{job}**이라고 말씀하셨습니다."),
        ],
        [
            ("제가 이번 주말에 {location}으로 1박 2일 여행을 가려고 해요.", "{location} 여행 정말 기대되시겠어요! 날씨와 숙소 잘 챙기시길 바랍니다."),
            ("제가 주말에 어디로 여행 간다고 했었나요?", "이번 주말에 **{location}**으로 1박 2일 여행을 계획 중이라고 말씀하셨습니다."),
        ],
        [
            ("저는 평소에 커피 중에서 {coffee}만 즐겨 마셔요.", "{coffee}를 즐겨 마시시는군요! 향과 풍미가 매력적인 음료죠."),
            ("제가 좋아하는 커피 메뉴가 뭐였죠?", "평소에 **{coffee}**만 즐겨 드신다고 말씀하셨습니다."),
        ],
        [
            ("중요한 회의가 {day}요일 오후 {time}시에 잡혀 있어요.", "{day}요일 오후 {time}시 회의 메모해 두겠습니다. 중요한 안건 잘 풀리시길 바랍니다."),
            ("회의가 언제 잡혀 있다고 했었죠?", "중요 회의 일정은 **{day}요일 오후 {time}시**라고 말씀하셨습니다."),
        ],
        [
            ("우리 집 반려견 이름은 {dog}이고, 견종은 {breed}예요.", "{dog}라는 이름 정말 다정하네요! {breed} 특유의 활발함이 넘칠 것 같아요."),
            ("우리 강아지 이름과 견종이 뭐였는지 기억해요?", "반려견의 이름은 **{dog}**이고, 견종은 **{breed}**라고 말씀하셨습니다."),
        ],
    ]
    NAMES = ["동현", "서연", "재민", "하은", "우진", "수빈", "태양", "예진", "현우", "지민", "민재", "채원", "준서", "지우"]
    JOBS = ["데이터 분석가", "소프트웨어 엔지니어", "그래픽 디자이너", "회계사", "마케터", "건축사", "연구원", "초등학교 교사", "약사"]
    LOCATIONS = ["경주", "포항", "여수", "강릉", "전주", "통영", "속초", "단양", "제주도", "남해"]
    COFFEES = ["바닐라 라떼", "콜드브루", "카페 라떼", "카푸치노", "에스프레소", "플랫 화이트", "아메리카노"]
    DAYS = ["월", "화", "수", "목", "금"]
    TIMES = ["2", "3", "4", "5", "6"]
    DOGS = ["보리", "콩이", "두부", "호두", "뭉치", "별이", "구름이", "초코", "까미", "모찌"]
    BREEDS = ["골든 리트리버", "비숑 프리제", "말티즈", "웰시코기", "시바견", "포메라니안", "푸들"]

    while len(samples) < target:
        tmpl = random.choice(MEMORY_TEMPLATES)
        n = random.choice(NAMES)
        j = random.choice(JOBS)
        loc = random.choice(LOCATIONS)
        c = random.choice(COFFEES)
        d = random.choice(DAYS)
        tm = random.choice(TIMES)
        dg = random.choice(DOGS)
        b = random.choice(BREEDS)

        t1_u = tmpl[0][0].format(name=n, job=j, location=loc, coffee=c, day=d, time=tm, dog=dg, breed=b)
        t1_a = tmpl[0][1].format(name=n, job=j, location=loc, coffee=c, day=d, time=tm, dog=dg, breed=b)
        t2_u = tmpl[1][0].format(name=n, job=j, location=loc, coffee=c, day=d, time=tm, dog=dg, breed=b)
        t2_a = tmpl[1][1].format(name=n, job=j, location=loc, coffee=c, day=d, time=tm, dog=dg, breed=b)

        if is_benchmark_overlap(t2_u, bench_prompts):
            continue

        samples.append({
            "id": f"inst_gen_{len(samples)}",
            "task": "stateful_memory",
            "category": "instruction_preservation",
            "messages": [
                {"role": "user", "content": t1_u},
                {"role": "assistant", "content": t1_a},
                {"role": "user", "content": t2_u},
                {"role": "assistant", "content": t2_a},
            ],
        })

    random.shuffle(samples)
    selected = samples[:target]
    print(f"Selected {len(selected)} multi-turn & instruction samples.")
    return selected


def main():
    bench_prompts = load_benchmark_prompts()
    print(f"Loaded {len(bench_prompts)} benchmark prompts for decontamination.")

    target_dia = 2700
    target_fact = 2400
    target_inst = 900
    total = target_dia + target_fact + target_inst

    dia = get_dialect_samples(target_dia, bench_prompts)
    fact = get_factual_samples(target_fact, bench_prompts)
    inst = get_multiturn_instruction_samples(target_inst, bench_prompts)

    all_train = dia + fact + inst
    random.shuffle(all_train)

    train_split = all_train[:5600]
    val_split = all_train[5600:]

    train_file = OUT_DIR / "train.jsonl"
    val_file = OUT_DIR / "validation.jsonl"

    with open(train_file, "w", encoding="utf-8") as f:
        for r in train_split:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    with open(val_file, "w", encoding="utf-8") as f:
        for r in val_split:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    summary = {
        "total_train": len(train_split),
        "total_val": len(val_split),
        "train_category_counts": dict(Counter(r["category"] for r in train_split)),
        "train_category_percentages": {
            k: round(v / len(train_split) * 100, 2)
            for k, v in Counter(r["category"] for r in train_split).items()
        },
    }
    with open(OUT_DIR / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\nDataset Generation Complete:")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
