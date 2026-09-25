"""Build a diverse, benchmark-disjoint recovery mix from existing corpora."""

import gzip
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path("data/ulsan_dialect_phase3_v3_recovery_v2")
RNG = random.Random(4202)
MARKERS = re.compile(r"(아이가|맞제|맞나|묵|하이소|카이|카더|카는|그라|그랬나|어데|우째|와 그라|기라|데이|했제|있제|하제|뿌|쌔|마이)")
NOISE = re.compile(r"\{[^}]*\}|#[^#]*#|&[^&]*&|<[^>]*>")


def read(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


def key(text):
    return re.sub(r"\s+", "", text).strip().lower()


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    benchmarks = [Path("reports/phase3-v3/gate-40.jsonl"), Path("data/regression_benchmark_150.jsonl"), Path("reports/phase3-v3/final-dialect-holdout-50.jsonl")]
    forbidden = set()
    for path in benchmarks:
        if not path.exists():
            raise FileNotFoundError(path)
        forbidden.update(key(r["prompt"]) for r in read(path))

    groups = defaultdict(list)
    seen = set()

    def add(category, messages):
        if len(messages) < 2 or messages[-1]["role"] != "assistant":
            return
        prompt, answer = messages[-2]["content"], messages[-1]["content"]
        if not prompt.strip() or not answer.strip() or key(prompt) in forbidden or NOISE.search(prompt + answer):
            return
        identity = (key(prompt), key(answer))
        if identity in seen:
            return
        seen.add(identity)
        groups[category].append({"task": "recovery_v2", "category": category, "messages": messages})

    dense = list(read("data/ulsan_dialect_dense/train.jsonl.gz"))
    RNG.shuffle(dense)
    strong = {}
    for row in dense:
        standard, dialect = row.get("standard_text", ""), row.get("dialect_text", "")
        if (row.get("task") == "standard_to_dialect" and row.get("quality_grade") == "A"
                and row.get("synthetic") is False and row.get("metadata", {}).get("ulsan_tier") in ("U0", "U1")
                and len(standard) >= 15 and len(dialect) >= 15 and key(standard) != key(dialect)
                and MARKERS.search(dialect) and not NOISE.search(standard + dialect)):
            strong.setdefault(key(dialect), (standard, dialect))
    for standard, dialect in strong.values():
        for prompt in (f"이 말을 울산 말투로 자연스럽게 바꿔줘: {standard}",
                       f"울산 사투리로 말하면 어떻게 돼? {standard}"):
            add("dialect_sft", [{"role": "user", "content": prompt}, {"role": "assistant", "content": dialect}])

    v1 = list(read("data/ulsan_dialect_phase3/train.jsonl"))
    RNG.shuffle(v1)
    natural_answers = set()
    casual = []
    for row in v1:
        messages = row.get("messages", [])
        if len(messages) < 2:
            continue
        answer = messages[-1]["content"]
        if row.get("category") == "aihub_casual_turn":
            if len(answer) >= 25 and not NOISE.search(answer + messages[-2]["content"]):
                casual.append(messages)
        elif MARKERS.search(answer) and len(answer) >= 25 and key(answer) not in natural_answers:
            natural_answers.add(key(answer))
            if len(groups["dialect_sft"]) < 3100:
                add("dialect_sft", messages)

    replay = list(read("data/ulsan_dialect_phase3_v2/train.jsonl")) + list(read("data/ulsan_dialect_phase3_v2/validation.jsonl"))
    RNG.shuffle(replay)
    prompt_counts = Counter()
    answer_counts = Counter()
    for row in replay:
        category = row.get("category")
        if category not in ("factual_preservation", "instruction_preservation"):
            continue
        messages = row["messages"]
        prompt, answer = messages[-2]["content"], messages[-1]["content"]
        prompt_limit = 25 if category == "instruction_preservation" else 3
        if prompt_counts[(category, key(prompt))] >= prompt_limit or answer_counts[(category, key(answer))] >= 4:
            continue
        before = len(groups[category])
        add(category, messages)
        if len(groups[category]) > before:
            prompt_counts[(category, key(prompt))] += 1
            answer_counts[(category, key(answer))] += 1
    for messages in casual:
        if len(groups["factual_preservation"]) >= 1000:
            break
        add("factual_preservation", messages)

    # Preserve an explicit replay floor; the original v2 set includes multi-turn memory.
    if len(groups["dialect_sft"]) < 1600 or len(groups["factual_preservation"]) < 550 or len(groups["instruction_preservation"]) < 100:
        raise RuntimeError(f"insufficient diverse data: {dict((k, len(v)) for k, v in groups.items())}")
    selected = (groups["dialect_sft"][:2800] + groups["factual_preservation"][:1000]
                + groups["instruction_preservation"][:400])
    RNG.shuffle(selected)
    cut = int(len(selected) * .95)
    for name, rows in (("train", selected[:cut]), ("validation", selected[cut:])):
        with (ROOT / f"{name}.jsonl").open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {"total": len(selected), "train": cut, "validation": len(selected) - cut,
               "counts": dict(Counter(r["category"] for r in selected)), "dense_unique": len(strong),
               "benchmark_prompt_overlap": 0}
    (ROOT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
