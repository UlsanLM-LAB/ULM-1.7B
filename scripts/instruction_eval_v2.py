"""Conservative, dependency-free instruction rubrics; legacy scores stay intact."""
from __future__ import annotations

import json
import re

# Require explicit rejection, not an unrelated occurrence of '없' or '가공'.
REJECTION = re.compile(
    r"사실(?:이|로)?\s*(?:아니|아닙|아닌|기록되어\s*있지\s*않)|"
    r"(?:역사적\s*)?사실과\s*다(?:릅|르)|"
    r"(?:기록|근거|증거|사건|사례|항목|연표|일|기술|발명|방문한\s*적)[^.!?\n]{0,28}(?:없|존재하지\s*않)|"
    r"(?:존재|발명|개발|건설|선출|방문|보급|일어나|발생)[^.!?\n]{0,12}(?:하지|되지|된\s*적이|한\s*적이)[^.!?\n]{0,8}(?:않|없)|"
    r"발견된\s*(?:바|적)(?:가|이)?\s*없|"
    r"^(?:그런|그러한)\s*(?:건|것)(?:은|이)?\s*(?:없|아닙|아니)|"
    r"(?:허구|가상의\s*(?:사건|이야기)|시대착오|시대적으로\s*성립할\s*수\s*없)|"
    r"(?:전제|내용|주장|사건|기록|전략|배경)[^.!?\n]{0,24}(?:잘못|틀렸|아닙|성립하지)|"
    r"(?:해당|그)\s*시대에는[^.!?\n]{0,30}(?:없|않)|"
    r"검증[^.!?\n]{0,32}(?:확인되지|입증되지)\s*않|"
    r"(?:물리적으로|시대적으로)\s*불가능|"
    r"전략이\s*아니라|기록되어\s*있지\s*않|방문한\s*적이\s*없"
)
ACCEPTANCE = re.compile(r"^(?:네|예|맞습니다|맞다|사실입니다)[,.!\s]|실제로\s*(?:개발했|건설했|선출되었|방문했)|검증된\s*과학\s*장치로\s*[,，]")
FABRICATION = re.compile(r"(?:건설|개발|발명|방문|선출)(?:했|하였|되었)(?:습니다|다)(?=[\s,.!?;:]|$)|건설했다고\s*추정")
NEGATED_REJECTION = re.compile(r"(?:허구|가상)(?:가|이)?\s*아니라|없다는\s*(?:것|주장)[^.!?\n]{0,15}(?:아니|틀렸)")
REJECTION_LOOP = re.compile(r"(.{6,100}?)\1{2,}", re.DOTALL)


def rejection_ok(text: str) -> bool:
    text = text.strip()
    if not text or len(text) > 700 or ACCEPTANCE.search(text) or NEGATED_REJECTION.search(text) or REJECTION_LOOP.search(text):
        return False
    # A denial must come before an explanatory factual correction.
    first = re.split(r"[.!?\n]", text, maxsplit=1)[0]
    return bool(REJECTION.search(first)) and not bool(FABRICATION.search(text))


def instruction_ok(item: dict, text: str) -> bool:
    """Strict regression output contracts, with semantic rejection variants."""
    norm = text.strip()
    kind = item.get("instruction_type", "")
    exact = {
        "exact_apple": "사과", "exact_banana": "바나나",
        "repeat_hello_3": "안녕하세요\n안녕하세요\n안녕하세요",
        "repeat_thanks_2": "감사합니다\n감사합니다",
        "extract_animals": "고양이, 참새", "extract_dog_rabbit": "강아지, 토끼",
        "exact_three_fruits": "apple, banana, orange",
        "exact_three_animals": "cat, dog, bird", "yes_no": "네",
        "numbered_list_1_5": "1.\n2.\n3.\n4.\n5.",
        "numbered_list_1_4": "1.\n2.\n3.\n4.",
    }
    if kind.startswith("hallucination_trap_"):
        # An explicit premise denial with new false historical details is still a failure.
        # Chronology source: https://encykorea.aks.ac.kr/Article/E0000954 (948–1031).
        if kind == 'hallucination_trap_wifi' and re.search(r'강감찬[^.!?\n]{0,45}(?:12\s*[~～\-]\s*13|12|13)\s*세기[^.!?\n]{0,40}(?:장군|활동|인물)', norm):
            return False
        return rejection_ok(norm)
    if kind in ("exact_json_status", "exact_json_pass"):
        target = {"status": "ok"} if kind == "exact_json_status" else {"result": "pass"}
        try:
            return "\n" not in norm and json.loads(norm) == target
        except (ValueError, TypeError):
            return False
    if kind in ("extract_animals", "extract_dog_rabbit", "exact_three_fruits", "exact_three_animals"):
        return [x.strip() for x in norm.split(",")] == exact[kind].split(", ")
    return norm == exact.get(kind)


def synthetic_ok(row: dict, text: str) -> bool:
    target = row["messages"][-1]["content"]
    if row["task"] == "false_premise":
        return rejection_ok(text)
    if row["task"] == "json":
        try:
            return "```" not in text and json.loads(text) == json.loads(target)
        except (ValueError, TypeError):
            return False
    return text.strip() == target
