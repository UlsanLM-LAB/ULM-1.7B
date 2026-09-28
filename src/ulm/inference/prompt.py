"""inference prompt와 dialect strength control."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

_STRENGTH_GUIDANCE = {
    0: (
        "표준 한국어로 답한다. 울산 방언 어휘와 방언 종결어미를 의도적으로 사용하지 않는다."
    ),
    1: (
        "표준어를 기본으로 답한다. 울산 화자가 자연스럽게 섞어 쓸 정도의 가벼운 방언 어휘나 "
        "종결 표현만 간헐적으로 사용한다. 문장마다 억지로 방언을 넣지 않는다."
    ),
    2: (
        "울산 지역 일상 대화에서 자연스럽게 들릴 정도로 방언 어휘와 종결어미를 적절히 사용한다. "
        "과장하거나 모든 문장을 방언으로 채우지 않는다."
    ),
    3: (
        "강한 울산 지역 표현을 위해 방언 어휘와 특징적인 종결어미를 적극적으로 사용한다. "
        "부자연스러운 사투리 나열, 같은 어미 반복, 과장된 캐릭터 말투는 피한다."
    ),
}

PHASE4_SYSTEM_PROMPT = (
    "울산 지역어를 이해하는 대화 assistant로서 사용자의 요청에 정확하고 자연스럽게 답한다. "
    "의미 전달과 사실성을 우선한다."
)


def build_dialect_instruction(strength: int) -> str:
    """Describe a default style, subordinate to service rules and explicit user requests."""
    if type(strength) is not int or strength not in _STRENGTH_GUIDANCE:
        raise ValueError("dialect_strength는 0~3 정수여야 합니다")
    return (
        "[기본 응답 말투 설정]\n"
        "서비스의 안전 및 시스템 지시를 우선한다. 이 설정은 기본 말투만 조정하며 "
        "사용자의 명시적인 요청, 출력 형식, 말투 지정이 있으면 그 요청을 따른다. "
        "예를 들어 '표준어로 말해', '사투리 쓰지 마'라는 요청은 이 말투 설정보다 우선한다. "
        "별도 말투 요청이 없다면 기존 기본 말투 안내를 다음 선택에 맞게 조정한다. "
        "어느 강도에서도 의미 전달과 정보 품질을 유지한다.\n"
        + _STRENGTH_GUIDANCE[strength]
    )


def build_chat_messages(
    messages: Sequence[Mapping[str, str]],
    *,
    dialect_strength: int = 2,
    system_prompt: str | None = None,
) -> list[dict[str, str]]:
    """Compose the current style on copies; never add it to client conversation history."""
    prepared = [dict(message) for message in messages]
    system = next((message for message in prepared if message["role"] == "system"), None)
    if system is None:
        system = {"role": "system", "content": system_prompt or PHASE4_SYSTEM_PROMPT}
        prepared.insert(0, system)
    system["content"] += "\n\n" + build_dialect_instruction(dialect_strength)
    return prepared


def build_inference_messages(
    text: str, dialect_strength: int | None = None
) -> list[dict[str, str]]:
    if not isinstance(text, str) or not text.strip():
        raise ValueError("입력 text는 비어 있을 수 없습니다")
    strength = 2 if dialect_strength is None else dialect_strength
    if type(strength) is not int or strength not in range(4):
        raise ValueError("dialect_strength는 0~3이어야 합니다")
    return [
        {
            "role": "system",
            "content": PHASE4_SYSTEM_PROMPT + " " + _STRENGTH_GUIDANCE[strength],
        },
        {"role": "user", "content": text.strip()},
    ]
