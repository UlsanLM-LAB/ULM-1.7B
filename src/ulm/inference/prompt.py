"""Inference prompt and prompt-based dialect strength control."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

_STRENGTH_GUIDANCE = {
    0: (
        "별도 말투 요청이 없으면 이번 응답은 표준 한국어로 답하고 "
        "울산 방언 표현을 사용하지 않는다."
    ),
    1: (
        "별도 말투 요청이 없으면 기본 울산 말투보다 지역 표현을 약하게 사용한다."
    ),
    2: "",
    3: (
        "별도 말투 요청이 없으면 기본 울산 말투보다 문맥에 맞는 지역 어휘와 "
        "종결어미를 조금 더 적극적으로 사용한다. 같은 어미를 반복하거나 과장하지 않는다."
    ),
}

PHASE4_SYSTEM_PROMPT = (
    "울산 지역어 대화 assistant로서 자연스럽고 일상적인 울산 사투리로 상대방과 친근하게 대화한다. "
    "자연스럽고 편안한 일상 울산 말투를 기본으로 구사한다. 의미와 사실을 정확히 유지한다. "
    "사용자가 말투나 출력 형식을 직접 지정하면 그 요청을 우선한다."
)


def build_dialect_instruction(strength: int) -> str:
    """Return the smallest style override needed for a dialect level."""
    if type(strength) is not int or strength not in _STRENGTH_GUIDANCE:
        raise ValueError("dialect_strength는 0~3 정수여야 합니다")
    return _STRENGTH_GUIDANCE[strength]


def build_chat_messages(
    messages: Sequence[Mapping[str, str]],
    *,
    dialect_strength: int = 2,
    system_prompt: str | None = None,
) -> list[dict[str, str]]:
    """Compose style instructions on copies; never mutate client history."""
    prepared = [dict(message) for message in messages]
    system = next((message for message in prepared if message["role"] == "system"), None)
    if system is None:
        system = {"role": "system", "content": system_prompt or PHASE4_SYSTEM_PROMPT}
        prepared.insert(0, system)

    instruction = build_dialect_instruction(dialect_strength)
    if instruction:
        system["content"] += "\n\n[기본 응답 말투 설정]\n" + instruction
    return prepared


def build_inference_messages(
    text: str, dialect_strength: int | None = None
) -> list[dict[str, str]]:
    if not isinstance(text, str) or not text.strip():
        raise ValueError("입력 text는 비어 있을 수 없습니다")
    strength = 2 if dialect_strength is None else dialect_strength
    return build_chat_messages(
        [{"role": "user", "content": text.strip()}], dialect_strength=strength
    )
