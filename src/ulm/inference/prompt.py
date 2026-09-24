"""inference prompt와 dialect strength control."""

from __future__ import annotations

_STRENGTH_GUIDANCE = {
    0: "표준어를 사용한다.",
    1: "의미를 보존하고 약한 울산 지역색만 사용한다.",
    2: "의미를 보존하는 자연스러운 일상 울산 지역어를 사용한다.",
    3: "의미를 보존하고 원어민 검수를 거친 강한 울산 지역 표현을 사용한다.",
}

PHASE4_SYSTEM_PROMPT = (
    "울산 지역어 대화 assistant로서 자연스럽고 일상적인 울산 사투리로 상대방과 친근하게 대화한다. "
    "자연스럽고 편안한 일상 울산 말투를 기본으로 구사한다."
)


def build_inference_messages(
    text: str, dialect_strength: int | None = None
) -> list[dict[str, str]]:
    if not isinstance(text, str) or not text.strip():
        raise ValueError("입력 text는 비어 있을 수 없습니다")
    if dialect_strength is not None and dialect_strength not in range(4):
        raise ValueError("dialect_strength는 0~3이어야 합니다")
    return [
        {
            "role": "system",
            "content": (
                PHASE4_SYSTEM_PROMPT
                if dialect_strength is None
                else "울산 지역어 대화 assistant로서 의미와 사실성을 우선한다. "
                f"{_STRENGTH_GUIDANCE[dialect_strength]}"
            ),
        },
        {"role": "user", "content": text.strip()},
    ]
