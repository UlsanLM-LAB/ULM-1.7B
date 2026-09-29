from __future__ import annotations

import pytest

from ulm.inference.prompt import build_inference_messages
from ulm.training.cpt import build_cpt_text
from ulm.training.sft import _model_dtype_kwargs, _partition_embedded_splits, build_messages


def test_inference_strength_is_explicit() -> None:
    messages = build_inference_messages("오늘 뭐 해?", dialect_strength=3)
    assert "강한 울산" in messages[0]["content"]
    with pytest.raises(ValueError):
        build_inference_messages("입력", dialect_strength=4)


def test_inference_default_remains_strength_two_without_style_conflict() -> None:
    messages = build_inference_messages("오늘 뭐 해?")
    system = messages[0]["content"]
    assert "의미 전달과 사실성을 우선" in system
    assert "울산 지역 일상 대화" in system
    assert "울산 말투를 기본으로 구사" not in system


def test_strength_zero_has_no_conflicting_default_dialect_instruction() -> None:
    from ulm.inference.prompt import build_chat_messages

    prepared = build_chat_messages(
        [{"role": "user", "content": "안녕"}],
        dialect_strength=0,
    )
    system = prepared[0]["content"]
    assert "표준 한국어로 답한다" in system
    assert "울산 말투를 기본으로 구사" not in system


def test_model_dtype_argument_has_one_supported_name() -> None:
    assert set(_model_dtype_kwargs(object())) in [{"dtype"}, {"torch_dtype"}]


def test_sft_messages_preserve_task_and_control() -> None:
    messages = build_messages(
        {
            "task": "standard_to_dialect",
            "standard_text": "어디 가?",
            "dialect_text": "어데 가노?",
            "dialect_strength": 2,
            "metadata": {},
        }
    )
    assert messages[-1]["content"] == "어데 가노?"
    assert "dialect_strength: 2" in messages[1]["content"]


def test_region_message_requires_label() -> None:
    with pytest.raises(ValueError, match="region_label"):
        build_messages(
            {
                "task": "region_classification",
                "dialect_text": "문장",
                "metadata": {},
            }
        )


def test_cpt_text_uses_available_parallel_fields() -> None:
    assert build_cpt_text({"dialect_text": "방언", "standard_text": "표준어"}) == "방언\n표준어"
    assert build_cpt_text({"text": "이미 합쳐진 text"}) == "이미 합쳐진 text"
    with pytest.raises(ValueError):
        build_cpt_text({})


class _FakeDataset:
    column_names = ["split", "text"]

    def __init__(self, rows: list[dict[str, str]]) -> None:
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def filter(self, function):
        return _FakeDataset([row for row in self.rows if function(row)])


def test_embedded_split_field_is_partitioned_without_dropping_rows() -> None:
    dataset = _partition_embedded_splits(
        {
            "train": _FakeDataset(
                [
                    {"split": "train", "text": "a"},
                    {"split": "validation", "text": "b"},
                ]
            )
        }
    )
    assert set(dataset) == {"train", "validation"}
    assert len(dataset["train"]) == 1
    assert len(dataset["validation"]) == 1


def test_embedded_split_field_rejects_unknown_values() -> None:
    with pytest.raises(ValueError, match="split field"):
        _partition_embedded_splits({"train": _FakeDataset([{"split": "unknown", "text": "a"}])})


@pytest.mark.parametrize("strength, phrase", [
    (0, "의도적으로 사용하지 않는다"),
    (1, "간헐적으로 사용한다"),
    (2, "울산 지역 일상 대화"),
    (3, "적극적으로 사용한다"),
])
def test_dialect_instruction_and_priority(strength, phrase) -> None:
    from ulm.inference.prompt import build_dialect_instruction

    instruction = build_dialect_instruction(strength)
    assert phrase in instruction
    assert "서비스의 안전 및 시스템 지시를 우선" in instruction
    assert "사용자의 명시적인 요청" in instruction
    assert "'표준어로 말해', '사투리 쓰지 마'" in instruction
    assert "의미 전달과 정보 품질을 유지" in instruction


def test_chat_prompt_preserves_system_and_history_without_accumulating() -> None:
    from copy import deepcopy

    from ulm.inference.prompt import build_chat_messages

    messages = [
        {"role": "system", "content": "상위 서비스 규칙."},
        {"role": "system", "content": "사용자가 제공한 추가 지시."},
        {"role": "user", "content": "표준어로 말해."},
        {"role": "assistant", "content": "알겠습니다."},
        {"role": "user", "content": "계속 설명해 줘."},
    ]
    original = deepcopy(messages)
    for strength in (3, 0, 1, 2):
        prepared = build_chat_messages(messages, dialect_strength=strength, system_prompt="미사용")
        assert prepared[0]["content"].startswith("상위 서비스 규칙.\n\n")
        assert prepared[0]["content"].count("[기본 응답 말투 설정]") == 1
        assert prepared[1:] == original[1:]
        assert "미사용" not in prepared[0]["content"]
        assert messages == original


@pytest.mark.parametrize("system_prompt", [None, "추가 서비스 지시."])
def test_chat_prompt_composes_default_or_system_prompt(system_prompt) -> None:
    from ulm.inference.prompt import PHASE4_SYSTEM_PROMPT, build_chat_messages

    history = [{"role": "user", "content": "안녕"}]
    prepared = build_chat_messages(history, system_prompt=system_prompt)
    assert prepared[0]["content"].startswith((system_prompt or PHASE4_SYSTEM_PROMPT) + "\n\n")
    assert prepared[1:] == history
    assert len(history) == 1


@pytest.mark.parametrize("strength", [-1, 4, None, True, "2", 1.5])
def test_dialect_instruction_rejects_invalid_levels(strength) -> None:
    from ulm.inference.prompt import build_dialect_instruction

    with pytest.raises(ValueError):
        build_dialect_instruction(strength)


@pytest.mark.parametrize("strength", [None, 0, 1, 2, 3])
def test_single_input_uses_same_style_priority_as_chat(strength):
    from ulm.inference.prompt import build_chat_messages

    assert build_inference_messages("  표준어로 말해  ", dialect_strength=strength) == (
        build_chat_messages(
            [{"role": "user", "content": "표준어로 말해"}],
            dialect_strength=2 if strength is None else strength,
        )
    )
