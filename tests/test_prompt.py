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
