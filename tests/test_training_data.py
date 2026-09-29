from __future__ import annotations

import json
from pathlib import Path

import pytest

from ulm.training.config import TrainingConfig
from ulm.training.sft import _load_dataset

datasets = pytest.importorskip("datasets")


def write_rows(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def load(tmp_path: Path, train: Path, evaluation: Path | None = None) -> dict:
    config = TrainingConfig(
        model_name="unused",
        dataset_path=str(train),
        eval_dataset_path=str(evaluation) if evaluation is not None else None,
        output_dir=str(tmp_path / "output"),
    )

    def json_loader(name, **kwargs):
        return datasets.load_dataset(name, cache_dir=str(tmp_path / "cache"), **kwargs)

    return _load_dataset(json_loader, config)


def test_external_validation_keeps_embedded_test_out_of_training(tmp_path):
    train = write_rows(
        tmp_path / "mixed.jsonl",
        [{"split": split, "id": split} for split in ("train", "validation", "test")],
    )
    evaluation = write_rows(tmp_path / "eval.jsonl", [{"id": "external"}])
    result = load(tmp_path, train, evaluation)
    assert result["train"]["id"] == ["train"]
    assert result["validation"]["id"] == ["external"]
    assert result["test"]["id"] == ["test"]


@pytest.mark.parametrize("filename", ["validation.jsonl", "dev.jsonl", "train.jsonl"])
def test_evaluation_only_directory(tmp_path, filename):
    train = write_rows(tmp_path / "train.jsonl", [{"id": "train"}])
    write_rows(tmp_path / "eval" / filename, [{"id": "eval"}])
    result = load(tmp_path, train, tmp_path / "eval")
    assert result["validation"]["id"] == ["eval"]


def test_single_evaluation_file_with_embedded_splits_uses_validation(tmp_path):
    train = write_rows(tmp_path / "train.jsonl", [{"id": "train"}])
    evaluation = write_rows(
        tmp_path / "evaluation.jsonl",
        [{"id": split, "split": split} for split in ("train", "validation", "test")],
    )
    result = load(tmp_path, train, evaluation)
    assert result["validation"]["id"] == ["validation"]


def test_named_validation_does_not_bypass_embedded_train_partition(tmp_path):
    write_rows(
        tmp_path / "data" / "train.jsonl",
        [{"split": split, "id": split} for split in ("train", "validation", "test")],
    )
    write_rows(tmp_path / "data" / "validation.jsonl", [{"id": "explicit"}])
    result = load(tmp_path, tmp_path / "data")
    assert result["train"]["id"] == ["train"]
    assert result["validation"]["id"] == ["explicit"]
    assert result["test"]["id"] == ["test"]


@pytest.mark.parametrize("field", ["id", "speaker_id"])
def test_shared_canonical_id_or_speaker_is_rejected(tmp_path, field):
    train = write_rows(tmp_path / "train.jsonl", [{field: "shared"}])
    evaluation = write_rows(tmp_path / "eval.jsonl", [{field: "shared"}])
    with pytest.raises(ValueError, match=f"{field} leakage"):
        load(tmp_path, train, evaluation)


def test_multiple_utterances_from_one_training_speaker_are_allowed(tmp_path):
    train = write_rows(
        tmp_path / "train.jsonl",
        [{"id": f"train-{i}", "speaker_id": "train-speaker"} for i in range(2)],
    )
    evaluation = write_rows(
        tmp_path / "eval.jsonl", [{"id": "eval", "speaker_id": "eval-speaker"}]
    )
    result = load(tmp_path, train, evaluation)
    assert len(result["train"]) == 2


def test_invalid_embedded_split_is_rejected_even_with_named_validation(tmp_path):
    write_rows(tmp_path / "data" / "train.jsonl", [{"split": "invalid", "id": "train"}])
    write_rows(tmp_path / "data" / "validation.jsonl", [{"id": "eval"}])
    with pytest.raises(ValueError, match="split field"):
        load(tmp_path, tmp_path / "data")


def test_missing_training_split_is_rejected(tmp_path):
    write_rows(tmp_path / "data" / "validation.jsonl", [{"id": "eval"}])
    with pytest.raises(ValueError, match="train split"):
        load(tmp_path, tmp_path / "data")


def test_evaluation_directory_with_only_test_file_is_rejected(tmp_path):
    train = write_rows(tmp_path / "train.jsonl", [{"id": "train"}])
    write_rows(tmp_path / "eval" / "test.jsonl", [{"id": "test"}])
    with pytest.raises(ValueError, match="validation.*train"):
        load(tmp_path, train, tmp_path / "eval")


def test_evaluation_directory_without_split_files_is_rejected(tmp_path):
    train = write_rows(tmp_path / "train.jsonl", [{"id": "train"}])
    evaluation = tmp_path / "invalid-eval"
    evaluation.mkdir()
    with pytest.raises(FileNotFoundError, match="split 파일"):
        load(tmp_path, train, evaluation)


def test_same_unsplit_file_without_ids_cannot_be_training_and_validation(tmp_path):
    train = write_rows(tmp_path / "train.jsonl", [{"text": "no identity columns"}])
    alias = tmp_path / "evaluation.jsonl"
    alias.symlink_to(train)
    with pytest.raises(ValueError, match="train/test 파일"):
        load(tmp_path, train, alias)


def test_same_embedded_file_can_supply_disjoint_splits_without_ids(tmp_path):
    train = write_rows(
        tmp_path / "mixed.jsonl",
        [{"split": split, "text": split} for split in ("train", "validation", "test")],
    )
    result = load(tmp_path, train, train)
    assert result["train"]["text"] == ["train"]
    assert result["validation"]["text"] == ["validation"]
    assert result["test"]["text"] == ["test"]
