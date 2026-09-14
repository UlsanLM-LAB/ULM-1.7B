from __future__ import annotations

import pytest

from ulm.training.resume import ensure_output_dir, find_latest_checkpoint, resolve_resume_checkpoint


def test_latest_checkpoint_uses_numeric_order(tmp_path) -> None:
    output = tmp_path / "run"
    (output / "checkpoint-9").mkdir(parents=True)
    (output / "checkpoint-100").mkdir()
    (output / "checkpoint-old").mkdir()
    assert find_latest_checkpoint(output).name == "checkpoint-100"
    assert resolve_resume_checkpoint(True, output).name == "checkpoint-100"


def test_resume_requires_existing_checkpoint(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        resolve_resume_checkpoint(True, tmp_path / "run")


def test_new_output_does_not_overwrite(tmp_path) -> None:
    output = tmp_path / "run"
    output.mkdir()
    (output / "user-file").write_text("keep", encoding="utf-8")
    with pytest.raises(FileExistsError):
        ensure_output_dir(output, None)
    assert (output / "user-file").read_text(encoding="utf-8") == "keep"
