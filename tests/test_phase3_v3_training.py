"""Checkpoint copies must remain resumable while replacing the EBS latest copy."""

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from train_phase3_v3 import (  # noqa: E402
    V3GateCallback,
    checked_copy,
    preserved_rng_state,
    stalled_with_forgetting,
)


def make_checkpoint(path: Path, marker: str, complete: bool = True) -> Path:
    path.mkdir()
    for name in ("optimizer.pt", "trainer_state.json", "adapter_model.safetensors"):
        (path / name).write_text(marker)
    if complete:
        (path / "scheduler.pt").write_text(marker)
    return path


def test_incomplete_copy_does_not_replace_existing_checkpoint(tmp_path: Path) -> None:
    old = make_checkpoint(tmp_path / "old", "old")
    target = tmp_path / "latest"
    checked_copy(old, target)

    incomplete = make_checkpoint(tmp_path / "incomplete", "new", complete=False)
    with pytest.raises(RuntimeError, match="Incomplete checkpoint"):
        checked_copy(incomplete, target)

    assert (target / "optimizer.pt").read_text() == "old"
    assert (target / "scheduler.pt").read_text() == "old"
    assert not (tmp_path / "latest.copying").exists()


def test_latest_rotation_preserves_best_checkpoint(tmp_path: Path) -> None:
    nvme = tmp_path / "nvme"
    ebs = tmp_path / "ebs"
    reports = tmp_path / "reports"
    for path in (nvme, ebs, reports):
        path.mkdir()
    first = make_checkpoint(nvme / "checkpoint-70", "first")
    second = make_checkpoint(nvme / "checkpoint-140", "second")
    callback = V3GateCallback("base", "gate", reports, ebs, nvme, 70)

    callback._mirror(first, 70, "latest")
    callback._mirror(first, 70, "best")
    callback._mirror(second, 140, "latest")

    assert not (ebs / "latest-checkpoint-70").exists()
    assert (ebs / "latest-checkpoint-140" / "optimizer.pt").read_text() == "second"
    assert (ebs / "best-checkpoint-70" / "optimizer.pt").read_text() == "first"


def test_gate_sampling_does_not_change_training_rng() -> None:
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state()

    with preserved_rng_state():
        random.random()
        np.random.rand()
        torch.rand(1)

    assert random.getstate() == python_state
    restored_numpy = np.random.get_state()
    assert restored_numpy[0] == numpy_state[0]
    assert np.array_equal(restored_numpy[1], numpy_state[1])
    assert restored_numpy[2:] == numpy_state[2:]
    assert torch.equal(torch.get_rng_state(), torch_state)


def test_first_gate_compares_against_base_quality() -> None:
    base = {
        "factual_qa": {"accuracy_pct": 95.0},
        "multi_turn": {"accuracy_pct": 90.0},
        "dialect_eval": {"accuracy_pct": 20.0},
    }
    first_gate = {
        "factual_qa": {"accuracy_pct": 70.0},
        "multi_turn": {"accuracy_pct": 50.0},
        "dialect_eval": {"accuracy_pct": 20.0},
    }
    assert stalled_with_forgetting(first_gate, base)
    first_gate["dialect_eval"]["accuracy_pct"] = 25.0
    assert not stalled_with_forgetting(first_gate, base)
