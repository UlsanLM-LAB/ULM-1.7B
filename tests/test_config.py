from __future__ import annotations

import pytest

from ulm.training.config import ConfigError, TrainingConfig, load_config, save_snapshot


def base_config(**overrides: object) -> dict[str, object]:
    values: dict[str, object] = {
        "model_name": "Qwen/Qwen3-0.6B",
        "dataset_path": "data/examples/sft_fixture.jsonl",
        "output_dir": "outputs/smoke",
    }
    values.update(overrides)
    return values


def test_config_defaults_and_alias(tmp_path) -> None:
    config = TrainingConfig.from_mapping(base_config(per_device_train_batch_size=1))
    assert config.batch_size == 1
    assert config.lora_r == 32
    assert config.target_modules == "all-linear"
    snapshot = save_snapshot(config, tmp_path / "run")
    assert snapshot.exists()


def test_config_yaml_load(tmp_path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(
        "model_name: Qwen/Qwen3-0.6B\n"
        "dataset_path: data/examples/sft_fixture.jsonl\n"
        "output_dir: outputs/test\n"
        "target_modules:\n  - q_proj\n  - v_proj\n",
        encoding="utf-8",
    )
    config = load_config(path)
    assert config.target_modules == ("q_proj", "v_proj")


@pytest.mark.parametrize(
    "overrides",
    [
        {"bf16": True, "fp16": True},
        {"warmup_ratio": 1.0},
        {"lora_dropout": 1.0},
        {"bnb_4bit_quant_type": "bad"},
        {"max_seq_length": 0},
        {"unknown": True},
    ],
)
def test_invalid_config_is_rejected(overrides: dict[str, object]) -> None:
    with pytest.raises(ConfigError):
        TrainingConfig.from_mapping(base_config(**overrides))


def test_snapshot_does_not_accept_different_resume_config(tmp_path) -> None:
    output = tmp_path / "run"
    first = TrainingConfig.from_mapping(base_config(output_dir=str(output)))
    save_snapshot(first, output)
    second = TrainingConfig.from_mapping(base_config(output_dir=str(output), seed=99))
    with pytest.raises(FileExistsError):
        save_snapshot(second, output, allow_existing=True)


def test_snapshot_allows_resume_flag_change(tmp_path) -> None:
    output = tmp_path / "run"
    first = TrainingConfig.from_mapping(base_config(output_dir=str(output)))
    save_snapshot(first, output)
    resumed = TrainingConfig.from_mapping(
        base_config(output_dir=str(output), resume_from_checkpoint=True)
    )
    assert save_snapshot(resumed, output, allow_existing=True) == output / "config.resolved.json"
