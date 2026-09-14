"""재현 가능한 training YAML config와 snapshot."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """training config가 유효하지 않을 때 발생하는 오류."""


_FIELDS = frozenset(
    {
        "model_name",
        "dataset_path",
        "eval_dataset_path",
        "max_seq_length",
        "batch_size",
        "per_device_train_batch_size",
        "per_device_eval_batch_size",
        "gradient_accumulation_steps",
        "learning_rate",
        "num_train_epochs",
        "max_steps",
        "warmup_ratio",
        "optimizer",
        "scheduler",
        "lora_r",
        "lora_alpha",
        "lora_dropout",
        "target_modules",
        "load_in_4bit",
        "bnb_4bit_quant_type",
        "double_quantization",
        "bf16",
        "fp16",
        "gradient_checkpointing",
        "packing",
        "save_steps",
        "eval_steps",
        "logging_steps",
        "save_total_limit",
        "seed",
        "output_dir",
        "resume_from_checkpoint",
        "report_to",
        "trust_remote_code",
    }
)


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    model_name: str
    dataset_path: str
    output_dir: str
    eval_dataset_path: str | None = None
    max_seq_length: int = 1024
    batch_size: int = 2
    per_device_eval_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    learning_rate: float = 1e-4
    num_train_epochs: float = 2.0
    max_steps: int | None = None
    warmup_ratio: float = 0.03
    optimizer: str = "adamw_torch"
    scheduler: str = "cosine"
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    target_modules: str | tuple[str, ...] = "all-linear"
    load_in_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    double_quantization: bool = True
    bf16: bool = False
    fp16: bool = True
    gradient_checkpointing: bool = True
    packing: bool = True
    save_steps: int = 250
    eval_steps: int = 250
    logging_steps: int = 10
    save_total_limit: int = 3
    seed: int = 42
    resume_from_checkpoint: bool | str | None = None
    report_to: str = "none"
    trust_remote_code: bool = False

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> TrainingConfig:
        if not isinstance(raw, Mapping):
            raise ConfigError("config 최상위 값은 mapping이어야 합니다")
        unknown = set(raw) - _FIELDS
        if unknown:
            raise ConfigError(f"알 수 없는 config key: {sorted(unknown)!r}")
        missing = {"model_name", "dataset_path", "output_dir"} - set(raw)
        if missing:
            raise ConfigError(f"필수 config key가 없습니다: {sorted(missing)!r}")
        values = dict(raw)
        if "per_device_train_batch_size" in values:
            if (
                "batch_size" in values
                and values["batch_size"] != values["per_device_train_batch_size"]
            ):
                raise ConfigError("batch_size와 per_device_train_batch_size가 서로 다릅니다")
            values["batch_size"] = values.pop("per_device_train_batch_size")
        if "target_modules" in values:
            target_modules = values["target_modules"]
            if isinstance(target_modules, list):
                values["target_modules"] = tuple(str(item) for item in target_modules)
            elif not isinstance(target_modules, str):
                raise ConfigError(
                    "target_modules는 all-linear 문자열 또는 문자열 목록이어야 합니다"
                )
        try:
            config = cls(**values)
        except TypeError as exc:
            raise ConfigError(f"config field type 또는 값이 올바르지 않습니다: {exc}") from exc
        return config.validate()

    def validate(self) -> TrainingConfig:
        errors: list[str] = []
        for field_name in ("model_name", "dataset_path", "output_dir"):
            if (
                not isinstance(getattr(self, field_name), str)
                or not getattr(self, field_name).strip()
            ):
                errors.append(f"{field_name}은(는) 비어 있을 수 없습니다")
        integer_fields = (
            "max_seq_length",
            "batch_size",
            "per_device_eval_batch_size",
            "gradient_accumulation_steps",
            "lora_r",
            "lora_alpha",
            "save_steps",
            "eval_steps",
            "logging_steps",
            "save_total_limit",
        )
        for field_name in integer_fields:
            value = getattr(self, field_name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                errors.append(f"{field_name}은(는) 양의 정수여야 합니다")
        if self.max_steps is not None and (
            not isinstance(self.max_steps, int)
            or isinstance(self.max_steps, bool)
            or self.max_steps <= 0
        ):
            errors.append("max_steps는 None 또는 양의 정수여야 합니다")
        if (
            isinstance(self.num_train_epochs, bool)
            or not isinstance(self.num_train_epochs, (int, float))
            or self.num_train_epochs <= 0
        ):
            errors.append("num_train_epochs는 양수여야 합니다")
        if (
            isinstance(self.learning_rate, bool)
            or not isinstance(self.learning_rate, (int, float))
            or self.learning_rate <= 0
        ):
            errors.append("learning_rate는 양수여야 합니다")
        if (
            isinstance(self.warmup_ratio, bool)
            or not isinstance(self.warmup_ratio, (int, float))
            or not 0 <= self.warmup_ratio < 1
        ):
            errors.append("warmup_ratio는 0 이상 1 미만이어야 합니다")
        if (
            isinstance(self.lora_dropout, bool)
            or not isinstance(self.lora_dropout, (int, float))
            or not 0 <= self.lora_dropout < 1
        ):
            errors.append("lora_dropout은 0 이상 1 미만이어야 합니다")
        if self.bnb_4bit_quant_type not in {"nf4", "fp4"}:
            errors.append("bnb_4bit_quant_type은 nf4 또는 fp4여야 합니다")
        if self.bf16 and self.fp16:
            errors.append("bf16과 fp16을 동시에 true로 설정할 수 없습니다")
        for field_name in (
            "load_in_4bit",
            "double_quantization",
            "bf16",
            "fp16",
            "gradient_checkpointing",
            "packing",
            "trust_remote_code",
        ):
            if not isinstance(getattr(self, field_name), bool):
                errors.append(f"{field_name}은(는) bool이어야 합니다")
        if not isinstance(self.report_to, str) or not self.report_to.strip():
            errors.append("report_to는 비어 있지 않은 문자열이어야 합니다")
        if not isinstance(self.optimizer, str) or not self.optimizer.strip():
            errors.append("optimizer는 비어 있지 않은 문자열이어야 합니다")
        if not isinstance(self.scheduler, str) or not self.scheduler.strip():
            errors.append("scheduler는 비어 있지 않은 문자열이어야 합니다")
        if (
            not isinstance(self.resume_from_checkpoint, (bool, str))
            and self.resume_from_checkpoint is not None
        ):
            errors.append("resume_from_checkpoint는 None, bool, 경로 문자열 중 하나여야 합니다")
        if isinstance(self.target_modules, tuple):
            if not self.target_modules or any(
                not isinstance(item, str) or not item.strip() for item in self.target_modules
            ):
                errors.append("target_modules 목록은 비어 있지 않은 문자열만 가져야 합니다")
        elif not isinstance(self.target_modules, str) or not self.target_modules.strip():
            errors.append("target_modules는 비어 있지 않은 문자열이어야 합니다")
        if errors:
            raise ConfigError("; ".join(errors))
        return self

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        result = asdict(self)
        if isinstance(self.target_modules, tuple):
            result["target_modules"] = list(self.target_modules)
        return result


def load_config(path: str | Path) -> TrainingConfig:
    source = Path(path)
    try:
        with source.open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)
    except OSError as exc:
        raise ConfigError(f"config를 읽을 수 없습니다: {source}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise ConfigError(f"YAML 문법 오류: {source}: {exc}") from exc
    return TrainingConfig.from_mapping(raw or {})


def save_snapshot(config: TrainingConfig, output_dir: str | Path) -> Path:
    """resolved config를 output에 기록하고 기존 내용과 충돌하면 실패한다."""

    destination_dir = Path(output_dir)
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / "config.resolved.json"
    snapshot_values = config.to_dict()
    # resume_from_checkpoint는 실행 시점의 재개 지시이며 모델 hyperparameter가 아니다.
    # 따라서 새 실행과 resume 실행이 동일한 output의 설정 snapshot을 공유할 수 있다.
    snapshot_values["resume_from_checkpoint"] = None
    rendered = json.dumps(snapshot_values, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if destination.exists():
        if destination.read_text(encoding="utf-8") != rendered:
            raise FileExistsError(
                f"기존 config snapshot과 다른 config로 덮어쓰지 않습니다: {destination}"
            )
        return destination
    destination.write_text(rendered, encoding="utf-8")
    return destination
