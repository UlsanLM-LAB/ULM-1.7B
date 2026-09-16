"""TRL `SFTTrainer` 기반 QLoRA adapter training.

ML dependency는 실제 training 함수가 호출될 때만 import한다. 따라서 CPU-only 환경에서도
config와 message 변환을 검증할 수 있다.
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

from ulm.data.schema import DatasetRecord

from .config import TrainingConfig, load_config, save_snapshot
from .resume import ensure_output_dir, resolve_resume_checkpoint

_SYSTEM_PROMPTS = {
    "dialect_to_standard": "울산 지역어 문장의 의미를 보존하면서 자연스러운 표준어로 바꾼다.",
    "standard_to_dialect": "의미를 보존하면서 요청된 강도의 울산 지역어로 표현한다.",
    "dialect_understanding": "울산 지역어의 의미를 정확히 이해하고 표준어로 설명한다.",
    "dialect_chat": "울산 지역어 대화에 자연스럽고 의미가 정확하게 답한다.",
    "region_classification": "입력 문장의 지역 분류를 주어진 label로 답한다.",
    "dialect_strength_control": (
        "의미를 보존하면서 요청된 dialect_strength의 울산 지역어로 표현한다."
    ),
}


def _value(record: DatasetRecord | Mapping[str, Any], key: str) -> Any:
    return getattr(record, key) if isinstance(record, DatasetRecord) else record.get(key)


def build_messages(record: DatasetRecord | Mapping[str, Any]) -> list[dict[str, str]]:
    """canonical record를 TRL conversational messages로 변환한다."""

    task = _value(record, "task")
    dialect_text = _value(record, "dialect_text")
    standard_text = _value(record, "standard_text")
    strength = _value(record, "dialect_strength")
    metadata = _value(record, "metadata") or {}
    if task not in _SYSTEM_PROMPTS:
        raise ValueError(f"지원하지 않는 SFT task: {task!r}")
    system = _SYSTEM_PROMPTS[task]
    if task == "dialect_to_standard":
        user = f"다음 문장을 표준어로 바꿔라.\n입력: {dialect_text}"
        assistant = standard_text
    elif task == "standard_to_dialect":
        control = f"\ndialect_strength: {strength}" if strength is not None else ""
        user = f"다음 문장을 울산 지역어로 바꿔라.{control}\n입력: {standard_text}"
        assistant = dialect_text
    elif task == "dialect_understanding":
        user = f"다음 울산 지역어의 뜻을 표준어로 설명해라.\n입력: {dialect_text}"
        assistant = standard_text or dialect_text
    elif task == "dialect_chat":
        user = f"다음 발화에 자연스럽게 답해라.\n발화: {dialect_text or standard_text}"
        assistant = standard_text or dialect_text
    elif task == "region_classification":
        region = metadata.get("region_label") if isinstance(metadata, Mapping) else None
        if not isinstance(region, str) or not region.strip():
            raise ValueError("region_classification record에는 metadata.region_label이 필요합니다")
        user = f"다음 문장의 지역 label을 답해라.\n문장: {dialect_text}"
        assistant = region
    else:
        control = f"dialect_strength: {strength}" if strength is not None else "dialect_strength: 2"
        user = f"{control}\n표준어 문장: {standard_text}"
        assistant = dialect_text
    if not isinstance(assistant, str) or not assistant.strip():
        raise ValueError(f"task={task!r}에 assistant text가 없습니다")
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
        {"role": "assistant", "content": assistant},
    ]


def _require_ml_dependencies() -> tuple[Any, ...]:
    try:
        import torch
        from datasets import load_dataset
        from peft import LoraConfig, prepare_model_for_kbit_training
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from trl import SFTConfig, SFTTrainer
    except ImportError as exc:  # pragma: no cover - depends on runtime extras
        raise RuntimeError(
            "training에는 ML extra가 필요합니다. `uv sync --extra dev --extra ml`을 실행하세요."
        ) from exc
    return (
        torch,
        load_dataset,
        LoraConfig,
        prepare_model_for_kbit_training,
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        SFTConfig,
        SFTTrainer,
    )


def _dataset_files(path: Path) -> dict[str, str]:
    if path.is_file():
        return {"train": str(path)}
    if not path.is_dir():
        raise FileNotFoundError(f"dataset path가 없습니다: {path}")
    files: dict[str, str] = {}
    for split, names in {
        "train": ("train.jsonl", "train.json"),
        "validation": ("validation.jsonl", "validation.json", "dev.jsonl", "dev.json"),
        "test": ("test.jsonl", "test.json"),
    }.items():
        for name in names:
            candidate = path / name
            if candidate.is_file():
                files[split] = str(candidate)
                break
    if not files:
        raise FileNotFoundError(f"dataset directory에 split 파일이 없습니다: {path}")
    return files


def _prepare_dataset(dataset: Any) -> Any:
    def convert(row: Mapping[str, Any]) -> dict[str, Any]:
        if isinstance(row.get("messages"), list):
            return {"messages": row["messages"]}
        return {"messages": build_messages(row)}

    return dataset.map(convert, remove_columns=dataset.column_names)


def _filtered_kwargs(constructor: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    parameters = inspect.signature(constructor).parameters
    return {key: value for key, value in kwargs.items() if key in parameters}


def _model_dtype_kwargs(compute_dtype: Any) -> dict[str, Any]:
    """Transformers 5.x와 4.x의 model dtype 인자 차이를 흡수한다."""

    try:
        import transformers

        major_version = int(transformers.__version__.split(".", 1)[0])
    except (ImportError, ValueError, AttributeError):
        major_version = 4
    key = "dtype" if major_version >= 5 else "torch_dtype"
    return {key: compute_dtype}


def _restore_fp16_trainable_parameters(model: Any, bfloat16_dtype: Any) -> None:
    """일부 TRL가 QLoRA adapter를 BF16으로 바꾼 뒤 FP16 scaler와 충돌하는 것을 막는다."""

    for parameter in model.parameters():
        if parameter.requires_grad and parameter.dtype == bfloat16_dtype:
            parameter.data = parameter.data.float()


def _partition_embedded_splits(dataset: Any) -> dict[str, Any]:
    """한 JSONL에 들어 있는 split field를 DatasetDict처럼 분리한다."""

    if "validation" in dataset or "test" in dataset:
        return dict(dataset)
    train_dataset = dataset["train"]
    if "split" not in train_dataset.column_names:
        return dict(dataset)
    split_values = train_dataset.filter(
        lambda row: row.get("split") in {"train", "validation", "test"}
    )
    if len(split_values) != len(train_dataset):
        raise ValueError(
            "single JSONL의 split field에는 train/validation/test만 사용할 수 있습니다"
        )
    if len(split_values) == 0:
        return dict(dataset)
    result: dict[str, Any] = {}
    for split in ("train", "validation", "test"):
        subset = split_values.filter(lambda row, expected=split: row.get("split") == expected)
        if len(subset) > 0:
            result[split] = subset
    return result


def _load_dataset(load_dataset: Any, config: TrainingConfig) -> dict[str, Any]:
    data_files = _dataset_files(Path(config.dataset_path))
    if config.eval_dataset_path:
        evaluation_files = _dataset_files(Path(config.eval_dataset_path))
        data_files["validation"] = evaluation_files.get("validation", evaluation_files["train"])
    dataset = _partition_embedded_splits(load_dataset("json", data_files=data_files))
    if "train" not in dataset or len(dataset["train"]) == 0:
        raise ValueError("training dataset에는 비어 있지 않은 train split이 필요합니다")
    return dataset


def run_sft(config: TrainingConfig, *, enable_dashboard: bool = False) -> Path:
    config.validate()
    (
        torch,
        load_dataset,
        LoraConfig,
        prepare_model_for_kbit_training,
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        SFTConfig,
        SFTTrainer,
    ) = _require_ml_dependencies()
    resume_checkpoint = resolve_resume_checkpoint(config.resume_from_checkpoint, config.output_dir)
    if config.load_in_4bit and not torch.cuda.is_available():
        raise RuntimeError("4-bit QLoRA는 현재 pipeline에서 CUDA GPU 실행만 허용합니다")
    if config.bf16 and not torch.cuda.is_bf16_supported():
        raise RuntimeError("config가 bf16을 요구하지만 현재 CUDA가 bf16을 지원하지 않습니다")
    output_dir = ensure_output_dir(config.output_dir, resume_checkpoint)
    save_snapshot(config, output_dir)

    dataset = _load_dataset(load_dataset, config)
    if "validation" in dataset and len(dataset["validation"]) > 1000:
        dataset["validation"] = dataset["validation"].select(range(1000))
    dataset = {split: _prepare_dataset(value) for split, value in dataset.items()}
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, use_fast=True)
    compute_dtype = torch.bfloat16 if config.bf16 else torch.float16
    quantization_config = None
    if config.load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=config.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=config.double_quantization,
            bnb_4bit_compute_dtype=compute_dtype,
        )
    model_kwargs: dict[str, Any] = {"device_map": "auto", **_model_dtype_kwargs(compute_dtype)}
    if quantization_config is not None:
        model_kwargs["quantization_config"] = quantization_config
    if config.trust_remote_code:
        model_kwargs["trust_remote_code"] = True
    model = AutoModelForCausalLM.from_pretrained(config.model_name, **model_kwargs)
    model.config.use_cache = False
    if config.load_in_4bit:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=config.gradient_checkpointing
        )
    target_modules: str | list[str]
    target_modules = (
        config.target_modules
        if isinstance(config.target_modules, str)
        else list(config.target_modules)
    )
    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )

    training_kwargs: dict[str, Any] = {
        "output_dir": str(output_dir),
        "max_length": config.max_seq_length,
        "max_seq_length": config.max_seq_length,
        "per_device_train_batch_size": config.batch_size,
        "per_device_eval_batch_size": config.per_device_eval_batch_size,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "learning_rate": config.learning_rate,
        "num_train_epochs": config.num_train_epochs,
        "max_steps": config.max_steps if config.max_steps is not None else -1,
        "warmup_ratio": config.warmup_ratio,
        "optim": config.optimizer,
        "lr_scheduler_type": config.scheduler,
        "weight_decay": 0.01,
        "logging_steps": config.logging_steps,
        "save_steps": config.save_steps,
        "eval_steps": config.eval_steps,
        "eval_strategy": "steps" if "validation" in dataset else "no",
        "evaluation_strategy": "steps" if "validation" in dataset else "no",
        "save_total_limit": config.save_total_limit,
        "gradient_checkpointing": config.gradient_checkpointing,
        "packing": config.packing,
        "packing_strategy": "wrapped",
        "eval_packing": False,
        "padding_free": False,
        "bf16": config.bf16,
        "fp16": config.fp16,
        "report_to": config.report_to,
        "seed": config.seed,
    }
    training_args = SFTConfig(**_filtered_kwargs(SFTConfig, training_kwargs))
    trainer_kwargs: dict[str, Any] = {
        "model": model,
        "args": training_args,
        "train_dataset": dataset["train"],
        "eval_dataset": dataset.get("validation"),
        "peft_config": lora_config,
        "processing_class": tokenizer,
        "tokenizer": tokenizer,
    }
    trainer = SFTTrainer(**_filtered_kwargs(SFTTrainer, trainer_kwargs))
    if config.fp16 and config.load_in_4bit:
        _restore_fp16_trainable_parameters(trainer.model, torch.bfloat16)

    if enable_dashboard:
        try:
            from ulm.utils.dashboard import RichDashboardCallback

            trainer.add_callback(
                RichDashboardCallback(
                    max_epochs=config.num_train_epochs, model_name=config.model_name
                )
            )
        except Exception:
            pass

    trainer.train(resume_from_checkpoint=str(resume_checkpoint) if resume_checkpoint else None)
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    metadata = {
        "model_name": config.model_name,
        "dataset_path": config.dataset_path,
        "seed": config.seed,
        "resume_checkpoint": str(resume_checkpoint) if resume_checkpoint else None,
        "torch_version": getattr(torch, "__version__", "unknown"),
        "cuda_available": bool(torch.cuda.is_available()),
    }
    (output_dir / "run.metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return output_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="canonical JSONL을 QLoRA SFT adapter로 학습합니다."
    )
    parser.add_argument("--config", required=True, type=Path, help="training YAML config")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="ML package와 GPU를 사용하지 않고 config만 검증합니다",
    )
    parser.add_argument(
        "--dashboard",
        action="store_true",
        help="터미널 실시간 Rich TUI 대시보드를 활성화합니다",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    if args.dry_run:
        print(json.dumps(config.to_dict(), ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    output_dir = run_sft(config, enable_dashboard=args.dashboard)
    print(f"training output: {output_dir}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
