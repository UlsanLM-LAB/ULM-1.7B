"""causal language modeling continued pretraining adapter."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .config import TrainingConfig, load_config, save_snapshot
from .resume import ensure_output_dir, resolve_resume_checkpoint
from .sft import _dataset_files, _filtered_kwargs


def build_cpt_text(row: Mapping[str, Any]) -> str:
    """canonical row에서 next-token prediction용 text를 만든다."""

    if isinstance(row.get("text"), str) and row["text"].strip():
        return row["text"].strip()
    parts = [
        value.strip()
        for value in (row.get("dialect_text"), row.get("standard_text"))
        if isinstance(value, str) and value.strip()
    ]
    if not parts:
        raise ValueError("CPT row에는 text 또는 dialect_text/standard_text가 필요합니다")
    return "\n".join(parts)


def _require_ml_dependencies() -> tuple[Any, ...]:
    try:
        import torch
        from datasets import load_dataset
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
            DataCollatorForLanguageModeling,
            Trainer,
            TrainingArguments,
        )
    except ImportError as exc:  # pragma: no cover - depends on runtime extras
        raise RuntimeError(
            "CPT에는 ML extra가 필요합니다. `uv sync --extra dev --extra ml`을 실행하세요."
        ) from exc
    return (
        torch,
        load_dataset,
        LoraConfig,
        get_peft_model,
        prepare_model_for_kbit_training,
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        DataCollatorForLanguageModeling,
        Trainer,
        TrainingArguments,
    )


def run_cpt(config: TrainingConfig) -> Path:
    config.validate()
    resume_checkpoint = resolve_resume_checkpoint(config.resume_from_checkpoint, config.output_dir)
    output_dir = ensure_output_dir(config.output_dir, resume_checkpoint)
    save_snapshot(config, output_dir)
    (
        torch,
        load_dataset,
        LoraConfig,
        get_peft_model,
        prepare_model_for_kbit_training,
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        DataCollatorForLanguageModeling,
        Trainer,
        TrainingArguments,
    ) = _require_ml_dependencies()
    if config.load_in_4bit and not torch.cuda.is_available():
        raise RuntimeError("4-bit CPT는 현재 pipeline에서 CUDA GPU 실행만 허용합니다")
    if config.bf16 and not torch.cuda.is_bf16_supported():
        raise RuntimeError("config가 bf16을 요구하지만 현재 CUDA가 bf16을 지원하지 않습니다")

    dataset = load_dataset("json", data_files=_dataset_files(Path(config.dataset_path)))
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    compute_dtype = torch.bfloat16 if config.bf16 else torch.float16
    quantization_config = None
    if config.load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=config.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=config.double_quantization,
            bnb_4bit_compute_dtype=compute_dtype,
        )
    model_kwargs: dict[str, Any] = {"device_map": "auto", "torch_dtype": compute_dtype}
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
    model = get_peft_model(model, lora_config)

    def tokenize(row: Mapping[str, Any]) -> Mapping[str, Any]:
        return tokenizer(build_cpt_text(row), truncation=True, max_length=config.max_seq_length)

    tokenized = {
        split: values.map(tokenize, remove_columns=values.column_names)
        for split, values in dataset.items()
    }
    training_kwargs: dict[str, Any] = {
        "output_dir": str(output_dir),
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
        "eval_strategy": "steps" if "validation" in tokenized else "no",
        "evaluation_strategy": "steps" if "validation" in tokenized else "no",
        "save_total_limit": config.save_total_limit,
        "gradient_checkpointing": config.gradient_checkpointing,
        "bf16": config.bf16,
        "fp16": config.fp16,
        "report_to": config.report_to,
        "seed": config.seed,
        "remove_unused_columns": False,
    }
    training_args = TrainingArguments(**_filtered_kwargs(TrainingArguments, training_kwargs))
    trainer_kwargs: dict[str, Any] = {
        "model": model,
        "args": training_args,
        "train_dataset": tokenized["train"],
        "eval_dataset": tokenized.get("validation"),
        "data_collator": DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
        "processing_class": tokenizer,
        "tokenizer": tokenizer,
    }
    trainer = Trainer(**_filtered_kwargs(Trainer, trainer_kwargs))
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
        description="canonical text를 causal LM CPT adapter로 학습합니다."
    )
    parser.add_argument("--config", required=True, type=Path, help="training YAML config")
    parser.add_argument("--dry-run", action="store_true", help="config만 검증하고 학습하지 않음")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    if args.dry_run:
        print(json.dumps(config.to_dict(), ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    output_dir = run_cpt(config)
    print(f"training output: {output_dir}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
