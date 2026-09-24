"""base/adapter model local inference CLI."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from .policy import (
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
)
from .policy import (
    generation_kwargs as phase4_generation_kwargs,
)
from .prompt import build_inference_messages


def generate_text(
    model_name: str,
    text: str,
    *,
    dialect_strength: int | None = None,
    adapter_path: str | None = None,
    load_in_4bit: bool = False,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
    top_p: float = DEFAULT_TOP_P,
    do_sample: bool = True,
) -> str:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:  # pragma: no cover - depends on runtime extras
        raise RuntimeError(
            "inference에는 ML extra가 필요합니다. `uv sync --extra dev --extra ml`을 실행하세요."
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model_kwargs: dict[str, Any] = {"device_map": "auto", "torch_dtype": "auto"}
    if load_in_4bit:
        from transformers import BitsAndBytesConfig

        compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype,
        )
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    if adapter_path:
        try:
            from peft import PeftModel
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("adapter inference에는 `peft`가 필요합니다") from exc
        model = PeftModel.from_pretrained(model, adapter_path)
    messages = build_inference_messages(text, dialect_strength)
    template_kwargs: dict[str, Any] = {
        "tokenize": False,
        "add_generation_prompt": True,
        "enable_thinking": False,
    }
    try:
        prompt = tokenizer.apply_chat_template(messages, **template_kwargs)
    except TypeError:
        template_kwargs.pop("enable_thinking", None)
        prompt = tokenizer.apply_chat_template(messages, **template_kwargs)
    inputs = tokenizer(prompt, return_tensors="pt")
    if hasattr(model, "device"):
        inputs = {key: value.to(model.device) for key, value in inputs.items()}
    generation_kwargs = phase4_generation_kwargs(
        max_new_tokens=max_new_tokens,
        temperature=temperature if do_sample else 0,
        top_p=top_p,
        eos_token_id=tokenizer.eos_token_id,
    )
    with torch.inference_mode():
        output = model.generate(**inputs, **generation_kwargs)
    input_length = inputs["input_ids"].shape[-1]
    return tokenizer.decode(output[0][input_length:], skip_special_tokens=True).strip()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="학습된 ULM model 또는 adapter로 local inference를 실행합니다."
    )
    parser.add_argument("--model-name", required=True, help="base model ID 또는 local path")
    parser.add_argument("--adapter", type=Path, help="선택적 PEFT adapter directory")
    parser.add_argument("--text", help="입력 text")
    parser.add_argument(
        "--text-file", type=Path, help="입력 text file; --text와 함께 사용할 수 없음"
    )
    parser.add_argument(
        "--dialect-strength", type=int, default=None, choices=range(4), metavar="0-3"
    )
    parser.add_argument(
        "--load-in-4bit",
        action="store_true",
        help="메모리 절약을 위해 base model을 4-bit(NF4)로 로드",
    )
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--top-p", type=float, default=DEFAULT_TOP_P)
    parser.add_argument("--greedy", action="store_true", help="sampling 없이 greedy decoding")
    parser.add_argument("--output", type=Path, help="출력 text 경로; 생략하면 stdout")
    parser.add_argument("--overwrite", action="store_true", help="기존 출력 파일을 덮어씀")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if bool(args.text) == bool(args.text_file):
        raise ValueError("--text 또는 --text-file 중 정확히 하나를 지정해야 합니다")
    text = args.text if args.text is not None else args.text_file.read_text(encoding="utf-8")
    result = generate_text(
        args.model_name,
        text,
        dialect_strength=args.dialect_strength,
        adapter_path=str(args.adapter) if args.adapter else None,
        load_in_4bit=args.load_in_4bit,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        do_sample=not args.greedy,
    )
    if args.output is None:
        print(result)
    else:
        if args.output.exists() and not args.overwrite:
            raise FileExistsError(f"기존 출력 파일을 덮어쓰지 않습니다: {args.output}")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(result + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
