"""ULM-1.7B interactive dialect chat CLI."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from .prompt import _STRENGTH_GUIDANCE


def run_chat(
    model_name: str = "Qwen/Qwen3-1.7B",
    adapter_path: str | Path | None = None,
    *,
    dialect_strength: int = 2,
    load_in_4bit: bool = True,
    temperature: float = 0.7,
    max_new_tokens: int = 128,
    enable_tts: bool = False,
    tts_output: str | Path = "outputs/tts/chat_reply.wav",
) -> None:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("ML extra required: uv sync --extra ml") from exc

    print("\n모델 로딩 중... (잠시만 기다려주세요)")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model_kwargs: dict[str, Any] = {"device_map": "auto"}

    if load_in_4bit:
        from transformers import BitsAndBytesConfig

        compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype,
        )
    else:
        model_kwargs["torch_dtype"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    if adapter_path:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(adapter_path))

    tts_pipeline = None
    if enable_tts:
        try:
            from transformers import VitsModel

            tts_tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-kor")
            tts_model = VitsModel.from_pretrained("facebook/mms-tts-kor")
            tts_device = "cuda" if torch.cuda.is_available() else "cpu"
            tts_model = tts_model.to(tts_device).eval()
            tts_pipeline = (tts_model, tts_tokenizer, tts_device)
        except Exception as e:
            print(f"[경고] TTS 로드 실패: {e}")

    guidance = _STRENGTH_GUIDANCE.get(dialect_strength, _STRENGTH_GUIDANCE[2])
    system_prompt = (
        "울산 지역어 대화 assistant로서 자연스럽고 일상적인 울산 사투리로 "
        f"상대방과 친근하게 대화한다. {guidance}"
    )

    history: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]

    print("=" * 60)
    print("  ULM-1.7B 울산 사투리 대화 모드")
    print(f"  - 모델: {model_name} (adapter: {adapter_path or 'none'})")
    print(f"  - 사투리 강도: {dialect_strength} ({guidance})")
    print("  - 종료하려면 'q', 'quit', 'exit'를 입력하거나 Ctrl+C를 누르세요.")
    print("=" * 60 + "\n")

    while True:
        try:
            user_input = input("나 > ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n대화를 종료합니다.")
            break

        if not user_input:
            continue
        if user_input.lower() in ("q", "quit", "exit", "종료"):
            print("대화를 종료합니다.")
            break

        history.append({"role": "user", "content": user_input})

        template_kwargs: dict[str, Any] = {
            "tokenize": False,
            "add_generation_prompt": True,
            "enable_thinking": False,
        }
        try:
            prompt = tokenizer.apply_chat_template(history, **template_kwargs)
        except TypeError:
            template_kwargs.pop("enable_thinking", None)
            prompt = tokenizer.apply_chat_template(history, **template_kwargs)

        inputs = tokenizer(prompt, return_tensors="pt")
        if hasattr(model, "device"):
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.inference_mode():
            output = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=0.9,
                do_sample=True,
            )

        in_len = inputs["input_ids"].shape[-1]
        reply = tokenizer.decode(output[0][in_len:], skip_special_tokens=True).strip()

        print(f"ULM > {reply}\n")
        history.append({"role": "assistant", "content": reply})

        # Keep history from growing too long (sliding window)
        if len(history) > 21:
            history = [history[0]] + history[-20:]

        if tts_pipeline:
            try:
                import scipy.io.wavfile

                tm, tt, td = tts_pipeline
                t_inputs = tt(reply, return_tensors="pt").to(td)
                with torch.inference_mode():
                    waveform = tm(**t_inputs).waveform[0].cpu().float().numpy()
                out_path = Path(tts_output)
                out_path.parent.mkdir(parents=True, exist_ok=True)
                scipy.io.wavfile.write(str(out_path), rate=tm.config.sampling_rate, data=waveform)
                print(f"  (음성 생성됨: {out_path})")
            except Exception as e:
                print(f"  (음성 생성 실패: {e})")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="ULM-1.7B 대화형 채팅 터미널")
    parser.add_argument("--model-name", default="Qwen/Qwen3-1.7B")
    parser.add_argument(
        "--adapter",
        default="outputs/qwen3-1.7b-sft/checkpoint-50",
        help="PEFT adapter directory",
    )
    parser.add_argument("--dialect-strength", type=int, default=2, choices=range(4))
    parser.add_argument("--load-in-4bit", action="store_true", default=True)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument(
        "--voice", action="store_true", help="답변할 때마다 TTS 음성 파일도 함께 생성"
    )
    args = parser.parse_args(argv)

    adapter = args.adapter if (args.adapter and Path(str(args.adapter)).exists()) else None
    run_chat(
        model_name=args.model_name,
        adapter_path=adapter,
        dialect_strength=args.dialect_strength,
        load_in_4bit=args.load_in_4bit,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        enable_tts=args.voice,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
