"""ULM text-to-speech pipeline: LLM dialect generation -> TTS synthesis.

Connects the ULM inference engine (dialect_strength-controlled text generation)
with the TTS backend to produce spoken Ulsan-dialect audio from standard Korean
input text.
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


def generate_and_synthesize(
    text: str,
    *,
    llm_model: str = "Qwen/Qwen3-1.7B",
    llm_adapter: str | None = None,
    dialect_strength: int = 2,
    tts_model_id: str = "facebook/mms-tts-kor",
    tts_config: dict[str, Any] | None = None,
    output_path: str | Path = "outputs/tts/pipeline_out.wav",
    load_in_4bit: bool = False,
    max_new_tokens: int = 128,
    temperature: float = 0.7,
    skip_llm: bool = False,
) -> dict[str, Any]:
    """End-to-end pipeline: standard Korean -> Ulsan dialect text -> speech.

    If *skip_llm* is True, the input text is passed directly to TTS without
    dialect conversion (useful when the text is already in dialect form).

    Returns a dict with timing, generated text, and output path.
    """
    result: dict[str, Any] = {"input_text": text}

    # Step 1: LLM dialect conversion
    if skip_llm:
        dialect_text = text
        result["llm_time"] = 0.0
    else:
        from ulm.inference.cli import generate_text

        t0 = time.time()
        dialect_text = generate_text(
            llm_model,
            text,
            dialect_strength=dialect_strength,
            adapter_path=llm_adapter,
            load_in_4bit=load_in_4bit,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
        result["llm_time"] = time.time() - t0
        log.info("LLM output (%d chars): %s", len(dialect_text), dialect_text[:80])

    result["dialect_text"] = dialect_text

    # Step 2: TTS synthesis
    try:
        import scipy.io.wavfile
        import torch
        from transformers import AutoTokenizer, VitsModel
    except ImportError as exc:
        raise RuntimeError("TTS deps missing: pip install -e '.[tts]'") from exc

    tts_cfg = tts_config or {}
    inf_cfg = tts_cfg.get("inference", {})

    t1 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(tts_model_id)
    model = VitsModel.from_pretrained(tts_model_id)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()

    inputs = tokenizer(dialect_text, return_tensors="pt").to(device)
    with torch.inference_mode():
        waveform = (
            model(
                **inputs,
                noise_scale=float(inf_cfg.get("noise_scale", 0.667)),
                noise_scale_duration=float(inf_cfg.get("noise_scale_duration", 0.8)),
                speaking_rate=float(inf_cfg.get("speaking_rate", 1.0)),
            )
            .waveform[0]
            .detach()
            .cpu()
            .float()
            .numpy()
        )
    tts_time = time.time() - t1
    result["tts_time"] = tts_time

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    sr = model.config.sampling_rate
    scipy.io.wavfile.write(str(out), rate=sr, data=waveform)

    audio_duration = len(waveform) / sr
    result["output_path"] = str(out)
    result["audio_duration"] = audio_duration
    result["rtf"] = tts_time / max(audio_duration, 1e-6)
    result["total_time"] = result.get("llm_time", 0.0) + tts_time
    log.info(
        "pipeline complete: audio=%.2fs rtf=%.3f output=%s",
        audio_duration,
        result["rtf"],
        out,
    )
    return result


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for the ULM text-to-speech pipeline."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    parser = argparse.ArgumentParser(
        description="ULM pipeline: Korean text -> Ulsan dialect -> TTS audio"
    )
    parser.add_argument("text", help="input Korean text")
    parser.add_argument("--config", default="configs/tts/mms_vits.yaml")
    parser.add_argument("--output", default="outputs/tts/pipeline_out.wav")
    parser.add_argument("--llm-model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--adapter", default=None, help="PEFT adapter path")
    parser.add_argument("--dialect-strength", type=int, default=2, choices=range(4))
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument(
        "--skip-llm",
        action="store_true",
        help="skip LLM dialect conversion; use input text directly for TTS",
    )
    args = parser.parse_args(argv)

    import yaml

    tts_config: dict[str, Any] = {}
    cfg_path = Path(args.config)
    if cfg_path.is_file():
        with cfg_path.open(encoding="utf-8") as f:
            tts_config = yaml.safe_load(f) or {}

    tts_model = tts_config.get("model_id", "facebook/mms-tts-kor")

    result = generate_and_synthesize(
        args.text,
        llm_model=args.llm_model,
        llm_adapter=args.adapter,
        dialect_strength=args.dialect_strength,
        tts_model_id=tts_model,
        tts_config=tts_config,
        output_path=args.output,
        load_in_4bit=args.load_in_4bit,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        skip_llm=args.skip_llm,
    )

    print(f"dialect_text: {result['dialect_text']}")
    print(f"output: {result['output_path']}")
    print(f"audio: {result['audio_duration']:.2f}s  rtf: {result['rtf']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
