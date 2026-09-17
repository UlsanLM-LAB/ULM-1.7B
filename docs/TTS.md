# ULM-TTS

Experimental speech layer for ULM-1.7B. The text SLM remains the core research model; this package adds a reproducible audio contract, preprocessing, trainable VITS fine-tuning, evaluation metrics, and an integrated LLM→TTS pipeline.

## Architecture

```
[Korean text] → ulm-infer (dialect_strength) → [Ulsan dialect text] → ulm-tts → [.wav]
                     ↑                                                     ↑
              Qwen3-1.7B + adapter                              MMS/VITS fine-tuned
```

The full pipeline is available via `ulm-pipeline-tts`, which chains the LLM dialect conversion with TTS synthesis. Use `--skip-llm` to bypass dialect conversion when the input is already in dialect form.

## Current backend

The baseline is `facebook/mms-tts-kor` (VITS architecture). Its model license is CC-BY-NC-4.0; do not assume a future checkpoint can be commercially redistributed under this repository's Apache-2.0 code license.

### Training approach

The HuggingFace `VitsModel` does not expose a stable adversarial training API. The trainer (`ulm-train-tts`) implements mel-spectrogram reconstruction fine-tuning:

1. Freeze discriminator, flow, and posterior encoder
2. Keep text encoder and decoder (generator) trainable
3. Compute L1 loss between predicted mel spectrograms and target mel from real audio
4. Support gradient accumulation, AMP (fp16/bf16), warmup+cosine LR schedule, and checkpoint/resume

This is a practical approximation for dialect adaptation. For production quality, consider migrating to a fully trainable VITS or StyleTTS2 implementation.

## Modules

| Module | Description |
|--------|-------------|
| `data.py` | TTSRecord schema, manifest loading, speaker leakage check, consent validation, path traversal prevention |
| `preprocess.py` | Audio preprocessing: stereo→mono, resample, silence trim, clipping detection, peak normalization, transcript NFC normalization |
| `train.py` | VITS mel-reconstruction fine-tuning with checkpoint/resume |
| `inference.py` | Single-utterance TTS inference |
| `evaluate.py` | RTF, F0 pitch, duration, energy stats, batch synthesis, checkpoint comparison |
| `pipeline.py` | End-to-end LLM dialect conversion + TTS synthesis |

## Manifest

Use JSONL with all required fields: `utterance_id`, `speaker_id`, `audio_path`, `transcript`, `region_tier`, `age_group`, `dialect_strength`, `sample_rate`, `duration`, `consent_scope`, `quality_grade`, `split`, `metadata`.

Strict validation enforces:
- `consent_scope` in `{research-training, research-evaluation, commercial, example-only}`
- `quality_grade` in `{A, B, C, D, F}`
- `age_group` in `{10s, 20s, 30s, 40s, 50s, 60s, 70s, 80+}`
- `region_tier` in `{U0, U1, U2, GX}`
- No unknown fields (typo detection)
- No path traversal in `audio_path`
- No speaker leakage across splits

When `require_consent: true` is set in config, records with `consent_scope: example-only` are rejected.

Never commit private audio.

## Commands

```bash
pip install -e '.[tts]'

# Preprocess raw audio
ulm-preprocess-tts --manifest manifest.jsonl --audio-root raw/ --output-dir preprocessed/

# Validate data only
ulm-train-tts --config configs/tts/mms_vits.yaml --validate-only

# Train
ulm-train-tts --config configs/tts/mms_vits.yaml

# Inference
ulm-tts '밥 묵었나' --output outputs/tts/sample.wav

# Evaluate
ulm-eval-tts --config configs/tts/mms_vits.yaml --texts-file eval.txt --output-dir eval_out/

# Full pipeline (LLM → dialect → TTS)
ulm-pipeline-tts '오늘 날씨가 좋네요' --dialect-strength 2
```

## Config

See `configs/tts/mms_vits.yaml` for the full configuration reference including data, preprocessing, training, inference, and evaluation sections.
