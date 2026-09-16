# ULM-TTS

Experimental speech layer for ULM-1.7B. The text SLM remains the core research model; this package adds a reproducible audio contract and a replaceable Korean TTS backend.

## Current backend

The first inference baseline is `facebook/mms-tts-kor` (VITS). It is intentionally treated as a baseline rather than the final ULM-TTS architecture. Its model license is CC-BY-NC-4.0, so do not assume a future checkpoint can be commercially redistributed under this repository's Apache-2.0 code license.

The generic Hugging Face `VitsModel` interface is reliable for inference but does not expose a stable waveform-supervised fine-tuning objective. `ulm-train-tts` therefore validates and loads the full dataset but deliberately stops before acoustic optimization instead of pretending to train. Code review should pin a trainable Korean TTS implementation before removing this gate.

## Manifest

Use JSONL with the fields documented in `PLAN.md`: `utterance_id`, `speaker_id`, `audio_path`, `transcript`, `region_tier`, `age_group`, `dialect_strength`, `sample_rate`, `duration`, `consent_scope`, `quality_grade`, `split`, `metadata`.

Never commit private audio. A speaker may occur in only one split.

## Commands

```bash
pip install -e '.[tts]'
ulm-train-tts --config configs/tts/mms_vits.yaml --validate-only
ulm-tts '밥 묵었나' --output outputs/tts/sample.wav
```

Local CUDA is supported by inference. The final acoustic trainer should support CUDA mixed precision, gradient accumulation, checkpoint/resume and a CPU validation-only path.

## Reviewer checklist

1. Audit actual Ulsan audio hours, speaker count, sample rates and consent scope.
2. Select/pin a trainable Korean TTS backend; compare StyleTTS2 Korean adaptation or another actively maintained implementation against MMS/VITS baseline.
3. Implement the backend-native acoustic loss/trainer; do not bolt an arbitrary loss onto Transformers `VitsModel`.
4. Add checkpoint/resume and validation synthesis.
5. Add native-speaker MOS/pairwise evaluation for naturalness and Ulsan authenticity.
6. Only then connect ULM `dialect_strength` to prosody controls.
