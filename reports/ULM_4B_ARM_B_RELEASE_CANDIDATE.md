# ULM-4B Arm B Release Candidate

Date: 2026-09-27

## Decision

Instruction Recovery v1 Arm B를 현재 ULM-4B release candidate로 고정합니다.

이 결정은 연구 실험의 모든 gate를 통과했다는 의미가 아닙니다. 내일 예정된 ULM Live 통합과 데모를 위해, 현재 확보된 후보 중 방언 생성·문법·문맥 응답·일반 능력 보존의 균형이 가장 실용적인 체크포인트를 고정한 것입니다.

## Lineage

```text
Qwen3.8-4B-Distill
  → Dialect Alignment v3
  → Context Repair v1 Arm B
  → Instruction Recovery v1 Arm B
```

Instruction Recovery Arm B:
- learning rate: 6e-7
- train fraction: 0.25
- training steps: 35
- completion-only loss: enabled
- initial adapter: Context Repair v1 Arm B

## Evaluation snapshot

| Gate / metric | Result | State |
| --- | ---: | --- |
| Factual QA | 84% | pass |
| Multi-turn | 100% | pass |
| Instruction trap | 65% | fail |
| Generation semantic | 0.9843 | pass |
| Generation dialectness | 0.6589 | pass |
| Grammar semantic | 0.9846 | pass |
| Grammar dialectness | 0.8955 | pass |
| Context semantic | 0.4578 | pass for RC threshold |
| Context repetition | 0 | pass |
| Context malformed | 0 | pass |
| Context echo | 0 | pass |
| Comprehension semantic | 0.9802 | pass |
| Identification accuracy | 32% | improvement needed |

## Why Arm B

Arm A degraded several UlsanBench dimensions, while Arm B preserved generation, grammar, comprehension and context quality much better.

Arm B also keeps:
- factual QA at 84%
- multi-turn at 100%
- context repetition at 0 with the release decoding policy

The remaining major blocker is strict instruction following. That work is separated from the release candidate so that ULM Live integration can proceed without continuing an open-ended retraining loop.

## Release decoding

The release candidate should be served with:

```text
enable_thinking = false
repetition_penalty = 1.10
no_repeat_ngram_size = 3
```

This setting removed the observed context repetition failures in the 75-item context slice used during the current ablation.

## Deployment verification

The release candidate was loaded on the EC2 GPU runtime as:

```text
Qwen3.8-4B-Distill + Instruction Recovery v1 Arm B adapter
```

The FastAPI text endpoint returned healthy status and was connected to ULM-LIVE.

An end-to-end request completed successfully:

```text
text prompt
  → ULM-4B Arm B
  → generated response
  → Qwen3-TTS
  → 24 kHz mono PCM WAV
```

## Remaining work

1. Repair strict instruction following without degrading dialect behavior.
2. Improve identification accuracy.
3. Add STT input to ULM Live.
4. Add streaming speech generation and interruption handling.
5. Run human evaluation for dialect naturalness and voice quality.

## Release policy

Until a later candidate beats Arm B on both core UlsanBench behavior and preservation gates, Arm B is the default integration target for ULM Live.
