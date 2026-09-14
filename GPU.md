# ULM-1.7B GPU 계획

이 문서의 비용과 VRAM은 정확한 실행 보장이 아니라 작업 분류 기준이다. 실제 시간·가격·가용성은 실행 시점에 다시 확인한다. 수 시간짜리 학습과 유료 instance 생성은 자동으로 수행하지 않는다.

## 현재 환경

| 항목 | 확인 결과 |
|---|---|
| 로컬 GPU | `NVIDIA GeForce RTX 5060 Laptop GPU` |
| 보고된 VRAM | 약 `8 GB` |
| Python | `uv` 경유 Python `3.11.15` |
| ML package | 현재 environment에서 미설치 |
| model weights/cache | 확인되지 않음 |
| 결론 | CPU foundation 가능, GPU smoke는 준비 상태 |

## 단계별 자원 계획

| 단계 | CPU 가능 여부 | 최소 VRAM | 권장 VRAM | Colab 가능 여부 | 유료 GPU 필요 여부 | 대안 | 비용 수준 |
|---|---|---:|---:|---|---|---|---|
| schema·validation·split | 가능 | 0 GB | 0 GB | 불필요 | 없음 | 로컬 CPU | 무료 |
| AI Hub audit | 가능 | 0 GB | 0 GB | 가능 | 없음 | 합법적으로 받은 local archive | 무료, 저장공간 별도 |
| benchmark metric | 가능 | 0 GB | 0 GB | 불필요 | 없음 | 로컬 CPU | 무료 |
| `Qwen/Qwen3-0.6B` tokenizer/inference | 제한적으로 가능 | 2~4 GB | 4~8 GB | 가능 | 없음 | 로컬 GPU 또는 무료 runtime | 무료 우선 |
| 0.6B 5~20 step QLoRA | 비권장 | 6 GB 이상 | 8 GB 이상 | 가능 | 없음 | 현재 로컬 GPU에서 짧게 검토 | 무료 smoke |
| 1.7B QLoRA SFT | 이론상 가능하나 느림 | 8 GB 이상 | 12~24 GB | 가능 여부 runtime에 따라 다름 | 기본값 아님 | 무료 Colab, 나중에 저비용 GPU | 무료 우선 |
| 1.7B CPT 반복 | 비권장 | 12 GB 이상 | 24~48 GB | 장시간 작업에는 부적합 | 필요할 수 있음 | 사용자가 별도 승인한 저비용 GPU | 승인 후 |
| TTS training | 범위 밖 | 별도 측정 | 별도 측정 | managed runtime 정책 검토 필요 | 별도 검토 | local GPU/자체 관리 환경 | 후속 |
| full training·merge | 범위 밖 | 모델/설정 의존 | 모델/설정 의존 | 장시간에 부적합 | 필요할 수 있음 | 사용자 결정 후 실행 | 승인 후 |

VRAM 수치는 smoke 분류용 보수적 범위이며, 실제 sequence length, quantization, batch, checkpoint 설정으로 달라진다. 근거 없는 training time은 기록하지 않는다.

## Colab 운영

- 무료 runtime의 GPU 종류와 사용 한도는 고정되지 않는다.
- checkpoint는 local `/content`에 저장하고 일정 간격으로 Drive에 백업한다.
- Drive에서 작은 파일을 직접 반복 읽지 않고 archive를 `/content`로 복사한다.
- notebook은 `save_steps`, `eval_steps`, `save_total_limit`, `seed`, `resume_from_checkpoint`를 config로 받는다.
- 현재 범위에서는 text QLoRA smoke만 실행한다.
- voice cloning/fine-tuning은 managed runtime 정책과 개인정보 위험 때문에 실행하지 않는다.

## 비용 정책

1. 로컬 CPU
2. 무료 Colab 또는 무료 GPU
3. 정말 필요한 짧은 저비용 GPU
4. 유료 GPU는 사용자 확인 후 최종 실험에서만

이 저장소는 결제, 유료 GPU instance 생성, 외부 로그인, API key 입력을 자동으로 하지 않는다. 현재 `S0`는 `READY_FOR_TRAINING`이며, 의존성과 공개 model download가 준비된 뒤 5~20 step만 실행한다.
