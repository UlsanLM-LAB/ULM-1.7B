# ULM-1.7B Roadmap

## 현재 위치

현재 저장소는 `README.md`, `LICENSE`, primary research report만 있는 초기 상태에서 시작했다. 문서 foundation을 먼저 고정하고, CPU에서 재현 가능한 data·evaluation·config foundation을 완성한 뒤 GPU 학습으로 넘어간다.

## Phase 0 — 연구 경계와 repository foundation

상태: `DONE` 진행 중인 문서 커밋으로 완료

- `deep-research-report.md` 전체 확인
- 연구 질문, 모델 후보, 데이터·license 경계 정리
- 무료 환경 우선과 TTS/STT 범위 제한
- 테스트·config·notebook 구조 결정

## Phase 1 — CPU foundation

상태: `DONE`

- Python package와 `pyproject.toml`
- unified dataset schema와 JSONL 입출력
- schema validation
- speaker-aware deterministic split
- AI Hub audit CLI
- ULM-Bench schema와 baseline metric
- training config loader

CPU foundation의 schema·audit·filter·split·benchmark·config·CLI와 테스트가 완료되었다. 실제 raw archive는 포함하지 않는다.

완료 기준은 외부 데이터나 GPU 없이 unit test가 통과하는 것이다.

## Phase 2 — Smoke pipeline

상태: `DONE`

- `Qwen/Qwen3-0.6B` tokenizer·model load
- 5~20 step QLoRA SFT
- checkpoint save 및 resume
- fixture evaluation
- local inference

로컬 RTX 5060 Laptop GPU에서 `Qwen/Qwen3-0.6B` 5 step smoke와 checkpoint resume을 완료했다. 결과는 fixture 검증이며 울산 성능 결과가 아니다.

무료 Colab 또는 현재 로컬 8GB GPU에서만 짧게 실행한다. ML package/model download가 필요하므로 실제 결과는 실행 후에만 기록한다.

## Phase 3 — ULM-1.7B 연구 실험

상태: `READY_FOR_TRAINING`

- `Qwen/Qwen3-1.7B` SFT MVP
- `Qwen/Qwen3-1.7B-Base` CPT
- `Ulsan Core`와 `Gyeongsang Auxiliary` ablation
- general Korean retention
- `dialect_strength` conditioning

실제 Ulsan subset audit와 native review가 끝나기 전에는 모델 성능을 주장하지 않는다.

## Phase 4 — ULM-Bench와 분석

상태: `PLANNED`

- 권리를 확보한 benchmark item 수집
- 원어민 검수 workflow
- speaker/template/near-duplicate leakage 점검
- 1,000개 이상으로 확장
- confusion matrix와 error analysis

## Phase 5 — 공개와 생태계 확장

상태: `REVIEW`

- model card와 dataset card
- adapter/GGUF 공개 여부 검토
- 동의된 voice profile 기반의 별도 `ULM-TTS`
- STT → ULM → TTS end-to-end demo

음성 원본·voice checkpoint 공개는 동의와 개인정보 검토 없이는 진행하지 않는다.

## 의존성 흐름

```text
문서
  -> package/config
  -> schema/validation
  -> split/audit
  -> benchmark
  -> smoke
  -> 1.7B experiments
  -> native evaluation
  -> release decision
```
