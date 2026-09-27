# ULM-4B

UlsanLM Lab의 울산 방언 특화 소형 언어모델 연구 프로젝트입니다.

> 저장소 이름은 초기 1.7B 실험명인 `ULM-1.7B`를 유지하고 있지만, 현재 주력 계열은 Qwen3.8-4B-Distill 기반 ULM-4B입니다.

![Python](https://img.shields.io/badge/Python-3.11%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange)
![Transformers](https://img.shields.io/badge/Hugging%20Face-Transformers-yellow)
![FastAPI](https://img.shields.io/badge/API-FastAPI-009688)
![License](https://img.shields.io/badge/License-Apache--2.0-green)

## 현재 릴리스 후보

현재 데모와 ULM Live 연동에 사용하는 체크포인트는 `ULM-4B Arm B`입니다.

| 항목 | 현재 값 |
| --- | --- |
| Base model | Qwen3.8-4B-Distill |
| Tuning | LoRA SFT |
| Lineage | Dialect Alignment v3 → Context Repair v1 Arm B → Instruction Recovery v1 Arm B |
| Status | Release Candidate |
| Serving | FastAPI / OpenAI-compatible chat API |
| Voice | ULM-LIVE + Qwen3-TTS |
| Thinking | disabled |
| Repetition control | repetition_penalty 1.10 + no_repeat_ngram_size 3 |

Arm B는 울산 방언 생성, 문법, 문맥 응답, 일반 지식 보존을 우선해 고정한 릴리스 후보입니다. instruction-trap 회귀는 아직 남아 있어 연구용 최종 체크포인트로 확정한 상태는 아닙니다.

자세한 수치와 선택 이유는 [Arm B release candidate report](reports/ULM_4B_ARM_B_RELEASE_CANDIDATE.md)와 [MODEL_CARD.md](MODEL_CARD.md)를 참고하세요.

## 평가 요약

| 평가 | Arm B |
| --- | ---: |
| UlsanBench generation semantic | 0.9843 |
| UlsanBench generation dialectness | 0.6589 |
| UlsanBench grammar dialectness | 0.8955 |
| UlsanBench context semantic | 0.4578 |
| Context repetition | 0 |
| Factual QA | 84% |
| Multi-turn | 100% |
| Instruction trap | 65% |
| UlsanBench identification accuracy | 32% |

Context 반복은 기존 greedy decoding에서 발생했지만, 현재 릴리스 디코딩 정책에서는 75개 context 항목에서 0회로 줄었습니다.

## 구조

```text
Browser / Client
       │
       ▼
FastAPI chat server
       │
       ▼
Qwen3.8-4B-Distill
       +
ULM-4B Arm B LoRA
       │
       ├── text response
       │
       └── ULM-LIVE
              │
              ▼
        Qwen3-TTS
              │
              ▼
          24 kHz WAV
```

음성 경로는 별도 저장소 [ULM-LIVE](https://github.com/UlsanLM-LAB/ULM-LIVE)에서 관리합니다.

## 빠른 시작

```bash
uv sync --extra dev --extra ml
uv run pytest
uv run ruff check .
```

모델 가중치와 AI Hub 원본 데이터는 저장소에 포함하지 않습니다.

### Arm B 서버 실행

현재 release candidate는 base model과 LoRA adapter를 분리해서 로드할 수 있습니다.

```bash
export ULM_MODEL_PATH=/path/to/Qwen3.8-4B-Distill
export ULM_ADAPTER_PATH=/path/to/ulm4b-arm-b/final_adapter

uv run python scripts/serve.py \
  --model "$ULM_MODEL_PATH" \
  --host 127.0.0.1 \
  --port 8000
```

상태 확인:

```bash
curl http://127.0.0.1:8000/health
```

비스트리밍 요청:

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "messages": [{"role": "user", "content": "오늘 뭐하노?"}],
    "stream": false,
    "temperature": 0
  }'
```

SSE 스트리밍도 같은 엔드포인트에서 `"stream": true`로 사용할 수 있습니다.

## 현재 디코딩 정책

Release Candidate에서 반복 루프를 줄이기 위해 다음 설정을 기본으로 사용합니다.

```text
enable_thinking = false
repetition_penalty = 1.10
no_repeat_ngram_size = 3
```

샘플링을 사용할 때는 기존 Phase 4 정책의 `temperature`, `top_p`, `top_k` 설정을 함께 사용합니다.

## 저장소 구성

```text
src/ulm/                 모델 학습·추론 핵심 코드
scripts/                 데이터 구축, 학습, 평가, 서빙 스크립트
configs/                 SFT 및 release 설정
data/                    공개 가능한 메타데이터·benchmark
benchmarks/              평가 설계 및 규칙
reports/                 실험 결과와 의사결정 기록
docs/                    아키텍처 및 운영 문서
portfolio/               포트폴리오용 시각 자료
tests/                   회귀 및 API 테스트
```

실험 산출물과 대형 체크포인트는 Git에 직접 커밋하지 않습니다.

## 주요 문서

- [MODEL_CARD.md](MODEL_CARD.md): 현재 ULM-4B 모델 카드
- [Arm B release candidate report](reports/ULM_4B_ARM_B_RELEASE_CANDIDATE.md): 현재 고정 후보와 평가 결과
- [PLAN.md](PLAN.md): 연구 목표와 단계
- [EXPERIMENTS.md](EXPERIMENTS.md): 실험 기록 규칙
- [GPU.md](GPU.md): GPU 운영 가이드
- [benchmarks/README.md](benchmarks/README.md): benchmark 작성 및 검수 규칙
- [data/README.md](data/README.md): 데이터 거버넌스

## 현재 상태

ULM-4B Arm B는 텍스트 추론과 ULM Live의 TTS 경로까지 end-to-end 동작을 확인한 release candidate입니다.

다음 우선순위는 instruction-following 회귀 복구, identification 성능 개선, STT 연결, 실시간 음성 스트리밍과 barge-in 지원입니다.

## 라이선스

코드 라이선스는 저장소의 [LICENSE](LICENSE)를 따릅니다. Base model, 데이터셋, 음성 모델은 각각의 원 라이선스와 이용 조건을 별도로 따라야 합니다.
