# ULM-1.7B

`UlsanLM Lab`의 `Ulsan Language Model` 연구 저장소다.

목표는 약 1.7B 규모의 모델로 울산 지역어 이해·변환·대화·강도 제어를 검증하고, 일반 한국어 보존과 speaker-disjoint benchmark를 함께 구축하는 것이다.

현재 저장소는 실제 AI Hub 원본이나 학습 결과를 포함하지 않는다. CPU에서 schema, 데이터 audit, speaker split, benchmark, config, QLoRA/CPT adapter, local inference 경로와 테스트를 갖추었고, TTS 파이프라인(전처리·학습·평가·추론·LLM 연동)까지 구현되어 있다.

## 빠른 시작

```bash
uv sync --extra dev
uv run pytest
uv run ruff check .
```

외부 데이터 없이 작은 fixture와 코드 검증만 실행한다. AI Hub 원본은 로그인·이용조건·개인정보 경계를 확인한 뒤 로컬 경로에서만 사용한다.

## ULM-1.7B 채팅 서버

Phase 4 merged model을 한 번 로드한 뒤 OpenAI Chat Completions와 유사한 HTTP/SSE API로 제공한다. 저장된 tokenizer의 chat template을 그대로 사용하며 CUDA에서는 bf16(지원되지 않으면 fp16), CPU에서는 float32를 선택한다.

```bash
uv sync --extra dev --extra ml

# CLI 인자
uv run python scripts/serve.py \
  --model /path/to/ulm-1.7b-phase4-best-merged \
  --host 127.0.0.1 \
  --port 8000

# 또는 환경변수
ULM_MODEL_PATH=/path/to/ulm-1.7b-phase4-best-merged \
  uv run python scripts/serve.py --port 8000
```

서버 확인:

```bash
curl http://localhost:8000/health

curl -N http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "messages": [{"role": "user", "content": "오늘 뭐하노?"}],
    "stream": true,
    "temperature": 0.7,
    "max_tokens": 512
  }'
```

`stream: true`는 `text/event-stream` chunk와 마지막 `data: [DONE]`을 반환한다. `stream: false`는 일반 JSON completion을 반환한다. 요청 취소 시 stopping criteria를 통해 생성 중단을 시도하며, 단일 모델의 동시 generation은 GPU 메모리 안전을 위해 직렬화한다.

## TTS 파이프라인

TTS 기능을 사용하려면 `[tts]` extra를 설치한다:

```bash
uv sync --extra dev --extra tts
```

### 명령어

| 명령어 | 설명 |
|--------|------|
| `ulm-preprocess-tts` | 오디오 전처리 (mono 변환, resample, silence trim, 정규화) |
| `ulm-train-tts` | VITS mel-reconstruction fine-tuning (checkpoint/resume 지원) |
| `ulm-tts` | MMS/VITS baseline 추론 |
| `ulm-eval-tts` | 배치 합성 및 평가 (RTF, F0, duration, energy) |
| `ulm-pipeline-tts` | LLM 사투리 변환 + TTS 합성 통합 파이프라인 |

### 예시

```bash
# 데이터 전처리
ulm-preprocess-tts --manifest data/private/tts/manifest.jsonl \
    --audio-root data/private/tts/audio --output-dir outputs/tts/preprocessed

# 데이터 검증만
ulm-train-tts --config configs/tts/mms_vits.yaml --validate-only

# 학습
ulm-train-tts --config configs/tts/mms_vits.yaml

# 단일 추론
ulm-tts '밥 묵었나' --output outputs/tts/sample.wav

# 평가
ulm-eval-tts --config configs/tts/mms_vits.yaml \
    --texts-file eval_texts.txt --output-dir outputs/tts/eval

# 통합 파이프라인 (LLM -> 사투리 변환 -> TTS)
ulm-pipeline-tts '오늘 날씨가 좋네요' --dialect-strength 2
```

## 주요 문서

- [PLAN.md](PLAN.md): 목표, architecture, 연구 질문, 단계 의존성
- [TODO.md](TODO.md): 작은 작업 단위와 상태
- [RESEARCH.md](RESEARCH.md): primary research에서 추출한 판단
- [EXPERIMENTS.md](EXPERIMENTS.md): 실험 matrix와 결과 기록 규칙
- [GPU.md](GPU.md): CPU·Colab·GPU 운영 경계
- [docs/TTS.md](docs/TTS.md): TTS 아키텍처, 매니페스트 스키마, 백엔드 전략
- [data/README.md](data/README.md): 데이터 거버넌스와 입력 규칙
- [benchmarks/README.md](benchmarks/README.md): benchmark 작성·검수 규칙

## 현재 상태

문서, CPU foundation, 0.6B smoke test를 완료했다. TTS 파이프라인(MMS/VITS baseline)은 전처리·학습·추론·평가·LLM 연동까지 구현되어 있으며, 실제 울산 음성 데이터로 fine-tuning할 준비가 되어 있다. 1.7B 연구 실험과 실측 평가는 데이터 확보 후 진행한다.
