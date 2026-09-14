# ULM-1.7B

`UlsanLM Lab`의 `Ulsan Language Model` 연구 저장소다.

목표는 약 1.7B 규모의 모델로 울산 지역어 이해·변환·대화·강도 제어를 검증하고, 일반 한국어 보존과 speaker-disjoint benchmark를 함께 구축하는 것이다.

현재 저장소는 실제 AI Hub 원본이나 학습 결과를 포함하지 않는다. CPU에서 schema, 데이터 audit, speaker split, benchmark, config, QLoRA/CPT adapter, local inference 경로와 테스트를 갖췄고, 이후 `Qwen/Qwen3-0.6B` QLoRA smoke test를 거쳐 `Qwen/Qwen3-1.7B` 연구 실험으로 확장한다.

## 빠른 시작

```bash
uv sync --extra dev
uv run pytest
uv run ruff check .
```

외부 데이터 없이 작은 fixture와 코드 검증만 실행한다. AI Hub 원본은 로그인·이용조건·개인정보 경계를 확인한 뒤 로컬 경로에서만 사용한다.

## 주요 문서

- [PLAN.md](PLAN.md): 목표, architecture, 연구 질문, 단계 의존성
- [TODO.md](TODO.md): 작은 작업 단위와 상태
- [RESEARCH.md](RESEARCH.md): primary research에서 추출한 판단
- [EXPERIMENTS.md](EXPERIMENTS.md): 실험 matrix와 결과 기록 규칙
- [GPU.md](GPU.md): CPU·Colab·GPU 운영 경계
- [data/README.md](data/README.md): 데이터 거버넌스와 입력 규칙
- [benchmarks/README.md](benchmarks/README.md): benchmark 작성·검수 규칙

## 현재 상태

문서와 CPU foundation 구현을 완료했다. 실제 울산 subset 통계와 모델 성능은 아직 없다. TTS/STT는 현재 scope가 아니며 후속 interface만 문서화한다.
