# ULM-1.7B TODO

상태는 `TODO`, `IN_PROGRESS`, `DONE`, `BLOCKED`, `READY_FOR_TRAINING`, `REVIEW` 중 하나를 사용한다. 각 항목은 가능하면 하나의 logical commit 크기로 유지한다.

## 완료

### T00 — 문서·범위 foundation

- 상태: `DONE`
- priority: P0
- label: `[LUNA] [FLASH] [FREE]`
- 목표: primary research를 바탕으로 scope, architecture, risk, DoD를 고정한다.
- 관련 파일: `PLAN.md`, `ROADMAP.md`, `RESEARCH.md`, `EXPERIMENTS.md`, `GPU.md`, `README.md`
- dependency: `deep-research-report.md` 정독
- 완료 조건: self-review 항목과 외부 작업 경계가 문서에 기록됨
- test: 문서 교차 검토

## 구현 순서

### T01 — Python package와 개발 도구

- 상태: `DONE`
- priority: P0
- label: `[FREE]`
- 목표: `src/ulm` package, `pyproject.toml`, test 명령, `.gitignore`를 만든다.
- 관련 파일: `pyproject.toml`, `src/ulm/`, `tests/`, `.gitignore`
- dependency: T00
- 완료 조건: CPU에서 package import와 test discovery가 동작함
- test: `uv run --extra dev pytest`, `uv run --extra dev ruff check`

### T02 — unified dataset schema

- 상태: `DONE`
- priority: P0
- label: `[LUNA] [FREE]`
- 목표: task, provenance, speaker, annotation, split을 표현하는 immutable record와 JSON 변환을 구현한다.
- 관련 파일: `src/ulm/data/schema.py`, `src/ulm/data/io.py`
- dependency: T01
- 완료 조건: 유효 record round-trip과 명시적 task/enum 검증
- test: schema unit test

### T03 — schema validator test

- 상태: `DONE`
- priority: P0
- label: `[FLASH] [FREE]`
- 목표: 누락 필드, 잘못된 범위, synthetic review 규칙, 개인정보 위험 field를 거부한다.
- 관련 파일: `src/ulm/data/schema.py`, `tests/test_schema.py`
- dependency: T02
- 완료 조건: invalid fixture마다 안정적인 오류가 발생함
- test: `pytest tests/test_schema.py`

### T04 — speaker-aware deterministic split

- 상태: `DONE`
- priority: P0
- label: `[LUNA] [FREE]`
- 목표: seed와 ratio로 speaker-disjoint train/validation/test split을 생성하고 검증한다.
- 관련 파일: `src/ulm/data/split.py`, `tests/test_split.py`
- dependency: T02
- 완료 조건: 재실행 동일성, 전체 record 보존, split 간 speaker overlap 0
- test: split unit test

### T05 — AI Hub audit와 위치 tier

- 상태: `DONE`
- priority: P0
- label: `[FLASH] [FREE]`
- 목표: 외부 raw directory를 다운로드하지 않고 실제 JSON 구조를 읽을 수 있는 audit CLI와 `U0/U1/U2/GX` 분류를 제공한다.
- 관련 파일: `src/ulm/data/aihub.py`, `src/ulm/data/filter_ulsan.py`, `scripts/audit_aihub.py`, `scripts/build_ulsan_dataset.py`, `tests/test_aihub.py`, `tests/test_filter_ulsan.py`
- dependency: T02
- 완료 조건: location inventory, speaker/utterance count, duration·누락 통계를 CSV로 출력하고 지정 tier를 canonical JSONL로 추출함
- test: local fixture audit

### T06 — ULM-Bench schema와 baseline metric

- 상태: `DONE`
- priority: P0
- label: `[LUNA] [FREE]`
- 목표: 실제 gold 문항을 대량 생성하지 않고 benchmark item validation과 예측 평가 framework를 만든다.
- 관련 파일: `src/ulm/evaluation/schema.py`, `src/ulm/evaluation/metrics.py`, `scripts/evaluate_benchmark.py`, `benchmarks/`
- dependency: T01
- 완료 조건: fixture prediction으로 task별 metric JSON 생성
- test: benchmark schema·metric test

### T07 — training config loader

- 상태: `DONE`
- priority: P0
- label: `[LUNA] [FREE]`
- 목표: YAML config에서 model, data, QLoRA, precision, checkpoint, seed를 검증하고 resolved snapshot을 저장한다.
- 관련 파일: `src/ulm/training/config.py`, `configs/`
- dependency: T01
- 완료 조건: 필수 필드와 상호 배타 옵션이 검증되고 재현 가능한 dict가 생성됨
- test: config unit test

### T08 — QLoRA SFT adapter

- 상태: `DONE`
- priority: P1
- label: `[LUNA] [FLASH] [GPU]`
- 목표: canonical JSONL을 conversational dataset으로 변환해 `SFTTrainer` 기반 QLoRA를 실행한다.
- 관련 파일: `src/ulm/training/sft.py`, `scripts/train_sft.py`, `configs/sft/`
- dependency: T02, T04, T07
- 완료 조건: ML dependency가 있으면 5~20 step smoke test가 가능하고 없으면 명확히 실패함
- test: message conversion CPU test; GPU smoke는 별도

### T08a — continued pretraining adapter

- 상태: `DONE`
- priority: P1
- label: `[LUNA] [FREE] [GPU]`
- 목표: canonical text를 `Trainer` 기반 next-token prediction QLoRA adapter로 학습하는 실행 경로를 준비한다.
- 관련 파일: `src/ulm/training/cpt.py`, `scripts/train_cpt.py`, `configs/cpt/`
- dependency: T07, T09
- 완료 조건: config·dataset·checkpoint·resume 경로를 SFT와 동일한 정책으로 지원함
- test: `build_cpt_text` CPU test와 `--dry-run`; 실제 학습은 GPU 단계

### T09 — checkpoint/resume

- 상태: `DONE`
- priority: P1
- label: `[LUNA] [FREE] [GPU]`
- 목표: checkpoint path, latest checkpoint 탐색, config snapshot, output overwrite 보호를 구현한다.
- 관련 파일: `src/ulm/training/resume.py`, `src/ulm/training/sft.py`, `tests/test_resume.py`
- dependency: T07, T08
- 완료 조건: 지정/최신 checkpoint resume 경로가 deterministic하고 기존 output을 기본 덮어쓰지 않음
- test: filesystem fixture test

### T10 — local inference CLI

- 상태: `DONE`
- priority: P1
- label: `[FREE] [GPU]`
- 목표: base 또는 adapter model과 `dialect_strength=0~3`을 받아 text inference를 실행한다.
- 관련 파일: `src/ulm/inference/cli.py`, `scripts/infer.py`
- dependency: T08
- 완료 조건: ML dependency 미설치 시 명확한 안내, 설치 시 local model inference 가능
- test: prompt construction CPU test; model inference는 smoke

### T11 — Colab notebook orchestration

- 상태: `DONE`
- priority: P1
- label: `[FLASH] [FREE] [GPU]`
- 목표: notebook이 핵심 로직을 복붙하지 않고 `src/ulm`과 config를 import한다.
- 관련 파일: `notebooks/`
- dependency: T05, T06, T08, T09, T10
- 완료 조건: 환경·audit·smoke·training·evaluation·inference notebook이 실행 순서를 설명함
- test: notebook JSON parse 및 source import 경로 검사

## 외부 상태에 의존하는 항목

### T12 — 실제 AI Hub inventory

- 상태: `BLOCKED`
- priority: P0
- label: `[FREE]`
- 목표: 실제 다운로드본에서 울산 화자·발화·시간·누락·token 통계를 확정한다.
- 관련 파일: `reports/data_audit/`, `data/private/`
- dependency: T05
- 완료 조건: 사용자가 AI Hub 접근 권한과 이용약관을 확인하고 raw archive 경로를 제공한 뒤 audit CSV 생성
- test: T05 local audit test로 코드만 검증
- blocked 이유: AI Hub 로그인·약관 동의·원본 데이터 접근은 자동 수행하지 않음
- 사용자 최소 행동: 이용조건을 확인하고 합법적으로 받은 archive를 로컬 경로에 둔 뒤 audit 명령 실행

### T13 — 0.6B QLoRA smoke

- 상태: `DONE`
- priority: P1
- label: `[GPU] [FREE]`
- 목표: 5~20 step으로 load → tokenize → train → save → resume → eval → inference를 증명한다.
- 관련 파일: `configs/sft/qwen3_0.6b_qlora.yaml`, `notebooks/04_qlora_smoke_test.ipynb`
- dependency: T08, T09, 공개 model download, ML dependencies
- 완료 조건: 실제 log·checkpoint·metric·resume 결과 기록
- test: 무료 Colab 또는 로컬 GPU에서 짧게 실행
- 결과: 로컬 RTX 5060 Laptop GPU에서 5 step 학습, checkpoint 저장, resume, adapter inference 완료. fixture 결과와 실험 로그는 `EXPERIMENTS.md`에 기록함.
- 비고: 1.7B 수 시간 학습은 실행하지 않음

### T14 — 1.7B 연구 실험

- 상태: `READY_FOR_TRAINING`
- priority: P2
- label: `[GPU]`
- 목표: B0/B1/U1/U2/U3 matrix를 실제 데이터와 예산 안에서 실행한다.
- 관련 파일: `configs/cpt/`, `configs/sft/`, `EXPERIMENTS.md`
- dependency: T12, T13, native review
- 완료 조건: 각 결과와 실패가 실측 metric으로 기록됨
- test: 실험별 validation 및 held-out evaluation
- 비고: 유료 GPU instance 생성은 사용자 명시 승인 전 자동 실행하지 않음

### T15 — TTS/STT 통합

- 상태: `REVIEW`
- priority: P3
- label: `[GPU]`
- 목표: 동의·개인정보·정책 검토 후 별도 repository 또는 후속 phase에서 interface를 구현한다.
- 관련 파일: `PLAN.md`, `RESEARCH.md`
- dependency: T14, 음성 수집 동의, 별도 정책 검토
- 완료 조건: 현재 저장소 scope 확장 여부를 재검토하고 승인된 경우에만 설계
- test: 현재는 문서 검토만
