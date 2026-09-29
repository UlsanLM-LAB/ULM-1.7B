# ULM-Bench

`ULM-Bench`는 울산 지역어 이해·변환·생성·지역 구분·강도 제어와 일반 한국어 보존을 평가하기 위한 benchmark다.

현재 `benchmarks/fixtures/`에는 schema와 CLI를 확인하기 위한 작은 fixture만 있다. fixture의 문장은 실제 울산 화자 gold나 성능 결과가 아니며 연구 점수에 사용하지 않는다.

## 목표 범주

| task | 최종 목표 | 초기 자동 지표 |
|---|---:|---|
| `dialect_understanding` | 250 | accuracy, human semantic score |
| `dialect_to_standard` | 250 | character F-score, meaning score |
| `standard_to_dialect` | 250 | dialect feature precision/recall, native score |
| `region_classification` | 300 | macro-F1, confusion matrix |
| `dialect_chat` | 250 | blind human preference |
| `dialect_strength_control` | 200 | monotonicity, meaning preservation |

최종 목표는 1,000개 이상이지만, 사람 검수와 권리 확보 없이 문항을 자동으로 채우지 않는다. AI Hub 문장을 그대로 공개 benchmark로 재배포하지 않는다.

## Item contract

`src/ulm/evaluation/schema.py`의 `BenchmarkItem`을 사용한다. 각 item은 `source`, `review_status`, `annotators`, `human_verified`를 기록한다. `fixture_only`는 framework 검증 전용 상태다.

## 평가 실행

예측 파일은 한 줄에 다음 형태를 사용한다.

```json
{"id": "fixture-understand-1", "prediction": "B"}
```

```bash
uv run ulm-evaluate \
  --benchmark benchmarks/fixtures/sample_benchmark.jsonl \
  --predictions benchmarks/fixtures/sample_predictions.jsonl
```

자동 metric은 baseline 비교용이다. 울산 화자 authenticity, fluency, 의미 보존은 native blind evaluation이 추가되어야 한다.

## UlsanBench 연구용 비교 입력

위 `BenchmarkItem` fixture는 패키지 계약 검사에 사용합니다. `scripts/run_model_comparison.py`와 `run_all_comparisons.py`가 쓰는 UlsanBench JSONL과는 별도 형식입니다.

UlsanBench의 task는 `comprehension`, `generation`, `identification`, `grammar`, `context`입니다. 각 행은 비어 있지 않은 `prompt`, `reference`, `standard_reference`가 필요하며 `id`를 넣으면 중복 없이 사용합니다. `ending`은 선택적 문자열입니다. identification의 reference는 `ULSAN`, `OTHER_GYEONGSANG`, `STANDARD` 중 하나입니다.

아래는 형식 설명용이며 연구 gold가 아닙니다.

```json
{"id":"format-example","task":"comprehension","prompt":"이 문장의 뜻을 설명해라.","reference":"형식 예시 설명","standard_reference":"형식 예시 설명"}
```

500문항 원 평가셋은 공개 main에 포함되어 있지 않습니다. 원 평가셋의 권한 보유자에게 사용 가능한 버전과 이용 범위를 확인한 뒤 로컬 JSONL을 준비하고 `--dataset`으로 전달해야 합니다. 본 저장소의 예시나 학습 데이터를 원 평가셋처럼 대신 사용하지 않습니다. 실제 버전의 공급 경로가 확보되지 않으면 기존 500문항 점수의 재현도 보장할 수 없습니다.

평가기에서 입력 파일 SHA-256, 실제 문항 수·task별 수, 모델·adapter·encoder revision 또는 로컬 hash, 평가 코드 hash와 실행 환경을 자동 기록합니다. 별도 버전명을 관리할 때는 해당 SHA-256과 대응해 기록하세요. prompt·reference의 내용과 평가 규칙을 바꾸면 새 평가 버전으로 취급해야 합니다.

사람 검수는 자동 proxy 결과와 분리합니다. task별 검수 문항을 고정하고 화자·평가자 정보는 비식별로 관리하며, 모델 이름을 가린 조건에서 의미 보존·자연스러움·울산 지역 적합성을 평가해야 합니다. 검수 대기열 생성만으로 검수 완료나 자연스러움 개선을 주장하지 않습니다.
