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
