# 실행 스크립트 안내

현재 텍스트 Stable Release는 ULM-4B v1.0(내부 lineage: Arm B)입니다. `src/ulm`은 공통 기능을 소유하고, 이 디렉터리는 실행 진입점과 일부 역사적 연구 스크립트를 담습니다. 최신 릴리스 실행 경로는 아래 표의 현재 API·추론·평가 진입점을 기준으로 사용하세요.

| 목적 | 진입점 | 참고 |
| --- | --- | --- |
| 현재 텍스트 API | [serve.py](serve.py), `ulm-serve` | [README의 서버 실행](../README.md#run-ulm-4b-v10-api), `ULM_MODEL_PATH`·`ULM_ADAPTER_PATH` |
| 단일 입력 추론 | [infer.py](infer.py), `ulm-infer` | `--model-name`과 선택적 `--adapter` |
| API 호환 진입점 | [serve_chat.py](serve_chat.py) | 과거 이름을 유지하는 서버 wrapper |
| 터미널 대화 | `ulm-chat` → [chat.py](../src/ulm/inference/chat.py) | 기본 모델 경로는 과거 1.7B 실험 경로이므로 모델을 명시 |
| 공통 SFT/CPT | [train_sft.py](train_sft.py), [train_cpt.py](train_cpt.py) | [configs/sft](../configs/sft), [configs/cpt](../configs/cpt) |
| 모델 비교 | [run_model_comparison.py](run_model_comparison.py) | `--dataset`·`--model`·`--output-dir` 명시, CPU/CUDA 선택 |
| 비교 일괄 실행 | [run_all_comparisons.py](run_all_comparisons.py) | `--dataset`·`--models-config` 명시; 현재 Python으로 순차 실행 |
| UlsanBench v2 평가 | [evaluate_ulsanbench_v2.py](evaluate_ulsanbench_v2.py) | `--model`·`--dataset`·`--output` 명시; 기존 방언 표현 proxy 유지 |
| 패키지 평가 구조 확인 | `ulm-evaluate` | [benchmarks의 fixture](../benchmarks/README.md) |
| 전체 lint·회귀 검사 | [check_project.py](check_project.py) | 현재 Python으로 `ruff check src` → `pytest`, 첫 실패 시 중단 |

`phase3`, `phase4`, `instruction_recovery` 등의 스크립트는 역사적 실험 재현용으로 남아 있으며 최신 릴리스 실행 경로가 아닙니다.

UlsanBench 비교에 필요한 `data/ulsanbench_v1/benchmark.jsonl`은 현재 공개 main에 포함되어 있지 않습니다. 권한 있는 평가 데이터를 준비하고 `--dataset`으로 경로를 전달해야 합니다. 패키지의 `BenchmarkItem` fixture와 UlsanBench 연구용 JSONL은 서로 다른 형식입니다. 입력 계약과 준비 절차는 [benchmarks/README.md](../benchmarks/README.md)를 참고하세요.

일괄 실행용 모델 설정 예시:

```json
[
  {
    "id": "ulm-arm-b",
    "display_name": "ULM-4B Arm B",
    "model": "/path/to/base-model",
    "adapter": "/path/to/adapter",
    "decoding_mode": "neutral"
  }
]
```

`model`과 `adapter`는 로컬 디렉터리 또는 Hugging Face ID를 받습니다. base-only 비교에서는 adapter를 생략합니다. Hub의 `revision`·`adapter_revision`을 지정할 수 있으며 실제 사용한 commit을 고정해 기록합니다. 과거 `models.json`의 ID-keyed 객체도 입력할 수 있지만 그 안의 예전 EC2 경로는 현재 위치로 수정해야 합니다.

```bash
uv run python scripts/run_all_comparisons.py \
  --dataset /path/to/benchmark.jsonl \
  --models-config /path/to/models.json \
  --output-dir outputs/model-comparison \
  --device cuda:0 --dtype bfloat16 --batch-size 16
```

CPU에서는 `--device cpu --dtype float32`를 사용합니다. Hub asset의 commit을 조회하거나 모델·평가 encoder를 로드할 때 네트워크 접근이 필요할 수 있습니다. 로컬 모델·adapter 파일은 SHA-256을 계산하므로 시작 시간이 파일 크기에 비례합니다.

기본 출력은 `outputs/model-comparison`입니다. 과거 공개 집계 `reports/model-comparison`을 재실행 기본 출력으로 쓰지 않습니다. 원본 예측과 집계가 모델별로 저장되고, 통합 `summary.json`은 전부 재사용한 경우에도 갱신됩니다.

`--skip-existing`는 입력 SHA-256, 모델·adapter·encoder identity, 디코딩·dtype·batch 설정, 평가 코드, 라이브러리 버전과 기존 메타데이터가 맞는 결과만 재사용합니다. 기존 실행 날짜·하드웨어는 모델별 provenance에 유지하고 이번 집계 시점은 별도로 기록합니다. provenance가 없는 과거 요약이나 불일치가 있으면 GPU 재실행 없이 실패합니다. 재평가가 필요하면 새 출력 디렉터리를 사용하거나 `--skip-existing`를 제거해 명시적으로 실행하세요.

토큰 수는 첫 EOS를 포함하고 EOS 뒤 batch padding을 제외합니다. `avg_latency_ms_per_item`은 batch 총 시간/문항 수이며 단일 요청 응답 지연이나 TTFT가 아닙니다.

원본 예측·수동 검수 대기열은 로컬 산출물로 보관합니다. `reports/model-comparison/raw/`, 모델별 `predictions.jsonl`, `manual_review_30.jsonl`은 Git에서 제외하며 공개 결과는 집계 요약과 보고서로 관리합니다.
