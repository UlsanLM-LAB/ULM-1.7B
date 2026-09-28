# 실행 스크립트 안내

현재 텍스트 릴리스 후보는 ULM-4B Arm B입니다. `src/ulm`은 공통 기능을 소유하고, 이 디렉터리는 실행 진입점과 단계별 연구 실험을 담습니다. 과거 실험의 경로·설정은 해당 보고서와 함께 읽어야 합니다.

| 목적 | 진입점 | 참고 |
| --- | --- | --- |
| 현재 텍스트 API | [serve.py](serve.py), `ulm-serve` | [README의 서버 실행](../README.md#run-arm-b-api), `ULM_MODEL_PATH`·`ULM_ADAPTER_PATH` |
| 단일 입력 추론 | [infer.py](infer.py), `ulm-infer` | `--model-name`과 선택적 `--adapter` |
| API 호환 진입점 | [serve_chat.py](serve_chat.py) | 과거 이름을 유지하는 서버 wrapper |
| 터미널 대화 | `ulm-chat` → [chat.py](../src/ulm/inference/chat.py) | 기본 모델 경로는 과거 1.7B 실험 경로이므로 모델을 명시 |
| 공통 SFT/CPT | [train_sft.py](train_sft.py), [train_cpt.py](train_cpt.py) | [configs/sft](../configs/sft), [configs/cpt](../configs/cpt) |
| 500문항 모델 비교 | [run_model_comparison.py](run_model_comparison.py) | `--dataset`·`--model`·`--output-dir` 명시, CUDA 필요 |
| 과거 EC2 비교 일괄 실행 | [run_all_comparisons.py](run_all_comparisons.py) | 원 실험 머신의 절대 경로 포함; 이식성 한계는 [리뷰](../reports/PROJECT_REVIEW.md) |
| UlsanBench v2 평가 | [evaluate_ulsanbench_v2.py](evaluate_ulsanbench_v2.py) | 임베딩 유사도·방언 표현 proxy; 사람 평가와 구분 |
| 패키지 평가 구조 확인 | `ulm-evaluate` | [benchmarks의 fixture](../benchmarks/README.md) |

`phase3`, `phase4`, `instruction_recovery` 등의 스크립트는 해당 실험의 구축·학습·게이트·회귀 평가 기록입니다. 최신 릴리스 실행 경로와 구분하며, 기존 보고서의 재현성을 위해 이름과 위치를 유지합니다.

UlsanBench 비교에 필요한 `data/ulsanbench_v1/benchmark.jsonl`은 현재 공개 main에 포함되어 있지 않습니다. 권한 있는 평가 데이터를 준비하고 단일 모델 비교기의 `--dataset`으로 경로를 전달해야 합니다. 패키지의 `BenchmarkItem` fixture와 UlsanBench 연구용 JSONL은 서로 다른 형식입니다.

원본 예측·수동 검수 대기열은 로컬 산출물로 보관합니다. `reports/model-comparison/raw/`, 모델별 `predictions.jsonl`, `manual_review_30.jsonl`은 Git에서 제외하며 공개 결과는 집계 요약과 보고서로 관리합니다.
