# 실행 스크립트 안내

현재 공개 main은 ULM-4B v1.0의 릴리스·평가·서빙에 필요한 진입점만 유지합니다. 단계별 학습 실험, 복구 orchestration, 임시 진단·모니터링 스크립트는 Git history에 남기고 현재 public surface에서는 제외했습니다.

| 목적 | 진입점 | 참고 |
| --- | --- | --- |
| 텍스트 API | [serve.py](serve.py), `ulm-serve` | `ULM_MODEL_PATH`와 `ULM_ADAPTER_PATH` 사용 |
| API 호환 wrapper | [serve_chat.py](serve_chat.py) | 기존 서버 진입점 호환 |
| 단일 입력 추론 | [infer.py](infer.py), `ulm-infer` | base 또는 adapter 지정 |
| 데이터 audit | [audit_aihub.py](audit_aihub.py), `ulm-audit` | 합법적으로 확보한 로컬 데이터만 입력 |
| 데이터 구축 | [build_ulsan_dataset.py](build_ulsan_dataset.py), `ulm-build-dataset` | 공개 저장소에 원본/파생 학습 corpus를 포함하지 않음 |
| 공통 SFT/CPT | [train_sft.py](train_sft.py), [train_cpt.py](train_cpt.py) | 사용자가 준비한 config와 데이터 경로를 명시 |
| 패키지 benchmark | [evaluate_benchmark.py](evaluate_benchmark.py), `ulm-evaluate` | fixture 기반 계약 검사 |
| UlsanBench v2 | [evaluate_ulsanbench_v2.py](evaluate_ulsanbench_v2.py) | 권한 있는 benchmark JSONL 필요 |
| 단일 모델 비교 | [run_model_comparison.py](run_model_comparison.py) | model/adapter/dataset 경로 명시 |
| 비교 일괄 실행 | [run_all_comparisons.py](run_all_comparisons.py) | 모델 설정 JSON과 dataset 명시 |
| benchmark chart | [generate_benchmark_charts.py](generate_benchmark_charts.py) | 집계 결과 시각화 |
| lint·CPU 회귀 | [check_project.py](check_project.py) | `ruff check src` 후 `pytest` |

UlsanBench 500문항 원 입력은 공개 main에 포함하지 않습니다. 공개 fixture와 실제 연구 평가셋은 서로 다른 데이터 계약이며, 권한이 확인된 입력을 `--dataset`으로 전달해야 합니다.

모델 비교의 기본 출력은 `outputs/model-comparison`입니다. 원본 predictions, manual-review queue, training logs, AWS operation records는 public Git에 커밋하지 않습니다.
