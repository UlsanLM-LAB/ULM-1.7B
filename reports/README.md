# Reports

현재 public main에는 릴리스 판단과 외부 검증에 직접 필요한 문서만 유지합니다. 중간 학습 로그, 실패 분석 dump, AWS operation record, raw/manual-review JSONL은 Git history 또는 비공개 실험 환경에만 둡니다.

| 구분 | 문서 | 의미 |
| --- | --- | --- |
| Stable Release | [ULM-4B v1.0 Stable Release](ULM_4B_V1_0_STABLE_RELEASE.md) | v1.0 동결 상태와 코드·실모델 검증 |
| Release Notes | [ULM-4B v1.0 Release Notes](ULM_4B_V1_0_RELEASE_NOTES.md) | 배포 요약과 Known Limitations |
| Model Comparison | [ULM-4B Model Comparison](ULM_4B_MODEL_COMPARISON.md) | UlsanBench v2 500문항 비교 |
| Dialect Strength Check | [Real-model strength test](ULM_4B_DIALECT_STRENGTH_REAL_MODEL_TEST.md) | 0/1/2/3 prompt control의 제한된 실모델 확인 |
| Arm B selection | [Arm B Release Candidate](ULM_4B_ARM_B_RELEASE_CANDIDATE.md) | 최종 checkpoint 선택 당시 근거 |
| Machine-readable benchmark | [model-comparison/](model-comparison/) | v1.0에서 고정한 비교 집계·설정 |

자동 semantic similarity·dialectness는 proxy 지표이며 native-speaker human evaluation을 대체하지 않습니다. 과거 내부 실험 산출물은 현재 public release surface에 포함하지 않습니다.
