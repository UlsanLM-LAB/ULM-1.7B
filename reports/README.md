# Reports

공개 릴리스에 필요한 검증·벤치마크 문서만 유지합니다. 내부 계획서, 작업 로그, 중간 단계별 분석 문서는 현재 release surface에서 제외했습니다.

| 구분 | 문서 | 의미 |
| --- | --- | --- |
| Stable Release | [ULM-4B v1.0 Stable Release](ULM_4B_V1_0_STABLE_RELEASE.md) | v1.0 동결 상태와 코드·실모델 검증 |
| Release Notes | [ULM-4B v1.0 Release Notes](ULM_4B_V1_0_RELEASE_NOTES.md) | 배포 요약과 Known Limitations |
| Model Comparison | [ULM-4B Model Comparison](ULM_4B_MODEL_COMPARISON.md) | UlsanBench v2 500문항 모델 비교 |
| Arm B selection | [Arm B Release Candidate](ULM_4B_ARM_B_RELEASE_CANDIDATE.md) | 최종 checkpoint 선택 당시의 근거 |
| Machine-readable results | [model-comparison/](model-comparison/) | 비교 설정·모델 메타데이터·집계 결과 |

자동 semantic similarity·dialectness는 proxy 지표이며 native-speaker human evaluation을 대체하지 않습니다. v1.0은 기존 Arm B를 재학습 없이 동결한 릴리스입니다.

과거 내부 계획·중간 실험 문서는 Git history에는 남아 있을 수 있지만 현재 공개 릴리스 문서 목록에는 포함하지 않습니다.
