# 보고서 안내

현재 릴리스와 과거 연구 실험을 구분해 읽기 위한 목록입니다. 기존 보고서와 집계 JSON의 경로는 연구 이력을 보존하기 위해 유지합니다.

| 구분 | 문서 | 의미 |
| --- | --- | --- |
| 현재 Stable Release | [ULM-4B v1.0](ULM_4B_V1_0_STABLE_RELEASE.md), [Release Notes](ULM_4B_V1_0_RELEASE_NOTES.md) | 기존 Arm B 동결, 코드 수정·검증과 유지된 한계 |
| 최초 checkpoint 선택 | [Arm B Release Candidate](ULM_4B_ARM_B_RELEASE_CANDIDATE.md) | 2026-09-27 당시 선택 근거; 역사 기록 유지 |
| 모델 비교 | [ULM-4B Model Comparison](ULM_4B_MODEL_COMPARISON.md) | 500문항 비교; 릴리스 스냅샷과 별도 실험 |
| 비교 집계 | [config](model-comparison/config.json), [models](model-comparison/models.json), [summary](model-comparison/summary.json) | 당시 평가 설정과 결과 |
| 후속 실험 | [Arm B+](ULM_4B_ARM_B_PLUS_REPORT.md), [Final Low LR](ULM_4B_ARM_B_FINAL_LOW_LR_REPORT.md) | 후속 학습의 결과; 현재 릴리스로 승격된 것을 뜻하지 않음 |
| 코드 상태 | [프로젝트 상세 리뷰](PROJECT_REVIEW.md) | 확인된 결함·재현 조건·검증 범위 |
| 저장소 운영 | [정리 기록](REPOSITORY_CLEANUP.md) | 동기화·브랜치 보존·복구 절차 |

`PHASE3_*`, `PHASE4_*`, `base-model-comparison/`, `instruction-recovery-*`는 단계별 연구 이력입니다. 실행 날짜, 베이스 모델, 어댑터 계보, 디코딩 정책이 다르면 점수를 같은 실험으로 합치지 않습니다. `manual_review_30`은 검수 대기열이며 사람 평가 완료의 증거가 아닙니다.

자동 semantic similarity·dialectness는 proxy 지표입니다. 방언의 자연스러움과 울산 지역 고유성은 별도 화자 검수와 청취 평가가 필요합니다.

v1.0은 추가 학습을 하지 않습니다. Arm B+와 final low-LR 후보는 rejected 연구 결과로만 유지합니다. 기존 집계 JSON의 점수와 원 release config는 변경하지 않았습니다. 패딩 포함 처리량 오류의 Arm B 재검증은 별도 v1.0 보고서와 검증 JSON에 기록하며 기존 runtime 값을 덮어쓰지 않습니다.
