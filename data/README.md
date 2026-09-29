# 데이터 운영 규칙

이 저장소는 ULM-4B의 코드와 공개 가능한 작은 fixture만 제공합니다. 실제 학습 corpus, 제한된 benchmark 입력, 원본 음성/JSON 및 단순 파생본은 public Git에 배포하지 않습니다.

## 공개 main에 허용되는 항목

- schema·validation·split 코드
- 데이터 audit/filter 코드
- 구조 검증용 작은 synthetic fixture
- 공개 가능한 집계 통계와 데이터 거버넌스 설명

## public Git에 포함하지 않는 항목

- AI Hub 원본 archive, JSON, WAV 및 이를 거의 그대로 옮긴 파생 파일
- 실제 ULM 학습용 train/validation/test corpus
- 제한된 benchmark 원문과 manual-review queue
- 화자 실명·연락처·동의서 원본
- dataset cache, 임시 audio, 로컬 절대경로와 운영 로그
- 라이선스나 재배포 권한을 확인하지 않은 외부 corpus

`src/ulm/data/schema.py`의 `DatasetRecord`가 canonical record 계약입니다. 실제 데이터를 사용할 때는 license/consent 확인 → 구조 audit → 위치 정규화 → speaker-level split → canonical 변환 순서를 따릅니다.

AI Hub 관련 도구는 이미 합법적으로 확보한 로컬 archive만 입력으로 사용하며 다운로드·로그인·약관 동의를 자동으로 수행하지 않습니다.
