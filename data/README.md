# 데이터 운영 규칙

이 디렉터리는 `ULM-1.7B` 내부 canonical data의 위치와 공개 경계를 설명한다. AI Hub 원본 JSON/WAV와 단순 가공 manifest는 저장소에 넣지 않는다.

## 허용되는 파일

- schema와 validation code
- 재현 가능한 audit·filter·split code
- 구조 검증용 작은 fixture
- 실제 데이터의 집계 통계와 provenance 문서

## 금지되는 파일

- AI Hub 원본 archive, JSON, WAV
- 원본을 거의 그대로 옮긴 JSONL
- 화자 실명·연락처·동의서 원본
- dataset cache와 임시 audio
- license를 확인하지 않은 외부 corpus

## Canonical record

`src/ulm/data/schema.py`의 `DatasetRecord`가 내부 계약이다. `speaker_id`는 pseudonymous ID이며 실명 대응표를 포함하지 않는다. `synthetic`과 `human_verified`를 분리해 기록한다.

학습 전에 다음 순서를 지킨다.

```text
license/consent 확인
  -> raw JSON 구조 audit
  -> location value 전체 확인
  -> U0/U1/U2/GX 분류
  -> text 정규화·중복 점검
  -> speaker-level split
  -> canonical JSONL 생성
```

## AI Hub audit

이미 합법적으로 받은 local archive만 입력으로 사용한다.

```bash
uv run ulm-audit \
  --input-root /path/to/aihub/json \
  --output reports/data_audit/aihub_ulsan_inventory.csv
```

이 명령은 다운로드·로그인·약관 동의를 하지 않는다. 처음에는 출력되는 `location_values`를 확인한 뒤 실제 값에 맞는 normalization 규칙을 사람이 검토해야 한다.

## Dataset group

- `Ulsan Core`: `U0`, `U1`, 검수된 일부 `U2`
- `Gyeongsang Auxiliary`: 울산이 확인되지 않는 경상권 보조 데이터

실제 record 수, audio 시간, token 수, 누락률은 audit 결과가 생성되기 전까지 미정이다.
