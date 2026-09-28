# 프로젝트 상세 리뷰

검토일: 2026-09-29 · 코드 기준: `a6fbfc1cf7ac943baf197810a2bb4e1b04e19217` · 대상: 최신 공개 main.

앞선 간단 리뷰는 원격 변경을 반영하기 전 `717e346`과 미커밋 파일을 대상으로 했습니다. 이번에는 원격 최신 상태를 반영하고 데이터·학습·평가·추론 연결을 다시 추적했습니다. 기존 간단 리뷰와 실험 파일은 로컬 보존 브랜치에 남아 있습니다.

**판단**

연구용 릴리스 후보로서 코드 분리, 실험 기록, 데이터 계약, 회귀 테스트의 기반은 갖추었습니다. 최신 테스트 136개는 모두 통과했습니다. 다만 데이터 독립성, 추론 취소 시 동시성, 평가 재현성에 아래 결함이 남아 있습니다. 테스트 통과만으로 이 경로까지 보장되지는 않습니다.

이번 작업에서는 저장소 상태와 문서를 정리했습니다. 아래 코드 결함은 재현·분류했으며 수정 완료로 표시하지 않습니다. GPU 학습·실제 모델 평가·외부 배포·데이터 공개 권한 심사는 수행하지 않았습니다.

**동작 흐름과 지켜야 할 조건**

| 경로 | 현재 흐름 | 유지해야 할 조건 |
| --- | --- | --- |
| 데이터 | canonical 검증 → 화자 단위 split → JSONL | 화자와 평가 문항이 학습 split에 섞이지 않음 |
| 학습 | config 검증 → 출력 보호 → 데이터 로딩 → TRL/Trainer → adapter 저장 | 설정과 resume 이력 보존, validation/test 독립성 |
| API | Pydantic 검증 → 방언 강도 prompt 조합 → 토큰 길이 검사 → 생성 스레드 → JSON/SSE | 입력 오류 구분, 한 모델의 생성 직렬화, 종료 확인 |
| 모델 비교 | task별 batch 생성 → 임베딩/표현 proxy → 모델별 결과 → 통합 JSON | 동일 문항·디코딩·평가기, 유효 토큰 수와 실제 provenance |
| 음성 | 패키지의 MMS/VITS 실험 경로 또는 별도 ULM-LIVE | 현재 Live 경로와 과거 패키지 실험을 구분 |

**확인된 결함과 우선순위**

1. **[P1] 외부 validation을 추가하면 원본 JSONL의 내장 split 분리가 생략됩니다.**

   위치: [sft.py](../src/ulm/training/sft.py) 175–176, 197–202행. [cpt.py](../src/ulm/training/cpt.py)도 같은 로더를 사용합니다.

   조건: 단일 학습 JSONL에 train/validation/test 행이 함께 있고 `eval_dataset_path`를 별도로 지정하는 경우. 외부 validation 키가 있으면 `_partition_embedded_splits()`가 즉시 반환해 원본 validation/test 행이 train에 남습니다.

   재현: 실제 `datasets.Dataset`에 세 split 행을 하나씩 만들고 외부 validation을 추가했습니다. 반환된 train의 ID는 `['train', 'validation', 'test']`였습니다. 별도 평가 데이터 없이 분리하면 각 split이 정상 분리됩니다.

   영향: 평가 데이터 독립성을 깨뜨리는 학습 누수. 실제 과거 실험이 이 조합을 사용했는지와 점수 오염 여부는 별도 확인이 필요합니다.

   권장: 내장 split을 먼저 분리하고 외부 validation을 적용합니다. 단일 JSONL+외부 평가 조합에서 train에 validation/test가 남지 않는 회귀 테스트가 필요합니다.

2. **[P2] validation 파일만 있는 평가 디렉터리를 로드하지 못합니다.**

   위치: [sft.py](../src/ulm/training/sft.py) 201행.

   `evaluation_files.get("validation", evaluation_files["train"])`는 기본값 식도 먼저 평가합니다. 평가 디렉터리에 `validation.jsonl`만 있으면 키를 정상 발견해도 `KeyError('train')`이 발생합니다. 임시 디렉터리로 `_load_dataset()`를 호출해 재현했고, 실제 데이터 로더 실행 전에 실패했습니다.

   권장: validation 키가 있는지 명시적으로 분기합니다. 평가 전용 디렉터리·단일 평가 파일·잘못된 디렉터리의 동작을 확인해야 합니다.

3. **[P2] 연결 종료 정리 후에도 이전 생성 스레드가 살아 있을 수 있습니다.**

   위치: [server.py](../src/ulm/inference/server.py) 188행의 생성 잠금, 242–244행의 종료 처리.

   조건: 연결 종료 시 현재 prefill/forward가 2초보다 오래 걸리는 경우. 중단 이벤트는 설정하지만 `join(2.0)` 뒤 스레드 생존 여부를 확인하지 않고 잠금을 해제합니다. 중단 기준은 진행 중인 forward 자체를 즉시 중단시키지 않습니다.

   재현: 실제 엔진에 3초 지연 `generate()` stub을 연결하고 즉시 연결 종료를 반환했습니다. 첫 요청 정리 후 활성 생성은 1개였고, 다음 요청에서 `concurrent_generate_observed=True`였습니다. 모델 다운로드나 GPU는 사용하지 않았습니다.

   영향: 같은 모델에서 생성 작업이 겹쳐 자원 사용량과 지연이 증가할 수 있습니다. 실제 GPU OOM은 검증하지 않았습니다.

   권장: 작업의 실제 종료까지 직렬화 소유권을 유지합니다. 느린 생성의 연결 종료·요청 취소·다음 요청 진입을 함께 검증해야 합니다.

4. **[P2] 토큰 문맥 초과를 서버 생성 장애로 응답합니다.**

   위치: [server.py](../src/ulm/inference/server.py) 156–160, 329–333, 362–364행.

   실제 토큰 길이 검사는 있으나 입력의 `ValueError`를 다른 생성 실패와 함께 처리합니다. 문맥 한도 8, 입력 5토큰, 요청 출력 4토큰인 stub으로 실제 준비 경로와 TestClient를 실행했습니다. 비스트리밍은 HTTP 500, 스트리밍은 HTTP 200 안의 `generation_error`를 반환했습니다.

   영향: 클라이언트가 대화를 줄여야 하는 상황을 서버 장애로 판단해 불필요한 재시도를 할 수 있습니다. SSE 헤더 전송 후 HTTP 상태를 바꿀 수 없으므로 검증 시점도 중요합니다.

   권장: 스트림 헤더 전송 전에 가능한 입력 검증을 완료하고, 문맥 초과를 명확한 4xx 입력 오류로 구분합니다. 실제 토크나이저 길이 검사를 사용하는 테스트가 필요합니다.

5. **[P2] 터미널 대화에서 잘못된 adapter 경로를 조용히 무시합니다.**

   위치: [chat.py](../src/ulm/inference/chat.py) 174행.

   조건: `ulm-chat --adapter /없는/경로`를 지정한 경우. 경로가 없으면 오류 대신 `None`으로 바꿔 base model만 로드합니다. `run_chat`을 mock으로 바꾸고 `main()`을 실행했을 때 전달된 `adapter_path`는 `None`이었습니다.

   영향: 사용자가 튜닝 모델을 평가한다고 생각하면서 base model 출력을 볼 수 있습니다.

   권장: 명시한 adapter 경로가 잘못되면 로딩 전 실패시킵니다. adapter를 생략한 경우와 잘못 지정한 경우를 구분하는 테스트가 필요합니다.

6. **[P2] 비교기의 생성 토큰 수에 조기 종료 후 패딩이 포함됩니다.**

   위치: [run_model_comparison.py](../scripts/run_model_comparison.py) 139–143, 151–156행.

   배치 내 응답은 같은 길이로 패딩되지만 `len(new_tokens)` 전체를 합산합니다. EOS 이후 패딩을 제거하는 것은 decode 출력뿐이며 토큰 수 집계에는 적용되지 않습니다.

   재현: 출력이 `[8, EOS, PAD]`, `[8, 9, EOS]`인 batch stub에서 6토큰으로 집계됐습니다. EOS를 포함하고 종료 뒤 패딩을 제외하면 5토큰입니다.

   영향: `total_tokens_generated`와 `tokens_per_second`가 과대 계산될 수 있습니다. 의미 점수의 오류를 뜻하지는 않습니다. `avg_latency_ms_per_item`도 배치 전체 시간/문항 수이므로 단일 요청 latency나 TTFT로 읽어서는 안 됩니다.

   권장: 최초 EOS까지 실제 생성된 토큰을 집계하고 EOS 포함 여부를 정의합니다. 길이가 다른 응답과 PAD=EOS 모델을 검증해야 합니다.

7. **[P2] 모든 모델을 건너뛰면 통합 결과 JSON을 저장하지 않습니다.**

   위치: [run_all_comparisons.py](../scripts/run_all_comparisons.py) 190–204행.

   `--skip-existing` 경로는 요약을 읽은 뒤 `continue`하므로 통합 파일 저장에 도달하지 않습니다. 기존 모델별 요약 하나와 빈 통합 디렉터리로 `main()`을 실행하니 성공 메시지는 출력됐지만 `summary.json`은 없었습니다. 기존 통합 파일이 있으면 이번에 읽은 모델별 결과로 갱신되지 않습니다.

   권장: 모든 모델의 처리 후 최종 통합 결과를 항상 저장합니다. 전부 skip·일부 skip·통합 파일 부재를 검증해야 합니다.

8. **[P2] 공개 checkout만으로 모델 비교 실행을 재현할 수 없습니다.**

   위치: [run_all_comparisons.py](../scripts/run_all_comparisons.py) 20행과 모델 설정; [run_model_comparison.py](../scripts/run_model_comparison.py) 174행.

   일괄 비교기는 `/home/ubuntu/ULM-1.7B/.venv/bin/python`과 원 EC2의 모델 경로를 고정합니다. 현재 머신에 그 Python 경로가 없음을 확인했습니다. 기본 평가 입력 `data/ulsanbench_v1/benchmark.jsonl`도 공개 main의 추적 파일에 없습니다. 기존 로컬 평가 데이터는 보존 브랜치에 있으며 공개 권한을 검토하지 않아 main에 추가하지 않았습니다.

   권장: 일괄 실행은 현재 인터프리터와 명시적 모델·adapter·dataset 경로를 사용하고, 평가 입력의 입수 절차·버전·체크섬을 기록해야 합니다. 당장은 단일 비교기의 `--dataset`·`--model`·`--adapter`를 명시하는 방법이 있습니다.

**평가와 아키텍처에 대한 추가 관찰**

- 공통 `BenchmarkItem` 평가와 UlsanBench 실험 평가가 서로 다른 task 이름·JSONL 계약을 사용합니다. 오류라고 단정할 수는 없지만 파일을 서로 바꿔 쓰지 않도록 이번 스크립트 안내에 명시했습니다.
- `score_rows()`의 dialectness는 참조 유사도 65%와 marker/ending 유무 35%의 proxy입니다. 단일 표현 포함만으로 보너스를 얻을 수 있고 경상권 공통 어휘도 포함되어 울산 고유성과 자연스러움을 직접 측정하지 않습니다. 공개 README는 이미 proxy 한계를 밝히고 있습니다.
- `run_all_comparisons.py`의 hardware 정보는 원 실험값을 상수로 기록하고 실행일만 현재 날짜로 바꿉니다. 다른 환경 재실행이나 skip 재사용 시 provenance가 어긋날 수 있어 실제 버전·하드웨어와 원 실행일을 수집·보존하는 편이 좋습니다.
- 모델 revision·입력 체크섬·평가기 revision이 통합 설정에 고정되지 않아 같은 모델 ID로 미래에 완전히 같은 결과를 재현하는 데 한계가 있습니다.
- API와 단일 입력 CLI의 강도 조합이 정리됐지만 대화형 CLI는 별도 prompt를 유지합니다. 강도 0에서도 앞부분에 사투리 대화 지시가 붙습니다. 실제 출력 위반은 검증하지 않았으며 공통 helper로 합치는 것이 적절합니다.
- `src/ulm`에는 데이터 계약·출력 보호·prompt·디코딩 정책의 책임이 잘 모여 있습니다. 반면 연구 스크립트에는 경로·생성·집계 코드가 반복됩니다. 모든 과거 스크립트를 일괄 리팩터링하기보다 현재 비교기부터 공통 정책과 데이터 로더를 재사용하는 순서가 적절합니다.
- 저장소 안의 MMS/VITS TTS는 과거 실험 경로이고 현재 Live의 Qwen3-TTS는 별도 저장소입니다. 이번에는 음성 학습의 gradient·품질과 Live/STT/실시간 스트리밍을 검증하지 않았습니다.

**검증 결과**

| 검사 | 결과 |
| --- | --- |
| `.venv/bin/pytest` | 136 passed, 3 warnings, 18.46초 |
| `.venv/bin/ruff check src --output-format concise` | 통과 |
| `.venv/bin/ruff check . --statistics` | 기존 위반 1,345건 |
| 실제 `datasets.Dataset` split 최소 재현 | 외부 validation 존재 시 train에 3종 split이 남음 |
| 평가 전용 디렉터리 최소 재현 | `KeyError('train')` |
| 실제 엔진의 지연 stub 취소 | 정리 후 활성 생성 1개, 후속 생성 중첩 확인 |
| 실제 토큰 준비 경로 + TestClient | 문맥 초과의 비스트리밍 500 / SSE 200+generation_error |
| 대화 CLI `main()` + mock | 없는 adapter를 `None`으로 전달 |
| 비교기의 batch 출력 stub | 패딩 포함 6토큰, 종료 뒤 패딩 제외 시 5토큰 |
| 일괄 비교 `main()` + 임시 요약 | 모든 모델 skip 시 통합 JSON 미생성 |

재현은 `PYTHONPATH=src:scripts .venv/bin/python`으로 CPU·임시 데이터·stub/mock을 사용했습니다. 모델과 평가 encoder를 다운로드하지 않았고 GPU를 사용하지 않았습니다. 생성 취소와 GPU token count 문제는 Python 경로를 검증한 것이며 실제 GPU 품질·성능 재측정을 대체하지 않습니다.

Ruff 전체 위반은 줄 길이 1,030건, 한 줄 복수 구문 173건, 기타 142건입니다. 이전 1,510건과 차이가 나는 이유는 최신 원격 반영과 미추적 실험 파일의 별도 보존이며 코드 스타일을 일괄 수정해 줄인 것이 아닙니다. 핵심 소스는 통과했으므로 현재 실행 스크립트부터 점진적으로 정리하면 됩니다. 테스트 경고는 FastAPI/Starlette 호환 API 2건과 torch JIT deprecation 1건입니다.

**다음 작업 순서**

1. 공통 SFT/CPT 데이터 로더의 split 독립성과 평가 경로 오류 수정.
2. 추론 취소 직렬화·입력 오류 분류·명시 adapter 오류 처리 수정.
3. 모델 비교 토큰 집계·최종 요약 저장·실행 경로와 provenance 정리.
4. 필요한 평가 데이터의 배포 정책과 체크섬 기록, native speaker 검수 설계.
5. 현재 유지하는 스크립트의 Ruff 정리와 회귀 검사 자동화.

릴리스 후보 점수와 모델 비교 점수는 서로 다른 실행 조건을 유지해 해석해야 합니다. 이번 코드 리뷰로 기존 Arm B 가중치나 릴리스 판단이 새로 검증된 것은 아닙니다.
