# ULM-4B v1.0 Stable Release

2026-09-29 (Asia/Seoul) · `v1.0.0` · 내부 checkpoint lineage: **Arm B**

## Decision and scope

기존 Instruction Recovery v1 Arm B를 **ULM-4B v1.0**으로 동결합니다. 추가 학습·weight merge·학습 데이터 변경을 하지 않았으며 추가 학습 계획도 없습니다. rejected Arm B+와 final low-LR 후보는 연구 기록으로만 유지합니다. 최초 [Arm B 선택 보고서](ULM_4B_ARM_B_RELEASE_CANDIDATE.md), 원 release config, 기존 평가 JSON은 보존했습니다.

v1.0의 Stable 판단은 릴리즈 경로의 9개 코드 결함 수정과 CPU/실제 모델 검증에 근거합니다. Instruction-trap 65%와 identification 32%, human evaluation 부재는 알려진 품질·용도 한계로 남습니다. 모든 역사적 연구 gate 통과나 모든 production 용도의 적합성을 주장하지 않습니다.

v1.0.0 릴리스는 별도 release branch에서 검증 후 병합했습니다. 릴리스 당시 모델 가중치와 평가 수치는 변경하지 않았으며, 현재 main의 public-hygiene 정리는 모델 동작이나 v1.0.0 tag를 변경하지 않습니다.

## Review revalidation

사전 코드 리뷰에서 추적한 9개 release-path 항목은 v1.0에서 **모두 fixed**입니다.

| 항목 | 수정 전 재현 | 수정 후 증거 |
| --- | --- | --- |
| Split 누수 | 외부 validation 존재 시 embedded validation/test가 train에 포함 | 실제 Dataset에서 먼저 분리, 외부 validation 교체, 원 test 유지; ID·화자 overlap과 동일 unsplit 파일 재사용 거부. SFT/CPT 공통 로더. |
| Validation-only directory | eager default로 KeyError(train) | validation.jsonl·dev.jsonl·train.jsonl 성공; test-only·split 없는 directory 오류. |
| Cancel 동시성 | 2초 join 종료 뒤 살아 있는 worker와 후속 generate 중첩 | 느린 forward barrier, disconnect·반복 cancel·stream 종료 후 cancel 모두 최대 active generate=1, thread 종료까지 lock 유지. |
| Context overflow | JSON 500 / SSE 200 generation_error | 입력+출력 예산을 헤더 전 검사; 두 endpoint × 두 mode 모두 400 invalid_request_error, generate 호출 없음. |
| Invalid adapter | 명시 missing 경로를 None으로 바꿔 base fallback | missing·file·빈 문자열은 로딩 전 오류; 생략/valid 경로 유지. |
| Token accounting | 길이가 다른 두 출력의 EOS 뒤 padding까지 6개 | 실제 CPU runner 5개, PAD=EOS·multi EOS·limit/no EOS; 실제 500문항 재검증 아래 기록. |
| Skip summary | all-skip이면 통합 JSON 없음 | all/partial/fresh 및 후속 실패 보존, atomic save, stale 결과의 GPU 자동 재실행 거부. |
| Paths/provenance | EC2 Python/model/dataset 기본값과 상수 hardware | 현재 Python과 명시적 입력, outputs 기본 경로, checksum·resolved revision·실제 hardware/software/date와 기존 원 실행 provenance 유지. |
| CLI dialect strength | 강도 0에도 별도 사투리 기본 지시 | API 공통 helper 재사용, 기본2 및 0/1/2/3 실제 CLI stub/API 회귀, history style 비누적. |

내부 `ChatEngine` 주입 구현은 `validate_request()`가 필요합니다. 정상 요청의 CPU tokenization은 사전검사와 생성 준비에서 두 번 실행되며 device 전송은 생성 시에만 합니다. 중단 불가능한 forward가 계속되면 다음 요청이 기다리는 것이 직렬화 보장입니다.

## Verification

| 검사 | 결과 |
| --- | --- |
| 전체 CPU/offline pytest | **230 passed**, 3 dependency deprecation warnings, 10.10s |
| Focused regression | **148 passed**, 2 warnings, 5.19s |
| `ruff check src` | pass |
| 수정된 비교 runner·새 회귀 테스트 Ruff | pass |
| `git diff --check` | pass |
| `uv lock --offline --check` | pass; dependency pins 유지, package version만 1.0.0 |
| 기존 release contract | training/inference/evaluation/lineage exact equality, 원 JSON SHA-256 유지 |
| Evaluator 비교 | score_rows와 scoring/generation helper의 AST가 기존 main과 동일; 산식 변경 없음. GPU 검증 뒤 독립 평가 CLI의 model/dataset 입력만 정리하고 CPU 회귀로 검증 |
| 실제 모델 HTTP smoke | health + 2 endpoint × JSON/SSE × 강도0–3 **16요청 성공** |
| 실제 affected runtime | Neutral/Context Guard 각500문항, 출력1,000개 모두 historical raw와 exact equality |
| Weights freeze | base14파일·adapter7파일 디렉터리 SHA 검증 전후 동일, 모든 모델 파라미터 non-trainable |
| AWS | 검증 전후 stopped 상태 확인; 학습 및 instance terminate 없음 |

CPU/offline 명령:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
CUDA_VISIBLE_DEVICES=-1 .venv/bin/python scripts/check_project.py

HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
CUDA_VISIBLE_DEVICES=-1 .venv/bin/pytest \
  tests/test_training_data.py tests/test_server.py tests/test_chat.py \
  tests/test_prompt.py tests/test_model_comparison.py tests/test_release_config.py
```

경고는 Starlette/httpx·AnyIO·torch JIT의 deprecation이며 실패·skip된 검사는 없습니다. 관련 없는 전체 연구 스크립트 Ruff 정리는 수행하지 않았습니다.

## Frozen artifacts and actual model smoke

실제 모델 검증은 AWS EC2 `g6e.xlarge` / NVIDIA L40S에서 기존 frozen base와 Arm B adapter만 로드해 수행했습니다. Hub offline 조건에서 새 모델 다운로드·학습·weight merge 없이 검증했습니다.

- Base revision: `c83cb7aa2999d2f35c43e9ae0634a30eb8985a1e`
- Base `model.safetensors` SHA-256: `e73742841e31c04332303abf6c0e730691f8d7a08ed0a9a55496f646d39d1f29`
- Arm B `adapter_model.safetensors` SHA-256: `fee03f872aaff55283be0413536e20b0930334333a20c6ee01cad63f57f43716`
- 기존 Arm B snapshot과 hash가 같으며 검증 전후 base/adapter 전체 디렉터리 hash도 같음
- HTTP JSON/SSE와 Live `token` 필드 유지; 강도0의 smoke 출력은 표준어 `모르면 모른다.`

Release-time raw verification JSON에는 로컬 경로와 운영 메타데이터가 포함되어 현재 public main에서는 제외했습니다. 필요한 핵심 digest·요청 수·검증 결과는 이 문서와 v1.0.0 release snapshot에 남아 있습니다. HTTP smoke는 텍스트 통합 검사이며 human evaluation이나 실제 Live TTS/STT 검증으로 해석하지 않습니다.

## Affected token count and runtime

원 500문항 입력의 SHA-256:

```text
dcb6df1da91287efd14f6c22b7ccaea162d493f90529ed1479ed834277343190
```

동일 기존 Arm B, bf16, batch16, 공식 chat template와 기존 neutral/context-only-guard 생성 조건으로 실행했습니다. EOS 뒤의 padding만 제거하며 첫 EOS는 생성 토큰에 포함합니다.

| 이번 재검증 | Neutral | Context-only Guard |
| --- | ---: | ---: |
| 문항 수 | 500 | 500 |
| 기존 집계 방식으로 센 이번 run 토큰 | 16,184 | 13,913 |
| 올바른 생성 토큰 | **8,271** | **8,005** |
| 제외한 EOS 뒤 padding | 7,913 | 5,908 |
| 실제 inference wall time | 90.29s | 79.47s |
| 실제 토큰 기준 throughput | **91.61 tok/s** | **100.73 tok/s** |
| historical raw와 다른 출력 | **0** | **0** |

과거 throughput 174.92/176.44 tok/s 등은 역사 기록으로 남기며 v1.0의 검증된 처리량으로 사용하지 않습니다. 위 값은 별도 검증 실행의 측정값이며 다른 비교 모델의 처리량은 이번에 재측정하지 않았습니다.

이번 runtime 검증은 HTTP smoke 뒤 같은 프로세스에서 수행했습니다. 기록된 peak allocated VRAM 17.17GiB에는 그 프로세스의 메모리 조건이 반영되므로 과거 독립 비교 프로세스의 9.27GiB와 직접 비교하지 않습니다. Avg latency는 batch wall time / 문항 수이며 단일 요청 latency나 TTFT가 아닙니다. Python3.11.16, torch2.14.0+cu130, Transformers5.17.0, PEFT0.20.0, NVIDIA driver595.91.07, torch CUDA13.0을 사용했습니다.

출력과 scorer가 같으므로 기존 semantic/dialectness/identification 수치를 유지합니다. 스코어를 좋게 보이게 바꾸거나 새 사람 평가로 표기하지 않았습니다.

## Known Limitations and non-blocking scope

- Instruction-trap 65%: 원 연구 instruction gate는 fail인 채 유지합니다.
- Identification: release snapshot32%, 별도 neutral comparison30%; 안정적인 지역 구분을 보장하지 않습니다.
- Native speaker evaluation과 voice listening evaluation은 **still open**입니다. 자동 proxy가 이를 대체하지 않습니다.
- Prompt strength 제어, 환각, 울산 지역·세대·화자 변이는 모델 카드에 유지합니다.
- 제한 benchmark 입력의 외부 배포는 **partially fixed**: checksum/준비 절차는 기록했으나 공개 이용 권한을 새로 승인하지 않았습니다.
- Live TTS/STT·speech interruption과 full deployed-mode500 quality rerun은 수행하지 않았습니다.

사용자가 지정한 재학습 없는 research/demo 제품 동결의 알려진 한계이며, 수정된 코드 경로를 사용하는 것을 막는 unresolved release blocker로 분류하지 않습니다. 모델 품질 문제가 해결됐다고 주장하지 않습니다.

## Changed files

- Training/inference: `src/ulm/training/sft.py`, `src/ulm/inference/{server,chat,cli,prompt,adapters}.py`
- Benchmark: `src/ulm/evaluation/comparison.py`, `scripts/{run_model_comparison,run_all_comparisons,evaluate_ulsanbench_v2}.py`
- Verification: `scripts/check_project.py`, `tests/{test_training_data,test_server,test_chat,test_prompt,test_model_comparison,test_release_config}.py`
- Metadata/contract: `src/ulm/__init__.py`, `pyproject.toml`, `uv.lock`, `configs/release/ulm4b-v1.0.json`, `CITATION.cff`
- Docs: `README.md`, `MODEL_CARD.md`, `benchmarks/README.md`, `scripts/README.md`, `reports/README.md`, `reports/ULM_4B_MODEL_COMPARISON.md`, this report and release notes

기존 `configs/release/ulm4b-arm-b.json`, v1.0 release contract와 고정된 benchmark snapshot은 유지합니다. 현재 main에서는 오래된 SFT/CPT 실험 config, 학습 corpus, 중간 실험 dump를 공개 표면에서 제외했으며 v1.0.0 tag의 모델·릴리스 snapshot은 변경하지 않습니다.

## Git and release procedure

별도 브랜치의 의미 있는 commit과 전체 diff self-review, 테스트 결과를 포함한 PR을 통해 mergeable 상태에서 squash merge합니다. 병합 후 main이 원격과 같고 working tree가 깨끗하며 열린 PR/blocker가 없는 것을 확인한 뒤 annotated tag `v1.0.0`과 Stable GitHub Release **ULM-4B v1.0**을 생성합니다. 최종 SHA와 release URL은 GitHub tag/release와 최종 작업 보고를 기준으로 확인할 수 있습니다.

릴리스 이후 repository hygiene는 별도 PR로 수행하며 public main에는 외부 사용·검증에 필요한 자료만 유지합니다. 내부 실험 백업과 운영 기록은 public tree와 분리합니다.

## License and citation

코드는 [Apache-2.0](../LICENSE). [Base model card](https://huggingface.co/empero-ai/Qwen3.8-4B-Distill)는 Apache-2.0을 표시하며 모델·데이터·encoder·TTS·파생 가중치의 upstream notice와 이용 조건은 별도로 유지됩니다. 가중치는 이 코드 릴리즈에 포함하지 않습니다.

[CITATION.cff](../CITATION.cff), [README Citation](../README.md#citation), [References](../README.md#references)를 사용합니다. 저장소 URL은 `https://github.com/UlsanLM-LAB/ULM-4B`입니다.
