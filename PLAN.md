# ULM-1.7B 개발 계획

## 1. 프로젝트 목표

`UlsanLM Lab`의 `ULM-1.7B`는 약 1.7B 파라미터 규모의 울산 지역어 특화 Small Language Model 연구 프로젝트다. 목표는 다음 기능을 한 모델과 평가 체계 안에서 검증하는 것이다.

- 울산 지역어 이해
- 울산 지역어에서 표준어로의 변환
- 표준어에서 울산 지역어로의 변환
- 자연스러운 울산 지역어 대화
- 울산 고유 표현 이해
- 가능한 범위에서 울산·부산·대구 지역 구분
- `dialect_strength=0~3` 제어
- 일반 한국어 능력 유지
- 양자화 후 로컬 실행
- 재현 가능한 학습·평가 pipeline

이 저장소의 현재 범위는 텍스트 모델 `ULM-1.7B`다. TTS와 STT는 장기 생태계의 interface와 데이터 거버넌스만 문서화하고, 현재 구현 범위에는 포함하지 않는다.

## 2. 문제 정의

공개 경상권 방언 데이터 전체를 울산 데이터로 간주하면 지역성, 데이터 provenance, 평가 타당성이 무너진다. 따라서 모델 학습보다 먼저 실제 화자 metadata를 audit하고, 울산 출생·성장·거주 조건을 구분한 `U0/U1/U2/GX` tier와 화자 단위 split을 구축한다.

이 프로젝트의 성공은 출력에 사투리 어미가 들어가는지로만 판단하지 않는다. 의미 보존, 울산 화자 평가, 지역 구분, 강도 제어, 일반 한국어 보존을 함께 측정한다. `deep-research-report.md`는 이 판단의 primary research source이며, 아직 다운로드하지 않은 외부 데이터의 크기나 성능을 사실처럼 사용하지 않는다.

## 3. Research Questions

1. 1.7B급 모델이 narrow domain인 울산 지역어에서 더 큰 범용 모델과 경쟁 가능한가?
2. `Ulsan Core`가 일반 `Gyeongsang Auxiliary`보다 얼마나 효과적인가?
3. `SFT`와 `CPT + SFT`의 차이는 무엇인가?
4. 검수된 synthetic data 비율에 따라 dialect authenticity가 어떻게 변하는가?
5. 울산·부산·대구 지역 구별이 가능한가? 구별 불가능한 shared 표현은 어떻게 표시해야 하는가?
6. `dialect_strength`를 의미 보존과 함께 안정적으로 제어할 수 있는가?

## 4. 설계 원칙

- 데이터가 없으면 데이터가 있다고 가정하지 않는다.
- AI Hub 원본, audio, checkpoint, cache, secret은 Git에 넣지 않는다.
- `speaker_id`는 학습용 manifest에서 pseudonymous ID로 취급하고, 실명 대응표는 별도 보안 영역에 둔다.
- 같은 화자의 발화는 train·validation·test에 섞지 않는다.
- 검수되지 않은 synthetic dialect data는 주 데이터로 사용하지 않는다.
- benchmark는 학습보다 먼저 schema와 평가 함수를 만들고, 실제 문항은 사람 검수 workflow를 거친다.
- GPU가 없어도 data contract, validation, split, metric, config, CLI가 동작해야 한다.
- 고정된 값은 코드에 흩뿌리지 않고 config에 둔다. 다만 불필요한 framework와 abstraction은 추가하지 않는다.

## 5. Architecture

```text
원본 경로(외부·비공개)
  -> AI Hub audit / 위치 정규화
  -> Ulsan tier filter / license gate
  -> canonical JSONL schema
  -> speaker-aware deterministic split
  -> SFT messages 또는 CPT text
  -> optional QLoRA trainer
  -> ULM-Bench evaluator
  -> local inference
```

구현은 다음 세 층으로 단순화한다.

1. `ulm.data`: schema, JSONL 입출력, AI Hub audit, 위치 정규화, speaker split
2. `ulm.evaluation`: benchmark schema, fixture, task별 baseline metric
3. `ulm.training` 및 `ulm.inference`: optional ML dependency를 lazy import하는 실행 adapter

핵심 data·metric 로직은 PyTorch나 Hugging Face를 import하지 않는다. 그러므로 CPU 환경에서 validation과 test를 실행할 수 있다. 모델 학습만 `transformers`, `datasets`, `peft`, `trl`, `bitsandbytes`, `torch` extra에 의존한다.

## 6. Model strategy

| 역할 | 기본 모델 | 목적 | 현재 판단 |
|---|---|---|---|
| smoke test | `Qwen/Qwen3-0.6B` | dataset loading부터 5~20 step까지 검증 | 우선 실행 |
| 빠른 SFT | `Qwen/Qwen3-1.7B` | MVP dialect instruction tuning | 준비 |
| 연구 주 모델 | `Qwen/Qwen3-1.7B-Base` | `CPT -> SFT` 해석 가능한 비교 | 최종 후보 |
| 선택 비교 | `Gemma 3 1B`, `SmolLM3-3B` | 선택적 multilingual/open-training baseline | 후순위 |

`Qwen3-1.7B-Base`와 post-trained `Qwen3-1.7B`를 같은 모델 family 안에서 비교한다. Apache-2.0 여부, 한국어 tokenizer 효율, `QLoRA`, `transformers`, `PEFT`, `TRL`, `GGUF`, `llama.cpp` 호환성은 실제 실험 시작 시 사용 버전의 model card와 license를 다시 기록한다. 연구 보고서의 추천은 시작점이지 최적값이나 울산 성능 보장이 아니다.

## 7. Dataset strategy

### 7.1 내부 canonical schema

각 JSONL record는 최소 다음 필드를 가진다.

`id`, `source`, `task`, `speaker_id`, `birthplace`, `raised_region`, `current_region`, `age_group`, `gender`, `dialect_text`, `standard_text`, `dialect_strength`, `synthetic`, `human_verified`, `quality_grade`, `split`, `metadata`

`speaker_id`는 raw 이름이 아닌 pseudonymous ID여야 한다. 원본 sample ID, license 상태, audit 정보처럼 공개할 수 없는 provenance는 `metadata`에 저장하되 공개 파생 데이터와 분리한다.

화자 tier는 다음 정책을 사용한다.

- `U0`: 울산 출생이고 울산 주 성장/주 거주가 확인됨
- `U1`: 울산 주 성장/주 거주와 현재 울산 거주가 모두 확인됨
- `U2`: 출생·성장·현재 거주 중 하나만 울산으로 확인됨
- `GX`: 울산 확인이 없는 경상권 보조 화자

실제 위치 값은 audit 전에 확정하지 않는다. `울산`, `울산광역시`, `울산광역시 ...`를 지원하되, `중구` 같은 구 이름만으로 울산을 판단하지 않는다.

최소 dataset group은 `Ulsan Core`와 `Gyeongsang Auxiliary`로 분리한다. 원본 다운로드와 약관 동의가 필요한 audit는 이 저장소에서 자동 수행하지 않는다.

### 7.2 Synthetic policy

synthetic record는 `synthetic=true`로 표시한다. native review가 끝난 경우에만 `human_verified=true`로 설정할 수 있으며, 두 값은 동시에 true일 수 있다. 검수되지 않은 synthetic sample은 benchmark gold와 핵심 test set에 사용할 수 없다.

### 7.3 Leakage 방지

split은 `speaker_id` 단위로 deterministic하게 생성하고, seed와 비율을 manifest metadata에 기록한다. validator는 split별 speaker 집합의 교집합이 비어 있는지 검사한다. 이후 규모가 충분해지면 session·conversation·near-duplicate family 단위 grouping을 추가한다.

## 8. Training strategy

실험 순서는 다음과 같다.

1. `B0`: base model baseline
2. `S0`: `Qwen3-0.6B` 5~20 step QLoRA smoke test
3. `U1`: 1.7B Ulsan SFT / QLoRA
4. `A1`: dataset size ablation
5. `A2`: `Ulsan Core` 대 `Ulsan Core + Gyeongsang Auxiliary`
6. `U2`: `Qwen3-1.7B-Base` continued pretraining
7. `U3`: CPT + SFT
8. `C1`: `dialect_strength` conditioning

SFT는 `messages` 형식으로 변환하고, 기본값은 4-bit NF4, double quantization, 가능하면 BF16, `target_modules=all-linear`인 QLoRA다. report에 기재된 `learning_rate=1e-4`, `max_seq_length=1024`, `lora_r=32` 등은 탐색 시작값으로만 기록한다.

CPT는 usable 울산 token 수를 먼저 측정한다. `1M` 미만이면 CPT를 주 실험으로 확대하지 않고 ablation 수준으로 제한한다. CPT를 실행할 경우 general Korean validation과 mix를 별도로 둬 catastrophic forgetting을 확인한다.

모든 실행은 resolved config snapshot, seed, dataset version, commit, runtime metadata, checkpoint path를 output directory에 저장한다. `resume_from_checkpoint`는 명시 경로 또는 가장 최신 checkpoint를 지원한다.

## 9. Evaluation

`ULM-Bench`의 최종 목표는 1,000개 이상이지만, 초기 구현은 schema와 작은 fixture만 포함한다. 공개 test는 AI Hub 문장을 그대로 재배포하지 않고, 권리를 확보한 문장과 원어민 검수 문장으로 별도 구축한다.

초기 task는 `dialect_understanding`, `dialect_to_standard`, `standard_to_dialect`, `dialect_chat`, `region_classification`, `dialect_strength_control`이다.

- 이해·지역 분류: accuracy, macro-F1
- 울산→표준어: character F-score와 semantic human score
- 표준어→울산: dialect feature precision/recall과 native authenticity
- 강도 제어: feature density monotonicity와 meaning preservation
- 일반 한국어: 별도 retention set의 정확도와 regression

자동 metric은 baseline 비교용이며 “사투리다움”을 단독으로 판정하지 않는다. 사람 평가가 가능한 단계에서는 울산 화자 blind evaluation과 참가자 단위 confidence interval을 사용한다.

## 10. Google Colab

Colab notebook은 orchestration만 담당하고 `src/ulm`을 import한다.

권장 흐름은 GPU 확인 → dependency 설치 → Drive mount → `/content`에 archive 복사 → audit/validation/split → baseline → QLoRA → checkpoint backup → benchmark evaluation → adapter export다.

Drive에는 작은 파일 수천 개를 직접 읽게 하지 않고 archive 또는 Parquet를 복사해 `/content`에서 처리한다. `save_steps`, `eval_steps`, `save_total_limit`, `seed`, `output_dir`, `resume_from_checkpoint`를 config로 제어한다. Colab의 정책과 runtime 종료 가능성을 고려해 TTS voice cloning은 현재 notebook에 넣지 않는다.

## 11. Local inference

학습된 base 또는 adapter를 `ulm.inference` CLI로 불러온다. `--dialect-strength 0~3`을 system/user prompt의 control로 변환하고, 출력은 text로 저장한다. GGUF 변환과 `llama.cpp` 실행은 adapter 품질이 검증된 후 별도 작업으로 둔다.

장기적으로는 다음 interface를 둔다.

```text
Speech -> STT -> ULM-1.7B -> text/prosody profile -> TTS -> Speech
```

현재는 TTS 내부 구현이나 임의 voice cloning API를 만들지 않는다. 공개 demo를 만들 때도 동의받은 고정 speaker profile만 선택하도록 한다.

## 12. Repository structure

```text
ULM-1.7B/
├── deep-research-report.md
├── README.md
├── PLAN.md / TODO.md / ROADMAP.md
├── RESEARCH.md / EXPERIMENTS.md / GPU.md / WORKLOG.md
├── LICENSE / pyproject.toml / .gitignore
├── configs/
│   ├── cpt/
│   └── sft/
├── data/
│   ├── README.md
│   ├── examples/
│   └── private/
├── benchmarks/
│   ├── README.md
│   └── fixtures/
├── notebooks/
├── scripts/
├── src/ulm/
│   ├── data/
│   ├── evaluation/
│   ├── inference/
│   ├── training/
│   └── utils/
└── tests/
```

필요한 경우에만 파일을 추가한다. `Manager`, `Factory`, `Repository`, dependency injection, microservice, plugin framework 계층은 도입하지 않는다.

## 13. Phase dependency

```text
문서·license 경계
  -> package/config
  -> schema/validation
  -> speaker split/audit
  -> benchmark/metric
  -> 0.6B smoke
  -> 1.7B SFT
  -> CPT/ablation
  -> human review/evaluation
```

데이터 audit가 외부 로그인·약관 동의를 요구하면 해당 단계만 보류하고, audit 입력을 받기 전까지 fixture와 코드 검증을 계속한다.

## 14. Risks and license

- AI Hub의 실제 울산 화자 수와 원본 key는 다운로드 후에만 확정된다.
- AI Hub 원본과 단순 가공 manifest는 재배포하지 않는다.
- 모델, adapter, merged model, benchmark, 직접 수집 voice는 각각 license와 consent를 따로 기록한다.
- 모델 license가 Apache-2.0이어도 데이터·화자 권리까지 해결되는 것은 아니다.
- synthetic data는 실제 울산 화자 발화와 동일한 증거로 취급하지 않는다.
- 작은 데이터에서 CPT를 강행하면 일반 한국어 저하와 과적합이 생길 수 있다.
- 지역 판별이 불가능한 공통 표현을 억지로 지역 gold로 만들지 않는다.
- TTS는 음성 개인정보와 impersonation 위험 때문에 현재 범위에서 실행하지 않는다.

프로젝트 코드의 기본 license는 Apache-2.0이다. 외부 모델·데이터 license는 별도 확인 후 model card와 dataset card에 기록한다.

## 15. Definition of Done

### 코드

- CPU 환경에서 schema, JSONL, split, benchmark metric, config loader가 재현 가능하게 동작한다.
- schema invalid record와 speaker leakage를 테스트로 거부한다.
- AI Hub audit CLI가 실제 raw directory를 입력으로 받아 inventory CSV를 생성한다.
- ML dependency가 없을 때 학습 CLI가 설치 요구사항을 명확히 설명한다.
- QLoRA script가 config, checkpoint save, resume, snapshot을 지원한다.
- inference CLI가 base/adapter와 `dialect_strength`를 받아 실행한다.

### 연구 운영

- 실제 데이터의 Ulsan tier·token·누락 통계가 별도 audit 결과로 존재한다.
- benchmark 공개 문항은 권리와 native review 상태를 기록한다.
- B0/S0/U1/U2/U3 비교에서 결과와 실패를 `EXPERIMENTS.md`에 남긴다.
- 학습 결과를 만들지 못한 단계는 `READY_FOR_TRAINING` 또는 `BLOCKED`로 명시한다.

## 16. Self-review 결과

- **scope**: 1.7B text SLM과 재현 가능한 pipeline으로 제한했다. TTS/STT는 interface만 남겼다.
- **비용**: 로컬 CPU와 무료 Colab 우선이며 유료 GPU·외부 결제는 자동 실행하지 않는다.
- **leakage**: speaker-level split을 schema 운영의 기본 경계로 두었다. template/near-duplicate grouping은 데이터 audit 후 확장한다.
- **license**: AI Hub 원본·파생 manifest를 Git에서 제외하고, 모델·데이터·voice 권리를 분리한다.
- **synthetic**: source flag와 human review flag를 분리했다. 대량 synthetic 생성은 하지 않는다.
- **evaluation**: 학습 전에 benchmark schema와 metric을 만든다.
- **resume**: config와 checkpoint 경로를 명시하고 output overwrite를 기본 금지한다.
- **구조**: pure Python core와 optional ML adapter만 두어 Gemini Flash가 이어받기 쉽게 했다.
- **재현성**: seed, config snapshot, split metadata, commit 기록을 산출물로 남긴다.

검토 결과 현재 계획은 개인 프로젝트에서 먼저 완성할 수 있는 foundation과 실제 GPU·사람 검수가 필요한 연구 단계를 분리하고 있으므로 구현을 시작한다.
