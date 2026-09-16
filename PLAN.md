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
- 검증된 ULM 출력의 울산 지역어 TTS 합성
- 장기적으로 텍스트·음성을 연결한 로컬 울산 지역어 AI pipeline 구축

핵심 개발 범위는 텍스트 모델 `ULM-1.7B`이며, 텍스트 모델의 품질이 검증된 뒤 후속 단계로 `ULM-TTS`를 개발한다. TTS는 단순히 사투리 문장을 읽는 기능이 아니라 울산 지역어의 억양·리듬·발음 특성을 가능한 범위에서 보존하는 음성 합성을 목표로 한다. STT는 이후 end-to-end 음성 대화를 위한 후속 interface로 둔다.

## 2. 문제 정의

공개 경상권 방언 데이터 전체를 울산 데이터로 간주하면 지역성, 데이터 provenance, 평가 타당성이 무너진다. 따라서 모델 학습보다 먼저 실제 화자 metadata를 audit하고, 울산 출생·성장·거주 조건을 구분한 `U0/U1/U2/GX` tier와 화자 단위 split을 구축한다.

이 프로젝트의 성공은 출력에 사투리 어미가 들어가는지로만 판단하지 않는다. 의미 보존, 울산 화자 평가, 지역 구분, 강도 제어, 일반 한국어 보존을 함께 측정한다. TTS 단계에서도 단순 음질뿐 아니라 울산 지역어의 발음·억양·리듬이 실제 화자에게 자연스럽게 인식되는지를 별도로 평가한다. `deep-research-report.md`는 텍스트 SLM 판단의 primary research source이며, 아직 다운로드하지 않은 외부 데이터의 크기나 성능을 사실처럼 사용하지 않는다.

## 3. Research Questions

1. 1.7B급 모델이 narrow domain인 울산 지역어에서 더 큰 범용 모델과 경쟁 가능한가?
2. `Ulsan Core`가 일반 `Gyeongsang Auxiliary`보다 얼마나 효과적인가?
3. `SFT`와 `CPT + SFT`의 차이는 무엇인가?
4. 검수된 synthetic data 비율에 따라 dialect authenticity가 어떻게 변하는가?
5. 울산·부산·대구 지역 구별이 가능한가? 구별 불가능한 shared 표현은 어떻게 표시해야 하는가?
6. `dialect_strength`를 의미 보존과 함께 안정적으로 제어할 수 있는가?
7. 텍스트에서 학습한 `dialect_strength`와 지역어 특징을 TTS의 prosody control에 연결할 수 있는가?
8. 범용 한국어 TTS를 울산 지역어 음성에 적응시켰을 때 표준어 음질을 크게 잃지 않고 울산 억양을 재현할 수 있는가?
9. SLM + TTS 전체 pipeline을 개인용 GPU 또는 로컬 환경에서 실용적인 지연시간으로 실행할 수 있는가?

## 4. 설계 원칙

- 데이터가 없으면 데이터가 있다고 가정하지 않는다.
- AI Hub 원본, audio, checkpoint, cache, secret은 Git에 넣지 않는다.
- `speaker_id`는 학습용 manifest에서 pseudonymous ID로 취급하고, 실명 대응표는 별도 보안 영역에 둔다.
- 같은 화자의 발화는 train·validation·test에 섞지 않는다.
- 검수되지 않은 synthetic dialect data는 주 데이터로 사용하지 않는다.
- benchmark는 학습보다 먼저 schema와 평가 함수를 만들고, 실제 문항은 사람 검수 workflow를 거친다.
- GPU가 없어도 data contract, validation, split, metric, config, CLI가 동작해야 한다.
- 고정된 값은 코드에 흩뿌리지 않고 config에 둔다. 다만 불필요한 framework와 abstraction은 추가하지 않는다.
- TTS는 동의와 사용 범위가 확인된 음성만 사용하며 임의 화자 복제 기능을 목표로 하지 않는다.
- 텍스트 모델과 음성 모델은 분리해 평가하고, end-to-end demo의 결과만으로 개별 모델의 성능을 주장하지 않는다.

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
  -> text/prosody profile
  -> ULM-TTS
  -> speech output
```

구현은 텍스트 SLM을 먼저 완성하고 음성 계층을 후속으로 연결한다.

1. `ulm.data`: schema, JSONL 입출력, AI Hub audit, 위치 정규화, speaker split
2. `ulm.evaluation`: benchmark schema, fixture, task별 baseline metric
3. `ulm.training` 및 `ulm.inference`: optional ML dependency를 lazy import하는 실행 adapter
4. 후속 `ULM-TTS`: 음성 manifest, acoustic/prosody feature, TTS adaptation, 음성 평가와 inference adapter

핵심 data·metric 로직은 PyTorch나 Hugging Face를 import하지 않는다. 그러므로 CPU 환경에서 validation과 test를 실행할 수 있다. 모델 학습만 `transformers`, `datasets`, `peft`, `trl`, `bitsandbytes`, `torch` extra에 의존한다. TTS dependency는 텍스트 SLM dependency와 분리한다.

## 6. Model strategy

| 역할 | 기본 모델 | 목적 | 현재 판단 |
|---|---|---|---|
| smoke test | `Qwen/Qwen3-0.6B` | dataset loading부터 5~20 step까지 검증 | 우선 실행 |
| 빠른 SFT | `Qwen/Qwen3-1.7B` | MVP dialect instruction tuning | 준비 |
| 연구 주 모델 | `Qwen/Qwen3-1.7B-Base` | `CPT -> SFT` 해석 가능한 비교 | 최종 후보 |
| 선택 비교 | `Gemma 3 1B`, `SmolLM3-3B` | 선택적 multilingual/open-training baseline | 후순위 |

`Qwen3-1.7B-Base`와 post-trained `Qwen3-1.7B`를 같은 모델 family 안에서 비교한다. Apache-2.0 여부, 한국어 tokenizer 효율, `QLoRA`, `transformers`, `PEFT`, `TRL`, `GGUF`, `llama.cpp` 호환성은 실제 실험 시작 시 사용 버전의 model card와 license를 다시 기록한다. 연구 보고서의 추천은 시작점이지 최적값이나 울산 성능 보장이 아니다.

TTS의 base architecture는 지금 고정하지 않는다. 실제 음성 데이터의 양, 라이선스, 화자 수, sampling rate, alignment 품질을 audit한 뒤 공개 한국어 TTS 중 fine-tuning 가능성, 로컬 추론성, prosody control, 라이선스를 비교해 결정한다. 특정 TTS 모델을 먼저 정한 뒤 데이터 조건을 끼워 맞추지 않는다.

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

### 7.4 TTS audio dataset

TTS 단계에서는 텍스트 manifest와 별도의 audio manifest를 사용한다. 최소 필드는 `utterance_id`, `speaker_id`, `audio_path`, `transcript`, `region_tier`, `age_group`, `dialect_strength`, `sample_rate`, `duration`, `consent_scope`, `quality_grade`, `split`, `metadata`로 둔다.

음성 데이터는 다음 원칙으로 준비한다.

- 공개·학습·배포 범위를 각각 확인하고 `consent_scope`에 기록한다.
- speaker-level split을 유지해 같은 화자의 유사 발화가 test로 새지 않게 한다.
- 지나친 배경음, clipping, transcript mismatch, 너무 짧거나 긴 발화를 quality gate에서 제외한다.
- 울산 고유 억양을 검증할 수 있도록 declarative, interrogative, imperative 등 문장 유형을 가능한 범위에서 균형 있게 확보한다.
- pitch/F0, duration, energy 등 prosody feature는 원본 음성을 대체하는 gold가 아니라 분석·제어용 보조 feature로 사용한다.
- 직접 수집할 경우 참가자 동의서에 학습, 연구 공개, demo 사용, 모델 배포 범위를 분리해 기록한다.

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

### 8.1 TTS training strategy

TTS는 `ULM-1.7B`의 텍스트 품질이 benchmark와 native review를 통과한 뒤 시작한다.

1. `T0`: 범용 한국어 TTS baseline으로 울산 문장 합성
2. `T1`: 동의된 울산 음성 데이터 audit 및 preprocessing
3. `T2`: 소규모 adaptation smoke test
4. `T3`: 울산 지역어 TTS fine-tuning
5. `T4`: 표준어 TTS와 울산 TTS blind comparison
6. `T5`: `dialect_strength`와 prosody control 연결 실험
7. `T6`: ULM-1.7B 출력 -> ULM-TTS end-to-end inference
8. `T7`: 양자화·경량화가 가능한 architecture라면 로컬 inference benchmark

TTS의 목표는 특정 개인의 목소리를 복제하는 것이 아니라 울산 지역어의 발음과 운율 특성을 자연스럽게 합성하는 것이다. 다화자 학습이 가능한 데이터 규모라면 speaker identity와 dialect feature를 분리해 학습하는 방향을 우선 검토한다.

## 9. Evaluation

`ULM-Bench`의 최종 목표는 1,000개 이상이지만, 초기 구현은 schema와 작은 fixture만 포함한다. 공개 test는 AI Hub 문장을 그대로 재배포하지 않고, 권리를 확보한 문장과 원어민 검수 문장으로 별도 구축한다.

초기 task는 `dialect_understanding`, `dialect_to_standard`, `standard_to_dialect`, `dialect_chat`, `region_classification`, `dialect_strength_control`이다.

- 이해·지역 분류: accuracy, macro-F1
- 울산→표준어: character F-score와 semantic human score
- 표준어→울산: dialect feature precision/recall과 native authenticity
- 강도 제어: feature density monotonicity와 meaning preservation
- 일반 한국어: 별도 retention set의 정확도와 regression

자동 metric은 baseline 비교용이며 “사투리다움”을 단독으로 판정하지 않는다. 사람 평가가 가능한 단계에서는 울산 화자 blind evaluation과 참가자 단위 confidence interval을 사용한다.

### 9.1 TTS evaluation

TTS는 최소 다음 축을 분리해 평가한다.

- intelligibility: 문장이 정확히 들리는가
- naturalness: 합성음 자체가 자연스러운가
- dialect authenticity: 울산 지역어 억양·발음으로 자연스럽게 인식되는가
- meaning preservation: ULM이 생성한 텍스트 의미가 음성에서 훼손되지 않는가
- speaker leakage/privacy: 평가용 화자를 단순 암기하거나 특정 개인을 과도하게 모사하지 않는가
- latency/RTF: 로컬 또는 Colab 환경에서 실용적인 속도로 합성되는가

가능하면 울산 화자를 포함한 blind MOS 또는 pairwise preference 평가를 사용한다. 자동 pitch/F0 similarity나 ASR 기반 intelligibility는 보조 metric으로 사용하고 dialect authenticity를 자동 metric 하나로 결론내리지 않는다.

## 10. Google Colab

Colab notebook은 orchestration만 담당하고 `src/ulm`을 import한다.

권장 흐름은 GPU 확인 → dependency 설치 → Drive mount → `/content`에 archive 복사 → audit/validation/split → baseline → QLoRA → checkpoint backup → benchmark evaluation → adapter export다.

Drive에는 작은 파일 수천 개를 직접 읽게 하지 않고 archive 또는 Parquet를 복사해 `/content`에서 처리한다. `save_steps`, `eval_steps`, `save_total_limit`, `seed`, `output_dir`, `resume_from_checkpoint`를 config로 제어한다.

TTS 단계가 시작되면 별도 notebook 또는 별도 실행 entry point를 사용한다. 음성 archive를 `/content`로 복사한 뒤 preprocessing과 학습을 수행하고, checkpoint를 Drive에 주기적으로 백업한다. 텍스트 SLM 학습 notebook과 TTS 학습 notebook을 하나의 거대한 notebook으로 합치지 않는다.

## 11. Local inference

학습된 base 또는 adapter를 `ulm.inference` CLI로 불러온다. `--dialect-strength 0~3`을 system/user prompt의 control로 변환하고, 출력은 text로 저장한다. GGUF 변환과 `llama.cpp` 실행은 adapter 품질이 검증된 후 별도 작업으로 둔다.

최종 음성 pipeline은 다음 구조를 목표로 한다.

```text
Text input
  -> ULM-1.7B
  -> dialect text + dialect_strength + optional prosody profile
  -> ULM-TTS
  -> Ulsan dialect speech
```

STT까지 연결하는 장기 구조는 다음과 같다.

```text
Speech
  -> STT
  -> ULM-1.7B
  -> dialect text + prosody profile
  -> ULM-TTS
  -> Speech
```

`ULM-1.7B`와 `ULM-TTS`는 각각 독립 실행이 가능해야 하며, end-to-end layer는 두 모델을 얇게 연결한다. 공개 demo에서는 동의된 고정 speaker profile 또는 프로젝트가 배포 권한을 가진 voice만 사용한다.

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
│   ├── sft/
│   └── tts/                 # TTS phase에서 추가
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
├── tts/                     # 텍스트 모델 검증 후 추가
│   ├── configs/
│   ├── evaluation/
│   ├── inference/
│   └── training/
└── tests/
```

TTS 코드는 텍스트 모델이 검증되기 전에는 빈 framework부터 만들지 않는다. 실제 TTS phase가 시작될 때 필요한 최소 구조만 추가한다. `Manager`, `Factory`, `Repository`, dependency injection, microservice, plugin framework 계층은 도입하지 않는다.

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
  -> ULM-1.7B text model freeze candidate
  -> TTS audio audit / consent gate
  -> TTS baseline
  -> ULM-TTS adaptation
  -> TTS native evaluation
  -> ULM-1.7B + ULM-TTS integration
  -> optional STT integration
  -> local end-to-end demo
```

데이터 audit가 외부 로그인·약관 동의를 요구하면 해당 단계만 보류하고, audit 입력을 받기 전까지 fixture와 코드 검증을 계속한다. TTS는 음성 데이터의 사용 권리와 화자 동의 범위가 확인되기 전에는 학습을 시작하지 않는다.

## 14. Risks and license

- AI Hub의 실제 울산 화자 수와 원본 key는 다운로드 후에만 확정된다.
- AI Hub 원본과 단순 가공 manifest는 재배포하지 않는다.
- 모델, adapter, merged model, benchmark, 직접 수집 voice는 각각 license와 consent를 따로 기록한다.
- 모델 license가 Apache-2.0이어도 데이터·화자 권리까지 해결되는 것은 아니다.
- synthetic data는 실제 울산 화자 발화와 동일한 증거로 취급하지 않는다.
- 작은 데이터에서 CPT를 강행하면 일반 한국어 저하와 과적합이 생길 수 있다.
- 지역 판별이 불가능한 공통 표현을 억지로 지역 gold로 만들지 않는다.
- TTS 학습 데이터가 적으면 특정 화자 음색과 울산 억양을 분리하지 못하고 과적합할 수 있다.
- TTS의 높은 음질이 곧 울산 지역어의 높은 authenticity를 의미하지 않는다.
- 직접 수집 음성은 consent 범위를 넘어 공개하거나 voice cloning 용도로 재사용하지 않는다.
- 외부 TTS base model의 license가 fine-tuned checkpoint 배포를 허용하는지 별도로 확인한다.

프로젝트 코드의 기본 license는 Apache-2.0이다. 외부 모델·데이터 license는 별도 확인 후 model card와 dataset card에 기록한다.

## 15. Definition of Done

### 텍스트 SLM 코드

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

### ULM-TTS

- 사용 가능한 모든 음성에 provenance와 consent scope가 기록되어 있다.
- TTS train/validation/test가 speaker-aware하게 분리되어 있다.
- 범용 한국어 TTS baseline과 ULM-TTS 결과를 같은 문장 집합에서 비교한다.
- intelligibility, naturalness, dialect authenticity를 서로 다른 평가 항목으로 기록한다.
- ULM의 `dialect_strength`가 TTS 단계까지 전달되는 interface가 정의되어 있다.
- `ULM-1.7B -> ULM-TTS` end-to-end inference가 최소 하나의 재현 가능한 실행 경로로 동작한다.
- 로컬 실행이 가능한 경우 모델 크기, peak memory, first-audio latency 또는 RTF를 기록한다.

## 16. Self-review 결과

- scope: 1.7B text SLM을 먼저 완성하고, 검증 이후 ULM-TTS를 별도 phase로 개발한다.
- 비용: 로컬 CPU와 무료 Colab 우선이며 유료 GPU·외부 결제는 자동 실행하지 않는다.
- leakage: speaker-level split을 schema 운영의 기본 경계로 두었다. template/near-duplicate grouping은 데이터 audit 후 확장한다.
- license: AI Hub 원본·파생 manifest를 Git에서 제외하고, 모델·데이터·voice 권리를 분리한다.
- synthetic: source flag와 human review flag를 분리했다. 대량 synthetic 생성은 하지 않는다.
- evaluation: 학습 전에 benchmark schema와 metric을 만들고 TTS에서는 음질과 방언 authenticity를 분리한다.
- resume: config와 checkpoint 경로를 명시하고 output overwrite를 기본 금지한다.
- 구조: pure Python core와 optional ML adapter를 유지하고 TTS dependency를 후속 phase로 분리한다.
- 재현성: seed, config snapshot, split metadata, commit 기록을 산출물로 남긴다.
- voice safety: 임의 화자 복제가 아니라 동의된 데이터 기반의 울산 지역어 합성을 목표로 한다.

검토 결과 현재 계획은 `ULM-1.7B` 텍스트 SLM을 먼저 연구적으로 검증한 뒤 `ULM-TTS`로 확장하고, 최종적으로 로컬에서 실행 가능한 울산 지역어 text-to-speech pipeline까지 연결하는 단계적 구조로 진행한다.
