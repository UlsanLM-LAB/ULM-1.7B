# ULM-1.7B Research Notes

이 문서는 `deep-research-report.md`를 primary research source로 삼아 현재 구현에 필요한 판단만 정리한 것이다. 이 저장소에서 아직 실행하지 않은 외부 조사, 데이터 다운로드, 학습 결과는 사실이나 결과로 기록하지 않는다.

## 핵심 연구 판단

### 프로젝트 정의

프로젝트의 핵심은 “경상도 말투 chatbot”이 아니라 화자 provenance가 확인된 울산 지역어를 텍스트 모델로 이해·변환·생성하고, 그 성능을 speaker-disjoint benchmark로 검증하는 것이다. 텍스트 모델의 이름은 `ULM-1.7B`, 장기 음성 생태계 이름은 `UlsanVoice`로 분리한다.

### 모델 후보

| 모델 | 용도 | 이유 | 확인할 것 |
|---|---|---|---|
| `Qwen/Qwen3-0.6B` | pipeline smoke | 작은 메모리와 빠른 반복 | 현재 Transformers/TRL 조합 |
| `Qwen/Qwen3-1.7B` | SFT MVP | 최종 규모와 같은 family | model card, tokenizer, chat template |
| `Qwen/Qwen3-1.7B-Base` | CPT→SFT | CPT 해석이 명확함 | base checkpoint license와 지원 |
| `Gemma 3 1B` | 선택 baseline | multilingual 비교 | Gemma license와 사용조건 |
| `SmolLM3-3B` | 선택 baseline | 공개 training 정보 | 한국어 성능과 compute |

report는 Qwen 계열을 시작 후보로 추천하지만, 이는 울산 성능의 증거가 아니다. 최종 선택은 smoke와 benchmark 결과를 보고 확정한다.

### 데이터 자산

report가 조사한 주요 후보는 AI Hub `한국어 방언 발화(경상도)`, AI Hub `중·노년층 한국어 방언 데이터(강원도, 경상도)`, 국립국어원 지역어 조사 자료, 우리말샘, 직접 수집한 울산 화자 음성이다.

AI Hub 공개 통계는 경상도 전체 규모이며 울산 subset 크기를 보장하지 않는다. 따라서 `Ulsan Core`의 크기와 token 수는 실제 사용자가 합법적으로 받은 archive를 audit한 뒤에만 결정한다.

### 울산 provenance

보고서의 우선 metadata는 `speaker_id`, `birthplace`, `principal_residence` 또는 `raised_region`, `current_residence`, `age`, `gender`, `dialect_text`, `standard_text`, `audio`, `source`다.

위치 문자열은 먼저 전체 unique value를 dump하고 사람이 normalization map을 확정한다. `울산광역시` prefix를 지원하되 공통 구 이름만으로 분류하지 않는다.

내부 tier는 다음과 같다.

- `U0`: 출생과 주 성장/주 거주가 울산으로 확인됨
- `U1`: 주 성장/주 거주와 현재 거주가 울산으로 확인됨
- `U2`: 관련 세 필드 중 하나만 울산으로 확인됨
- `GX`: 울산이 확인되지 않는 경상권 보조 화자

이는 원본 dataset의 공식 label이 아니라 이 연구의 분석 규칙이다.

### 학습 방법

첫 단계는 `SFT/QLoRA`다. `CPT`는 usable token 수를 측정한 후 결정한다. report의 heuristic은 `1M` 미만에서는 CPT를 주 실험으로 확장하지 말고, `1M~10M`에서는 QLoRA CPT를 탐색하며, `10M` 초과에서는 본 실험축으로 고려하는 것이다. 이는 연구 가설을 줄이는 실무 규칙이지 보편 법칙이 아니다.

QLoRA 시작점은 4-bit `NF4`, double quantization, 지원 시 `BF16`, `LoRA r=32`, `alpha=64`, dropout `0.05`, `learning_rate=1e-4`, sequence `1024`다. `r`, learning rate, epoch는 ablation 대상이며 최적값으로 주장하지 않는다.

### Evaluation

자동 평가만으로 지역어 authenticity를 판단하지 않는다.

- 이해·지역 구분: accuracy, macro-F1
- 변환: character F-score, meaning preservation
- 생성: dialect feature precision/recall, native authenticity, fluency
- 강도: monotonicity와 의미 보존
- 일반 한국어: retention set regression

최종 benchmark는 1,000개 이상을 목표로 하지만, 현재 저장소에는 framework와 fixture만 둔다. AI Hub 문장의 재배포를 피하고 직접 작성·권리 확보·원어민 검수를 거친 공개 문항을 별도로 만든다.

### TTS/STT 경계

report는 GPT-SoVITS를 TTS 기준 모델 후보로 제안하지만, 음성 cloning은 생체정보와 impersonation 위험이 있다. 현재 저장소는 TTS/STT를 구현하지 않고, `Speech -> STT -> ULM -> TTS -> Speech` 경로와 고정된 동의 speaker profile 원칙만 기록한다.

## 라이선스와 데이터 거버넌스

프로젝트 code는 Apache-2.0을 기본값으로 한다. 다음은 서로 다른 검토 대상이다.

1. base model과 tokenizer license
2. adapter와 merged model 배포 조건
3. AI Hub 원본 및 파생 manifest의 이용·재배포 조건
4. 자체 수집 text/audio의 저작권·개인정보·동의
5. benchmark 문항의 저작권과 annotator 검수
6. TTS speaker checkpoint와 공개 demo 범위

AI Hub 원본 WAV/JSON, 단순 가공 JSONL, dataset cache, checkpoint, secret은 Git에 넣지 않는다. 출처와 약관은 실제 다운로드 시점의 dataset page에서 다시 확인한다.

## 아직 확인하지 않은 사항

- 실제 AI Hub v1.4 archive의 key/value와 location 표현
- 울산 `U0/U1/U2` 화자 수와 usable audio/token 수
- 사용 시점의 package version별 Qwen3 chat template 동작
- 실제 benchmark의 native agreement와 inter-annotator agreement
- QLoRA가 현재 로컬 GPU와 무료 Colab runtime에서 안정적으로 동작하는지

위 항목은 추측으로 채우지 않고 각각 audit, smoke, human review 결과로 갱신한다.
