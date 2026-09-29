# ULM-4B v1.0

2026-09-29 · **v1.0 Stable Release** · Git tag `v1.0.0`

ULM-4B v1.0은 울산 방언 생성·이해와 일반 한국어 능력 보존을 목표로 하는 4B급 모델입니다. 이번 릴리즈는 기존 **Arm B**를 그대로 동결하고 데이터 로더·추론·비교 실행 경로를 수정했습니다. 추가 학습과 weight merge는 수행하지 않았으며 v1.0의 추가 학습 계획도 없습니다. rejected Arm B+와 final low-LR 후보는 연구 기록으로만 유지합니다.

## Frozen model

- Base: [empero-ai/Qwen3.8-4B-Distill](https://huggingface.co/empero-ai/Qwen3.8-4B-Distill), revision `c83cb7aa2999d2f35c43e9ae0634a30eb8985a1e`
- Lineage: Dialect Alignment v3 → Context Repair v1 Arm B → Instruction Recovery v1 Arm B
- Tuning: 기존 LoRA SFT, r8/alpha16/dropout0.05; 마지막 학습 lr `6e-7`, fraction `0.25`, 35 steps, completion-only loss
- Frozen adapter SHA-256: `fee03f872aaff55283be0413536e20b0930334333a20c6ee01cad63f57f43716`
- Immutable contract: [ulm4b-v1.0.json](../configs/release/ulm4b-v1.0.json); 원 Arm B config와 연구 기록은 보존

모델 가중치와 LoRA adapter는 Git 저장소와 이 코드 릴리즈에 포함되지 않습니다. 사용 권한이 있는 기존 Arm B artifact를 별도로 준비하고 checksum을 확인해야 합니다.

## Release-path fixes

- SFT/CPT가 embedded train/validation/test를 먼저 분리한 뒤 외부 validation을 적용하며 canonical ID·화자 중복을 거부합니다.
- validation-only / train-only 평가 디렉터리를 정상 처리하고 잘못된 평가 경로를 명확히 거부합니다.
- client disconnect와 반복 cancel 후 실제 generation thread 종료까지 잠금을 유지합니다.
- 토큰 문맥 초과는 JSON·SSE 모두 헤더 전 HTTP 400 `invalid_request_error`입니다.
- 두 text CLI의 명시한 잘못된 adapter 경로는 모델 로딩 전 실패합니다.
- 비교기의 토큰 집계는 첫 EOS를 포함하고 이후 padding을 제외합니다. PAD=EOS도 처리합니다.
- 모든 결과를 skip해도 통합 summary를 생성·갱신합니다. stale/legacy 결과를 자동으로 재실행하지 않습니다.
- 비교 실행은 현재 Python과 명시적 model/adapter/dataset 설정을 사용하고 checksum, revision, 실제 runtime provenance를 기록합니다.
- interactive CLI·단일 입력 CLI·API는 같은 dialect strength prompt helper를 사용합니다.

HTTP endpoint와 기본 디코딩은 유지됩니다. 내부 `ChatEngine`을 주입하는 코드는 `validate_request()`를 구현해야 합니다.

## Benchmark snapshot and verification

기존 Arm B release snapshot은 재학습 없이 유지합니다:

| Metric | Original release snapshot |
| --- | ---: |
| Generation semantic | 0.9843 |
| Generation dialectness proxy | 0.6589 |
| Grammar dialectness proxy | 0.8955 |
| Context semantic | 0.4578 |
| Context repetition | 0 |
| Factual QA / Multi-turn | 84% / 100% |
| Instruction trap | 65% |
| Identification | 32% |

별도 2026-09-28 neutral 500문항 비교는 generation semantic `0.9841`, generation dialectness `0.6591`, grammar dialectness `0.8909`, context semantic `0.4594`, identification `30%`입니다. Context-only Guard ablation은 context semantic `0.4656`, context repetition `0`입니다. 전체 deployed-mode 평가와 구분합니다.

수정된 runner로 Arm B Neutral·Context Guard 각 500문항을 다시 생성했고 출력 1,000개 모두 역사적 raw prediction과 정확히 일치했습니다. 점수 산식과 과거 quality score는 바꾸지 않았습니다. 패딩 집계만 수정해 이번 측정의 실제 생성 토큰은 `8,271 / 8,005`, 처리량은 `91.61 / 100.73 tokens/sec`입니다. 과거 padding-inclusive 처리량은 기록으로 보존하며 검증된 v1.0 성능 주장에 사용하지 않습니다. 다른 모델 처리량은 이번에 재측정하지 않았습니다.

검증: 전체 CPU/offline **230 passed**, focused **148 passed**, core Ruff 통과. 실제 frozen Arm B로 두 endpoint × JSON/SSE × 강도 0–3의 **16요청**이 성공했고 가중치·adapter 전체 디렉터리 hash는 검증 전후 동일했습니다. 기존 EC2만 잠시 시작했으며 작업 후 **stopped** 상태를 확인했습니다. 자세한 증거는 [Stable Release report](ULM_4B_V1_0_STABLE_RELEASE.md)에 있습니다.

## Dialect strength and ULM-LIVE

- API `dialect_strength`: 0 Standard / 1 Mild / 2 Ulsan / 3 Strong, 기본값 2
- ULM-LIVE UI: 1/2/3
- 기존 policy: `enable_thinking=false`, `repetition_penalty=1.10`, `no_repeat_ngram_size=3`
- 기존 defaults: temperature 0.7, top-p 0.9, top-k 20, max new tokens 150
- `/v1/chat/completions`, `/api/chat`, SSE의 OpenAI `choices`와 Live `token` 필드 유지
- [ULM-LIVE](https://github.com/UlsanLM-LAB/ULM-LIVE)의 Qwen3-TTS 음성 경로는 별도 저장소에서 관리

## Known Limitations

- Instruction-trap 65%: 엄격한 형식 제약과 지시 추종의 기존 연구 gate는 여전히 fail입니다.
- Identification 32% / 별도 비교 30%: 울산·다른 경상·표준어 분류가 안정적이지 않습니다.
- Semantic/dialectness는 자동 proxy입니다. Native speaker human evaluation을 대체하지 않으며 이번 릴리즈에서 사람 평가를 수행하지 않았습니다.
- 사실 오류·환각, 지역·세대·화자 변이, prompt-based 강도 제어의 비보장성이 남습니다.
- 이번 검증은 텍스트 경로입니다. Live TTS/STT·실시간 음성 인터럽트·청취 품질을 재검증하지 않았습니다.
- 원 benchmark 입력과 모델 artifact가 공개 묶음에 없어 외부 사용자의 역사적 점수 재현에는 제약이 있습니다.

Stable은 명시된 연구·교육·데모 용도의 코드·설정·체크포인트 고정이며 모든 품질 gate 통과나 모든 production 용도에 대한 보장이 아닙니다.

## License, citation and references

코드는 [Apache-2.0](../LICENSE)입니다. [Base model card](https://huggingface.co/empero-ai/Qwen3.8-4B-Distill)는 Apache-2.0을 표시하며 base·dataset·encoder·TTS·파생 가중치의 upstream notice와 이용 조건은 각각 유지됩니다. 코드 릴리즈를 모든 외부 artifact에 대한 이용 허가로 해석하지 않습니다.

인용은 [CITATION.cff](../CITATION.cff) 또는 [README의 v1.0 BibTeX](../README.md#citation)를 사용하세요. 저장소 URL은 `https://github.com/UlsanLM-LAB/ULM-4B`입니다. 연구·라이브러리 출처는 [README References](../README.md#references)에 있습니다.
