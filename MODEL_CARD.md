# ULM-4B Model Card

## Model summary

ULM-4B는 울산 방언의 이해와 생성을 목표로 하는 소형 언어모델 계열입니다.

현재 릴리스 후보는 Qwen3.8-4B-Distill을 base model로 사용하고, 울산 방언 데이터와 일반 능력 보존 데이터를 이용해 LoRA 기반으로 추가 학습한 Arm B입니다.

- Project: ULM / Ulsan Language Model
- Organization: UlsanLM Lab
- Current checkpoint: ULM-4B Arm B
- Status: Release Candidate
- Base: Qwen3.8-4B-Distill
- Fine-tuning: LoRA SFT
- Runtime: PyTorch + Hugging Face Transformers
- Serving: FastAPI
- Voice integration: ULM-LIVE + Qwen3-TTS

## Training lineage

```text
Qwen3.8-4B-Distill
  → Dialect Alignment v3
  → Context Repair v1 Arm B
  → Instruction Recovery v1 Arm B
  → ULM-4B Arm B release candidate
```

Arm B는 최종 연구 체크포인트라는 의미가 아니라, 현재 제품 통합과 ULM Live 데모를 위해 고정한 release candidate입니다.

## Evaluation

| Metric | Result |
| --- | ---: |
| Factual QA | 84% |
| Multi-turn | 100% |
| Instruction trap | 65% |
| Generation semantic | 0.9843 |
| Generation dialectness | 0.6589 |
| Grammar semantic | 0.9846 |
| Grammar dialectness | 0.8955 |
| Context semantic | 0.4578 |
| Context repetition | 0 |
| Context malformed | 0 |
| Identification accuracy | 32% |

The context benchmark contains conversational-next-turn references and should not be interpreted as a complete measure of open-ended conversation quality.

## Inference policy

The release candidate uses non-thinking chat templates and anti-loop decoding.

```text
enable_thinking = false
repetition_penalty = 1.10
no_repeat_ngram_size = 3
```

The HTTP server can load the base model and adapter separately through `ULM_MODEL_PATH` and `ULM_ADAPTER_PATH`.

## Intended use

- 울산 방언 생성과 변환 연구
- 지역어 챗봇 프로토타이핑
- ULM Live 음성 응답 시스템
- 소형 언어모델 fine-tuning 및 evaluation 연구
- 교육·포트폴리오·연구 데모

## Known limitations

- instruction-trap benchmark는 현재 65%로, 엄격한 형식 제약과 거짓 전제 거부에서 추가 개선이 필요합니다.
- identification accuracy는 현재 32%로 낮아 별도 개선이 필요합니다.
- dialect benchmark는 자동 지표를 포함하므로 실제 자연스러움에 대한 사람 평가를 대체하지 않습니다.
- 모델은 사실 오류나 환각을 만들 수 있습니다.
- 울산 방언에는 화자·세대·지역별 변이가 있으므로 하나의 출력 스타일을 전체 울산 방언으로 일반화하면 안 됩니다.

## Data and privacy

대형 원본 데이터와 비공개 음성 파일은 Git 저장소에 포함하지 않습니다. AI Hub 등 외부 데이터는 해당 서비스의 이용 조건과 개인정보 처리 조건을 따라야 합니다.

## Related repository

ULM Live voice runtime:

https://github.com/UlsanLM-LAB/ULM-LIVE
