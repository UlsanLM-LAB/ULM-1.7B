# ULM-1.7B Experiments

실행하지 않은 실험에는 점수나 시간을 기입하지 않는다. `status`는 `PLANNED`, `READY_FOR_TRAINING`, `RUNNING`, `DONE`, `BLOCKED` 중 하나다.

## 실험 matrix

| ID | hypothesis | base model | dataset version | training method | variables | fixed parameters | metric | expected cost | status | result | notes |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `B0` | 원본 모델도 지역어 입력의 일부를 이해할 수 있다 | `Qwen/Qwen3-1.7B` | 없음 | baseline inference | decoding | seed와 prompt set | ULM-Bench task metrics | CPU/GPU inference | `READY_FOR_TRAINING` | 미실행 | 학습 없는 기준선 |
| `S0` | 전체 pipeline은 작은 모델에서 끝까지 동작한다 | `Qwen/Qwen3-0.6B` | fixture 또는 합법적 소량 data | QLoRA SFT | 5~20 steps | config snapshot | loss, checkpoint, resume, fixture metric | 무료 GPU smoke | `READY_FOR_TRAINING` | 미실행 | 실제 결과만 기록 |
| `B1` | 광역 경상도 data가 baseline보다 개선한다 | `Qwen/Qwen3-1.7B` | `Gyeongsang Auxiliary` | SFT/QLoRA | data size | split seed, eval set | ULM-Bench | 무료/저비용 GPU | `PLANNED` | 미실행 | 울산으로 해석하지 않음 |
| `U1` | 울산 provenance data가 광역 data보다 지역성에 유리하다 | `Qwen/Qwen3-1.7B` | `Ulsan Core` | SFT/QLoRA | data size, LoRA rank | split seed, eval set | authenticity, meaning, region | 무료/저비용 GPU | `PLANNED` | 미실행 | audit 선행 |
| `U2` | CPT가 SFT만으로 얻기 어려운 지역어 적응을 제공한다 | `Qwen/Qwen3-1.7B-Base` | `Ulsan Core` | CPT -> SFT | CPT LR, token count | held-out standard set | perplexity, task metrics | GPU 필요 | `PLANNED` | 미실행 | token 수가 작으면 축소 |
| `U3` | general Korean mix가 CPT 후 retention 저하를 줄인다 | `Qwen/Qwen3-1.7B-Base` | Ulsan + general mix | CPT -> SFT | mix ratio | split seed, eval set | dialect gain, retention delta | GPU 필요 | `PLANNED` | 미실행 | catastrophic forgetting 확인 |
| `C1` | conditioning이 강도 증가를 단조롭게 만든다 | 선택된 ULM | native-reviewed strength set | SFT/QLoRA | strength 0~3 | prompt template | monotonicity, meaning | GPU + human eval | `PLANNED` | 미실행 | 치환 규칙으로 gold 생성 금지 |

## 실험 기록 규칙

각 실행은 다음 정보를 `reports/experiments/`에 남긴다.

- experiment ID와 실행 시각
- Git commit
- resolved config snapshot
- model revision과 tokenizer revision
- dataset version, source/license gate, record count, token count
- split seed와 speaker overlap 검사 결과
- hardware와 package versions
- 학습 log, checkpoint 경로, resume 여부
- 자동 metric과 사람 평가 결과
- 실패 원인과 다음 조치

## 현재 기록

현재 실제 학습·평가 결과는 없다. 저장소 foundation을 먼저 구현하고, 외부 데이터 접근과 GPU 실행이 가능한 시점에 `S0`부터 기록한다.
