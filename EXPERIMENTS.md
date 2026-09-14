# ULM-1.7B Experiments

실행하지 않은 실험에는 점수나 시간을 기입하지 않는다. `status`는 `PLANNED`, `READY_FOR_TRAINING`, `RUNNING`, `DONE`, `BLOCKED` 중 하나다.

## 실험 matrix

| ID | hypothesis | base model | dataset version | training method | variables | fixed parameters | metric | expected cost | status | result | notes |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `B0` | 원본 모델도 지역어 입력의 일부를 이해할 수 있다 | `Qwen/Qwen3-1.7B` | 없음 | baseline inference | decoding | seed와 prompt set | ULM-Bench task metrics | CPU/GPU inference | `READY_FOR_TRAINING` | 미실행 | 학습 없는 기준선 |
| `S0` | 전체 pipeline은 작은 모델에서 끝까지 동작한다 | `Qwen/Qwen3-0.6B` | fixture 또는 합법적 소량 data | QLoRA SFT | 5~20 steps | config snapshot | loss, checkpoint, resume, fixture metric | 무료 GPU smoke | `DONE` | 아래 실측 기록 | 실제 울산 성능으로 해석하지 않음 |
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

## S0 실측 기록

- 실행 환경: 로컬 `NVIDIA GeForce RTX 5060 Laptop GPU`, 보고 VRAM 약 8GB
- package: `torch 2.14.0+cu130`, `transformers 5.17.0`, `datasets 5.0.1`, `peft 0.20.0`, `trl 1.13.0`, `bitsandbytes 0.50.2`
- model: `Qwen/Qwen3-0.6B`, tokenizer vocabulary `151643`, load 시 parameter count `375848960`
- data: `data/examples/sft_fixture.jsonl`의 3개 fixture record. 모두 `synthetic=true`, `human_verified=false`, `not_for_research=true`이며 울산 성능 평가에 사용하지 않음
- 설정: 4-bit `NF4`, double quantization, `lora_r=16`, `lora_alpha=32`, `max_steps=5`, `save_steps=2`, seed `42`
- 결과: 기본 FP16 config로 5 step 학습 완료. log의 step별 loss는 `3.529`, `3.529`, `2.500`, `1.928`, `1.639`, 최종 `train_loss=2.625`, 최종 `mean_token_accuracy=0.7031`
- checkpoint: `checkpoint-4` 저장 후 `resume_from_checkpoint=true` 재실행을 완료했고, 후속 `checkpoint-5`와 `global_step=5`를 확인함
- inference: 저장 adapter와 같은 base model로 `오늘 뭐 해?`, `dialect_strength=2`, greedy decoding을 실행해 text 출력 완료. 출력 품질은 fixture smoke 증거일 뿐 지역어 성능 점수가 아님
- fixture evaluation: `ulm-evaluate` 실행 결과 `evaluated=2`, `missing_predictions=0`, fixture 기준 exact accuracy와 character F1은 각각 `1.0`

### S0에서 발견·수정한 실패

첫 FP16 실행은 첫 optimizer step에서 `BFloat16` gradient를 FP16 GradScaler가 unscale하지 못해 실패했다. 현재 `TRL 1.13.0`의 QLoRA adapter dtype 처리와 충돌한 것이 원인이었고, FP16 경로에서 trainable adapter parameter를 FP32로 복구하는 호환 처리를 추가한 뒤 동일 5 step이 완료됐다. 이 수정 후 non-Flash 환경의 `packing`은 `wrapped` strategy로 실행해 padding-free 경고도 제거했다.

현재 1.7B 학습·실제 Ulsan Core audit·native benchmark 결과는 없다.
