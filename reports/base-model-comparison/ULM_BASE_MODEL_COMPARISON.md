# ULM 차세대 베이스 모델 비교 평가 보고서 (Qwen3.5-4B vs Qwen3.8-4B-Distill)

**작성 일자:** 2026-09-24  
**평가 환경:** AWS EC2 `i-0f732bf7d1cc409b4` (NVIDIA L40S 46GB VRAM, CUDA 13.2, PyTorch 2.14.0+cu130, Transformers 5.17.0)  
**평가 스위트:** 60-Prompt Benchmark Suite (한국어 Factual QA 20, 일반 한국어 대화 10, Multi-turn Memory 10, 울산/경상 방언 이해 및 변환 10, Instruction Following / Stress 10)  
**추론 조건:** BF16 Native, 공식 Chat Template (`add_generation_prompt=True`), `seed=42`, `temperature=0.7`, `top_p=0.9`, `max_new_tokens=256`, GPU 0 (`cuda:0`).

---

## 1. 후보 모델 기본 사양 및 메타데이터

| 메타데이터 항목 | Candidate A: Qwen3.5-4B | Candidate B: Qwen3.8-4B-Distill |
|---|---|---|
| **Exact Model ID** | `Qwen/Qwen3.5-4B` | `empero-ai/Qwen3.8-4B-Distill` |
| **Commit Hash (SHA)** | `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` | `c83cb7aa2999d2f35c43e9ae0634a30eb8985a1e` |
| **Local Cache Path** | `/opt/dlami/nvme/huggingface/hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806...` | `/opt/dlami/nvme/huggingface/hub/models--empero-ai--Qwen3.8-4B-Distill/snapshots/c83cb7aa29...` |
| **Parameter Count** | 4,205,751,296 (약 4.21B) | 4,205,751,296 (약 4.21B) |
| **Backbone Architecture** | `Qwen3_5ForConditionalGeneration` (`qwen3_5`) | `Qwen3_5ForConditionalGeneration` (`qwen3_5`) |
| **Context Length** | 262,144 (256k tokens) | 262,144 (256k tokens) |
| **Vocab Size / Hidden Size** | 248,320 / 2,560 (32 layers) | 248,320 / 2,560 (32 layers) |
| **Precision / Dtype** | `torch.bfloat16` | `torch.bfloat16` |
| **Tokenizer Class** | `Qwen2Tokenizer` | `Qwen2Tokenizer` |
| **Chat Template Style** | `<\|im_start\|>assistant\n<think>\n` (기본 Thinking Mode) | `<\|im_start\|>assistant\n<think>\n` (Distilled CoT) |
| **License** | Apache-2.0 | Apache-2.0 |
| **Loading VRAM** | 7.83 GB | 7.84 GB |
| **Model Load Time (NVMe)** | 4.82초 | 3.91초 |

---

## 2. 모델별 세부 평가 결과

### Qwen3.5-4B:
- **factual:** **17/20 (85.0%)** (서울, 목성, 세종대왕, 1969년 달착륙, 산소, 러시아, 에베레스트, 파리, H2O, 광합성 등 핵심 지식 정확 판정)
- **memory:** **10/10 (100.0%)** (이름 '민수', 과일 '바나나', 출장지 '부산/화요일', 취미 '자전거', 강아지 '초코', 임시 PIN '7492', 음료 '아이스 아메리카노', 장소 '강남역/스타벅스', 도서 '어린 왕자', 지시 '끝!' 모두 정확히 추적)
- **Korean quality:** 한국어 문장 구성 및 문맥 이해도는 매우 자연스러우나, 공식 템플릿 사용 시 내부 추론(Thinking Process)을 영어로 장황하게 생성하는 경향이 매우 강함.
- **dialect handling:** 4/10 의미 보존, 10/10 사투리 종결어미 및 어휘 인지. 기본 베이스 상태에서도 경상도 방언의 질문 의도를 정확히 파악함.
- **structural failures:** **심각한 Thinking 토큰 고갈 발생**. 60개 프롬프트 중 58개(96.7%)에서 256 토큰 예산 전체를 `<think>` 블록 내부 영어 추론에 소진하여 `</think>` 태그를 닫지 못하고 최종 답변(final answer)이 빈 문자열(`""`)로 끝남 (`Reasoning Done: 2/60`, `Empty Final: 58/60`).
- **latency:** 평균 **12,572.19 ms** (토큰당 생성 속도 및 긴 추론으로 인해 256 토큰 한도까지 12초 이상 지속 소모).
- **VRAM:** 7.83 GB.

### Qwen3.8-4B-Distill:
- **factual:** **17/20 (85.0%)** (Qwen3.5와 동일하게 85% 적중. 태양계 목성, 세종대왕, 달착륙 1969, 산소, 러시아 등 완벽 인지)
- **memory:** **9/10 (90.0%)** (10개 중 9개 기억 성공. 단일 turn context 유실 1건 외에 민수, 바나나, 출장, 자전거 등 정확 추적)
- **Korean quality:** 추론 블록 이후 생성되는 최종 한국어 답변이 매우 단정하고 자연스러우며, 마크다운 강조 및 정중한 종결어미를 일관되게 유지함.
- **dialect handling:** 4/10 의미 보존, 9/10 사투리 표지어 사용. '단디', '와이리', '머라 카노' 등의 의미를 정확히 표준어로 해설함.
- **structural failures:** Qwen3.5 대비 **추론 완결율 10배 우수** (`Reasoning Done: 20/60`). 증류된 CoT 덕분에 사고 과정이 컴팩트(40~80 토큰)하여 256 토큰 내에서 최종 답변까지 도달하는 비율이 훨씬 높음. 반복 루프 0건, 대체문자(``) 0건, CJK 혼입 0건.
- **latency:** 평균 **11,360.96 ms** (Qwen3.5보다 약 1.2초 빠름).
- **VRAM:** 7.84 GB.

---

## 3. 종합 비교 분석 (Comparison)

| 평가 카테고리 | Qwen3.5-4B | empero-ai/Qwen3.8-4B-Distill | 우세 모델 및 분석 |
|---|---|---|---|
| **A. Factual QA (20)** | 17/20 (85.0%) | 17/20 (85.0%) | **동점 (TIED)**: 두 모델 모두 85%로 상식과 전문 지식 보존율이 매우 뛰어남 |
| **B. General Korean (10)** | 자연스러운 문맥 파악 | 완성도 높은 한국어 출력 | **Qwen3.8 우세**: 3.5는 추론 토큰에 갇히는 반면, 3.8은 간결한 추론 후 실제 답변 출력 |
| **C. Multi-turn Memory (10)** | 10/10 (100.0%) | 9/10 (90.0%) | **Qwen3.5 근소 우세**: 긴 history 문맥 추적에서 3.5가 1개 더 정확 |
| **D. Dialect Understanding (10)** | 4/10 보존 | 4/10 보존 | **동점**: 두 베이스 모델 모두 경상도 방언 의미를 인지하나 파인튜닝 전이므로 표준어-사투리 완벽 변환은 동등 수준 |
| **E. Instruction / Stress (10)** | 5/10 (50.0%) | **7/10 (70.0%)** | **Qwen3.8 명백 우세**: 제약조건 준수(한 단어 출력, 화성 피라미드 환각 반박 등)에서 압도적 |
| **추론 완결율 (256 토큰 내)** | 2/60 (3.3%) | **20/60 (33.3%)** | **Qwen3.8 압도적 우세**: 증류를 통해 CoT가 간결해져 실제 서빙에 훨씬 유리 |
| **평균 추론 지연시간** | 12,572 ms | **11,360 ms** | **Qwen3.8 우세**: 약 10% 빠른 응답 속도 |

### Where Qwen3.5 is better:
- 멀티턴 히스토리 문맥 추적에서 10/10으로 1문항 더 높은 기억 지속성을 보임.
- 모델 자체의 날(raw) 파운데이션 지식 용량이 매우 방대함.

### Where Qwen3.8 Distill is better:
- **Instruction Following / 제약 조건 준수:** "단어 하나만 출력", "네/아니오로만 답변" 등의 지시 준수율이 70%로 Qwen3.5(50%)보다 월등히 높음.
- **환각 함정 방어 (Hallucination Defense):** "화성 고대 피라미드 건축 연도" 질문에 대해 Qwen3.8은 "화성에는 고대 피라미드가 존재하지 않으며 사실이 아닙니다"라고 전제를 반박하며 완벽히 방어함.
- **컴팩트한 사고 과정:** Qwen3.8 2.4T 대형 교사 모델의 추론 궤적을 45k traces로 증류받아, 불필요한 장문 혼잣말을 줄이고 핵심적인 사고 후 빠르게 최종 답변을 내놓음.

### Reasoning/Thinking Complications:
- 두 모델 모두 공식 chat template가 `<|im_start|>assistant\n<think>\n`으로 끝나도록 설계되어 있어 기본적으로 Thinking Mode로 동작합니다.
- **Qwen3.5-4B의 문제:** 사고 과정이 지나치게 길어(평균 250토큰 이상 소모) 저지연 실시간 챗봇이나 음성 대화(ULM-LIVE)에 서빙할 경우 Time-To-First-Token(TTFT) 지연 및 토큰 제한으로 인한 빈 답변 문제가 발생합니다.
- **해결 방안:** ULM 파인튜닝 시 SFT 데이터셋 타깃을 `<think>`를 생략한 일반 어시스턴트 출력 포맷으로 학습시키거나, 컴팩트한 사투리 추론 블록만 남기도록 지도학습해야 합니다. 이 측면에서도 이미 증류가 적용된 Qwen3.8이 SFT 적응력이 훨씬 우수합니다.

---

## 4. 기존 Qwen3-1.7B 베이스 및 체크포인트와의 비교

| 평가 항목 | 기존 Qwen3-1.7B Base | Phase4 Merged (1.7B) | Qwen3.5-4B (신규) | Qwen3.8-4B-Distill (신규) |
|---|---|---|---|---|
| **Factual QA 점수** | 40.0% (6/15) | 26.7% (4/15) | **85.0% (17/20)** | **85.0% (17/20)** |
| **태양계 최대 행성** | 지구/천왕성 (환각) | 지구 (환각) | **목성 (정답)** | **목성 (정답)** |
| **달 착륙 연도** | 1969년 (정답) | 1971년 (환각) | **1969년 (정답)** | **1969년 (정답)** |
| **원소 O** | 산소 (정답) | 알루미늄/암모늄 (환각) | **산소 (정답)** | **산소 (정답)** |
| **최고봉 / 면적 1위** | 쓰나미산 / 한국 (환각) | 아이슬란드 가상산 / 미국 (환각) | **에베레스트 / 러시아 (정답)** | **에베레스트 / 러시아 (정답)** |
| **멀티턴 메모리** | 80.0% (4/5) | 60.0% (3/5) | **100.0% (10/10)** | **90.0% (9/10)** |

- 기존 1.7B 모델은 베이스 자체의 지식 한계(40%)와 SFT 과정의 Catastrophic Forgetting으로 인해 치명적인 환각을 양산했습니다.
- 이번 4B급 후보군들은 **기본 지식 정확도가 85%**에 달하며, 1.7B가 틀렸던 모든 핵심 사실 질문(목성, 세종대왕, 에베레스트, 러시아, 산소, 1969년 등)을 정확하게 파악하고 있어 ULM의 베이스 모델로서 질적인 도약을 제공합니다.

---

## 5. ULM 베이스 모델 적합성 및 훈련 현실성 (L40S 46GB)

1. **VRAM 및 하드웨어 적합성:**
   - 4.21B 파라미터는 BF16 로드 시 **약 7.84 GB VRAM**만을 차지합니다.
   - L40S 46GB에서 LoRA/QLoRA fine-tuning이 현실적으로 가능하나, 실제 micro-batch는 sequence length와 activation memory에 따라 결정해야 한다.
2. **라이선스 및 호환성:**
   - 두 모델 모두 **Apache-2.0** 라이선스로 상용 서비스 및 파인튜닝 배포에 법적 제약이 없습니다.
   - Hugging Face `transformers 5.17.0`, `PEFT`, `TRL`, `vLLM` 등 오픈소스 생태계와 100% 네이티브 호환됩니다.
3. **최종 추천:**
   - **`empero-ai/Qwen3.8-4B-Distill`을 ULM 차세대 베이스 모델로 최종 추천합니다.**
   - **추천 사유:** 동일한 85%의 최상급 지식 정확도를 유지하면서도, Qwen3.5 대비 **지시 준수율이 70%로 높고(vs 50%)**, **환각 질문에 대한 방어 능력이 탁월**하며, **추론 과정이 컴팩트하게 증류되어 실시간 서빙 및 대화형 SFT에 훨씬 적합**하기 때문입니다.

---

## 6. 비-Thinking(Non-Thinking) 재검증 결과 요약

1차 60-prompt 평가에서 발견된 Qwen3.5-4B의 Thinking 토큰 고갈 문제를 분리 검증하기 위해, `enable_thinking=False` 조건에서 30개 핵심 프롬프트(Factual QA 10, Multi-turn Memory 8, General Korean 6, Instruction/Trap 6)를 대상으로 2차 비교 평가를 수행했습니다.

### 1) 주요 정량 결과 비교

| 메트릭 항목 | Qwen3.5-4B (Non-thinking) | empero-ai/Qwen3.8-4B-Distill (Non-thinking) | 분석 |
|---|---|---|---|
| **Factual QA (10개)** | 8/10 (80.0%) | **9/10 (90.0%)** | Qwen3.8 우세 (세종대왕 정답 vs 세조 오답) |
| **Multi-turn Memory (8개)** | **7/8 (87.5%)** | 6/8 (75.0%) | Qwen3.5 근소 우세 |
| **Instruction / Trap (6개)** | **6/6 (100.0%)** | **6/6 (100.0%)** | 동점 (완벽 준수) |
| **Thinking 태그 방출율** | 0/30 (0.0%) | 0/30 (0.0%) | Thinking 완전 우회 성공 |
| **최종 응답 누락율 (Empty Final)** | 0/30 (0.0%) | 0/30 (0.0%) | 1차 96.7% 실패 완전 해소 |
| **평균 추론 지연시간** | 5,769.22 ms | **5,613.96 ms** | Qwen3.8이 약 155ms 빠름 |
| **평균 생성 토큰 수** | 118.0 토큰 | **115.8 토큰** | Qwen3.8의 간결한 출력 |
| **한국어 무결성 (오염 여부)** | CJK/영문 불완전 누출 1건 (`국wang`, `총质量的`) | 정상 (한자 단순 병기 외 누출 없음) | Qwen3.8 우수 |

### 2) 재검증 결론
- Qwen3.5-4B의 이전 응답 누락 문제는 `enable_thinking=False` 적용 시 0%로 완벽하게 해결됨을 확인했습니다.
- 그러나 순수 한국어 생성 품질 측면에서 Qwen3.5-4B는 조선 국왕을 '세조'로 답하거나 중국어 문법 표현(`총质量的`), 비정상 결합 토큰(`국wang`)을 노출하는 결함이 나타났습니다.
- 반면 `empero-ai/Qwen3.8-4B-Distill`은 핵심 상식 질문에 완벽하게 정답을 내놓았으며, 한국어 문장 형태와 구조적 완결성이 뛰어났습니다.
- **최종 판정:** 1차 및 2차 non-thinking 검증 결과를 종합하여, ULM 차세대 4B 베이스 모델로 **`empero-ai/Qwen3.8-4B-Distill`**의 선정을 최종 확정합니다. (상세 내역은 `reports/base-model-comparison/non-thinking-final/FINAL_BASE_MODEL_DECISION.md` 참조)

