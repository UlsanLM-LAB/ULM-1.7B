# ULM-1.7B Phase 3 Chat Alignment & Standalone Merge Report

## 1. 개요 및 배경

ULM-1.7B Phase 2 모델 평가 중 사용자 입력에 대해 단순 에코(Echo)를 수행하거나 동문서답을 출력하는 현상이 발견되었습니다. 원인 정밀 진단 결과 두 가지 독립적인 결함이 확인되었습니다:
1. **Web Frontend Mock Provider 버그**: 실제 백엔드 연결 지연 또는 에러 발생 시 ULM-chat 프론트엔드가 고정된 마케팅 템플릿 문구(`"게이야에 대해 물어보셨네예..."`)를 목업으로 반환하던 문제.
2. **Phase 2 훈련 데이터 Task Mismatch**: Phase 2 모델이 100% `standard_to_dialect` 형태의 단문 번역 데이터로만 학습되어, 어시스턴트로서 질문에 답변하는 것이 아니라 사용자의 모든 질문 입력을 1:1 사투리로 변환(에코)하는 동작을 보임.

이에 따라 ULM-chat의 Mock 자동 fallback을 차단하고, 질문에 실질적으로 답변하면서 자연스러운 울산 어조를 유지하는 **Phase 3 Chat Alignment**를 수행했습니다.

---

## 2. Phase 3 데이터셋 구축

- **총 데이터 규모**: 9,552건
  - `train.jsonl`: 8,596건
  - `validation.jsonl`: 956건
- **대화 구조**:
  - Multi-turn 비율: 25.9% (2,478건)
  - Single-turn 비율: 74.1% (7,074건)
- **카테고리 구성**:
  - 일상 자유 대화 (Casual Chat & Banter)
  - 일반 지식 및 사실 질의 (Factual QA, Science)
  - 코딩 및 알고리즘 설명 (Python, JS, SQL 등)
  - 기초 및 복합 연산 (Math)
  - 감정 공감 및 고민 상담 (Empathy & Counseling)
  - 울산 명소/문화/특산물 질의 (Ulsan Domain QA)
- **무결성 검증**:
  - High-similarity Echo: 0건
  - Duplicate: 0건
  - Format/Schema Validation: 100% 통과

---

## 3. AWS EC2 L40S 환경 및 20-Step 벤치마크

모든 훈련, 스모크, 벤치마크는 AWS EC2 `g6e.xlarge` (NVIDIA L40S 46GB VRAM) 환경에서 수행되었습니다.

### 20-Step 벤치마크 실측치

| 후보군 | 배치 설정 | Step Time | Peak VRAM | 소요 시간 | 평가 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Candidate A** | `batch: 8, ga: 2` (Eff. 16) | 3.790 s/step | 11.11 GiB (24.1%) | 75.8s | 그래디언트 누적 오버헤드로 인한 처리 지연 |
| **Candidate B** | `batch: 16, ga: 1` (Eff. 16) | **1.375 s/step** | **17.30 GiB (37.6%)** | **27.5s** | **2.75배 빠른 처리 속도**, L40S 대역폭 최적화 (**최종 선택**) |

---

## 4. Phase 3 Full Training 결과

- **기반 Base 모델**: `outputs/ulm-1.7b-phase2-merged`
- **총 Epochs / Steps**: 2 Epochs / 1,076 Steps
- **총 훈련 소요 시간**: 373.2초 (약 6분 13초)
- **Step당 평균 소요 시간**: 0.347초/step
- **Peak VRAM**: 19.61 GiB (42.6%)
- **손실률 추이**:
  - Train Loss: 2.16 ➔ 0.72 ➔ 0.59
  - Eval Loss: Epoch 1 (`step-537`) **0.7398** ➔ Epoch 2 (`step-1074`) **0.7365**
  - Eval Token Accuracy: Epoch 1 **84.85%** ➔ Epoch 2 **84.87%**

---

## 5. Checkpoint 선정: checkpoint-537 (Best)

20개 핵심 프롬프트 정밀 생성 비교 결과:
- **Epoch 2 (`checkpoint-1074/1076`)**: Eval Loss 개선폭이 0.003에 불과한 반면, Qwen 베이스 모델의 잠재 한자/다국어 토큰(`Both都要`, `今天`, `下去`, `啦`)이 일부 누출되는 과적합 징후가 포착됨.
- **Epoch 1 (`checkpoint-537`)**: 한국어 답변이 정갈하고 자연스러운 울산말 종결형(`~데이`, `~아이가`, `~이제`)을 완벽하게 구사하며 다국어 오염이 전혀 없음.
- **결론**: `checkpoint-537`을 최종 프로덕션 체크포인트로 확정.

---

## 6. Standalone 모델 병합 및 무결성 검증

- **Base**: `outputs/ulm-1.7b-phase2-merged`
- **Adapter**: `outputs/ulm-1.7b-phase3-chat-l40s/checkpoint-537`
- **산출물**: `outputs/ulm-1.7b-phase3-best-merged` (Standalone HuggingFace BF16)
- **병합 검증 (10개 프롬프트)**:
  - LoRA 어댑터 추론과 Merged 단독 모델 추론 간 응답 의미 및 형식 일치 확인 (무결성 통과).

---

## 7. 105개 프롬프트 종합 회귀 테스트 결과

13개 카테고리(일상, 단문, 슬랭, 수학, 상식, 코딩, 과학, 공감, 추천, 멀티턴, 명확화, 울산, 표준어) 105개 문장 테스트 결과:
- **전체 통과율**: **90.5%** (95 / 105)
- **단순 Echo율**: **0.0%** (0 / 105, 완전 해소)
- **질문 무시율**: **0.0%** (0 / 105, 완전 해소)
- **다국어 유출**: 9건 (과학/상식 전문 용어 설명 시 고립 한자 1~4자 수준)
- **반복 생성**: 0건 (정상 파이썬 코드 문법 확인)

---

## 8. Web E2E 및 ULM-LIVE Thinker 연동 검증

1. **ULM-chat Web E2E (포트 3001 -> 8008)**:
   - 브라우저 -> Next.js `/api/chat` -> FastAPI 추론 서버 (`outputs/ulm-1.7b-phase3-best-merged`) -> 실시간 SSE 스트리밍 정상 확인.
   - Mock Provider 자동 fallback 차단 유지.
2. **ULM-LIVE Thinker 연동**:
   - `inspect_thinker.py` 검증 완료:
   - Hidden size: `2048`, Layers: `28`, Vocab size: `151,669`, Hidden shape: `(1, 13, 2048) OK`.
