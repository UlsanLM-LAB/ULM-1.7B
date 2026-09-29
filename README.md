# ULM-4B v1.0

UlsanLM Lab에서 개발하는 울산 방언 특화 4B급 소형 언어모델입니다. 현재 Stable Release는 `ULM-4B v1.0`(내부 checkpoint lineage: Arm B)이며, 울산 방언 생성·이해와 일반 응답 능력 보존을 함께 목표로 합니다.

> Repository: [UlsanLM-LAB/ULM-4B](https://github.com/UlsanLM-LAB/ULM-4B) · ULM-4B v1.0

![Python](https://img.shields.io/badge/Python-3.11%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange)
![Transformers](https://img.shields.io/badge/Hugging%20Face-Transformers-yellow)
![FastAPI](https://img.shields.io/badge/API-FastAPI-009688)
![License](https://img.shields.io/badge/Code%20License-Apache--2.0-green)
![Status](https://img.shields.io/badge/Status-v1.0%20Stable%20Release-brightgreen)

[Model Card](MODEL_CARD.md) · [Benchmark Report](reports/ULM_4B_MODEL_COMPARISON.md) · [Release Report](reports/ULM_4B_V1_0_STABLE_RELEASE.md) · [Release Notes](reports/ULM_4B_V1_0_RELEASE_NOTES.md) · [ULM Live](https://github.com/UlsanLM-LAB/ULM-LIVE)

## Release snapshot

| 항목 | 내용 |
| --- | --- |
| Model | ULM-4B v1.0 (lineage: Arm B) |
| Status | v1.0 Stable Release; weights frozen |
| Release date | 2026-09-29 |
| Original Arm B snapshot | 2026-09-27 |
| Base model | [empero-ai/Qwen3.8-4B-Distill](https://huggingface.co/empero-ai/Qwen3.8-4B-Distill) |
| Parameter class | 4.21B |
| Adaptation | LoRA SFT |
| Training lineage | Dialect Alignment v3 → Context Repair v1 Arm B → Instruction Recovery v1 Arm B |
| Serving | FastAPI / OpenAI-compatible chat API |
| Voice | ULM-LIVE + Qwen3-TTS |
| Default dialect strength | 2 / Ulsan |
| Thinking | disabled |
| Repetition control | `repetition_penalty=1.10`, `no_repeat_ngram_size=3` |
| Code license | Apache-2.0 |
| Weights | 대형 체크포인트는 이 Git 저장소에 포함하지 않음 |

ULM-4B v1.0은 기존 Arm B 가중치를 그대로 동결한 Stable Release입니다. 추가 학습 계획은 없으며 rejected Arm B+는 연구 기록으로만 유지합니다. Stable은 릴리즈 코드·설정·체크포인트의 고정을 뜻합니다. instruction-trap과 방언 식별 성능, 사람 평가 부재 등 아래 Known Limitations는 유지되며 모든 연구 gate 통과나 모든 production 용도의 적합성을 뜻하지 않습니다.

## Model overview

ULM-4B는 범용 4B 모델을 울산 지역어에 맞게 추가 적응한 모델 계열입니다. v1.0에 고정한 Arm B는 다음 세 단계의 학습 계보를 갖습니다.

```text
empero-ai/Qwen3.8-4B-Distill
  → Dialect Alignment v3
  → Context Repair v1 Arm B
  → Instruction Recovery v1 Arm B
  → ULM-4B v1.0 (Arm B frozen)
```

학습은 LoRA 기반 supervised fine-tuning을 사용하며, v1.0 릴리스 설정은 [configs/release/ulm4b-v1.0.json](configs/release/ulm4b-v1.0.json)에 고정되어 있습니다. 원 Arm B release config도 함께 보존합니다.

주요 목적:

- 울산 방언 이해 및 생성
- 표준어 ↔ 울산 방언 변환 연구
- 지역어 챗봇 및 교육·포트폴리오 데모
- ULM Live 음성 대화 파이프라인의 텍스트 모델
- 소형 언어모델의 지역어 적응 및 평가 연구

## Evaluation

### Frozen Arm B evaluation snapshot

아래 값은 2026-09-27 Arm B 선택 당시의 내부 평가 스냅샷입니다. v1.0 승격은 재학습 없이 동일한 값을 유지합니다.

| 평가 | Arm B |
| --- | ---: |
| UlsanBench generation semantic | 0.9843 |
| UlsanBench generation dialectness | 0.6589 |
| UlsanBench grammar dialectness | 0.8955 |
| UlsanBench context semantic | 0.4578 |
| Context repetition | 0 |
| Factual QA | 84% |
| Multi-turn | 100% |
| Instruction trap | 65% |
| UlsanBench identification accuracy | 32% |

이 값들은 기존 Arm B 선택 시점의 결과이며, 아래 모델 비교 벤치마크와는 실행 조건 및 평가 시점이 일부 다르므로 동일한 실험으로 합쳐 해석하지 않습니다.

### UlsanBench v2 model comparison

동일한 500개 평가 항목과 동일한 평가 파이프라인에서 ULM-4B v1.0의 Arm B와 비교 모델을 직접 실행한 결과입니다.

![UlsanBench v2 모델 비교](assets/benchmarks/ulsanbench-model-comparison.svg)

평가 조건:

- Date: 2026-09-28
- Hardware: AWS EC2 `g6e.xlarge`, NVIDIA L40S 46GB, `ap-northeast-2`
- Decoding: `do_sample=false`, `temperature=0`, `enable_thinking=false`
- Prompting: 각 모델의 공식 tokenizer chat template, 동일 zero-shot system instruction
- Evaluator: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` + 울산 방언 어휘·종결어미 규칙
- Caveat: semantic similarity와 dialectness는 자동 proxy 지표이며 사람의 방언 자연스러움 평가를 대체하지 않습니다.

| 모델 | 파라미터 | Gen Semantic | Gen Dialect | Grammar Dialect | Context Semantic | Identification | Comprehension |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| ULM-4B v1.0 (Neutral) | 4.21B | 0.9841 | 0.6591 | 0.8909 | 0.4594 | 0.3000 | 0.9802 |
| ULM-4B v1.0 (Context Guard*) | 4.21B | 0.9841 | 0.6591 | 0.8909 | 0.4656 | 0.3000 | 0.9802 |
| Qwen3.8-4B-Distill (Base) | 4.21B | 0.8840 | 0.5629 | 0.7790 | 0.3593 | 0.2800 | 0.8220 |
| Qwen2.5-3B-Instruct | 3.09B | 0.6888 | 0.4537 | 0.6303 | 0.4186 | 0.2100 | 0.8150 |
| Qwen2.5-7B-Instruct | 7.61B | 0.7766 | 0.5007 | 0.7241 | 0.3859 | 0.2700 | 0.7193 |
| Llama-3.2-Korean-Bllossom-3B | 3.21B | 0.8695 | 0.5548 | 0.7530 | 0.3792 | 0.3300 | 0.8228 |

*Context Guard는 보조 ablation으로 Context 작업에만 `repetition_penalty=1.10`, `no_repeat_ngram_size=3`를 적용했습니다. 전체 production decoding을 그대로 재현한 별도 500-item rerun은 아닙니다.*

이 벤치마크에서는 동일 Arm B를 사용하는 ULM-4B v1.0이 base model보다 generation dialectness, grammar dialectness, comprehension semantic에서 높은 proxy score를 기록했습니다. Identification은 비교 모델 전체가 낮은 정확도를 보였고, Bllossom-3B가 이 항목에서는 ULM-4B보다 높은 값을 기록했습니다.

기존 처리량에는 EOS 뒤 batch padding을 포함한 오류가 있었습니다. 과거 수치는 기록으로 보존하며 v1.0의 검증된 성능 주장에 사용하지 않습니다. Arm B의 수정 후 처리량 재검증은 [Stable Release Report](reports/ULM_4B_V1_0_STABLE_RELEASE.md)에 별도로 기록합니다. 점수 산식과 역사적 semantic/dialectness 수치는 유지합니다.

세부 설정, 런타임, VRAM, 과제별 결과는 [ULM-4B Model Comparison Report](reports/ULM_4B_MODEL_COMPARISON.md)를 참고하세요.

## Architecture

```text
Browser / Client
       │
       ▼
FastAPI chat server
       │
       ▼
Qwen3.8-4B-Distill
       +
ULM-4B v1.0 LoRA (Arm B)
       │
       ├── text response
       │
       └── ULM-LIVE
              │
              ▼
        Qwen3-TTS
              │
              ▼
          24 kHz WAV
```

음성 경로는 별도 저장소 [UlsanLM-LAB/ULM-LIVE](https://github.com/UlsanLM-LAB/ULM-LIVE)에서 관리합니다.

## Quick start

개발 환경:

```bash
uv sync --extra dev --extra ml --extra tts
uv run python scripts/check_project.py
```

전체 테스트에는 TTS 전처리·추론 테스트도 포함되어 `tts` extra가 필요합니다. 위 명령은 `ruff check src`와 CPU 회귀 테스트를 순서대로 실행하고 실패 시 중단합니다. 개별 검사는 `uv run pytest`, `uv run ruff check src`로 실행할 수 있습니다. v1.0 검증 내역은 [Stable Release Report](reports/ULM_4B_V1_0_STABLE_RELEASE.md)에 기록합니다.

모델 가중치와 원본 대형 데이터는 저장소에 포함하지 않습니다.

### Run ULM-4B v1.0 API

v1.0은 기존 Arm B의 base model과 LoRA adapter를 분리해서 로드합니다. 서버 기본값과 API 경로는 유지합니다.

```bash
export ULM_MODEL_PATH=/path/to/Qwen3.8-4B-Distill
export ULM_ADAPTER_PATH=/path/to/ulm4b-arm-b/final_adapter

uv run python scripts/serve.py \
  --model "$ULM_MODEL_PATH" \
  --host 127.0.0.1 \
  --port 8000
```

Health check:

```bash
curl http://127.0.0.1:8000/health
```

Non-streaming chat:

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "messages": [{"role": "user", "content": "오늘 뭐하노?"}],
    "stream": false,
    "temperature": 0,
    "dialect_strength": 2
  }'
```

SSE streaming은 같은 endpoint에서 `"stream": true`로 사용할 수 있습니다.

## Dialect Strength

Text API는 `dialect_strength` 정수 필드로 응답의 기본 방언 강도를 조정합니다.

| 값 | API 이름 | 동작 |
| --- | --- | --- |
| 0 | Standard | 방언을 의도적으로 억제 |
| 1 | Mild | 표준어 중심 + 가벼운 울산 표현 |
| 2 | Ulsan | 자연스러운 울산 일상 말투, 기본값 |
| 3 | Strong | 울산 어휘와 종결어미를 적극 사용 |

```json
{
  "messages": [{"role": "user", "content": "오늘 뭐하노?"}],
  "stream": true,
  "dialect_strength": 2
}
```

- API 범위: 0–3
- Default: 2
- ULM Live 웹 UI: 1 / 2 / 3만 노출
- `0`은 API 호환성·직접 호출용으로 유지
- 음수, 4, 문자열, `null`, boolean, 실수는 422
- stream/non-stream 모두 동일한 prompt path 사용
- 강도 지시는 history에 누적하지 않음

현재 구현은 learned control token이 아니라 system prompt 기반 style control입니다. 사용자가 직접 지정한 말투·출력 형식은 기본 dialect strength보다 우선하도록 설계되어 있습니다.

## Decoding policy

v1.0의 기존 Arm B 기본 추론 정책:

```text
enable_thinking = false
repetition_penalty = 1.10
no_repeat_ngram_size = 3
```

샘플링을 사용할 때는 서버의 `temperature`, `top_p` 등 generation 옵션을 함께 사용합니다.

## Repository layout

```text
src/ulm/                 모델 학습·추론 핵심 코드
scripts/                 데이터 구축, 학습, 평가, 서빙 스크립트
configs/                 release metadata 및 선택적 TTS 설정
data/                    공개 fixture 및 데이터 운영 규칙
benchmarks/              평가 설계 및 규칙
reports/                 릴리스·벤치마크 결과
portfolio/               포트폴리오용 시각 자료
tests/                   회귀 및 API 테스트
```

실험 산출물, 원본 제한 데이터, 대형 체크포인트는 Git에 직접 커밋하지 않습니다.

## Reproducibility and release files

- [MODEL_CARD.md](MODEL_CARD.md): ULM-4B model card
- [configs/release/ulm4b-v1.0.json](configs/release/ulm4b-v1.0.json): v1.0 고정 설정·base revision·Arm B adapter SHA-256
- [configs/release/ulm4b-arm-b.json](configs/release/ulm4b-arm-b.json): 변경하지 않은 원 Arm B 설정
- [ULM-4B v1.0 Stable Release Report](reports/ULM_4B_V1_0_STABLE_RELEASE.md): 수정·검증·동결 결정
- [ULM-4B v1.0 Release Notes](reports/ULM_4B_V1_0_RELEASE_NOTES.md): 배포 안내와 한계
- [ULM-4B Arm B Release Candidate Report](reports/ULM_4B_ARM_B_RELEASE_CANDIDATE.md): Arm B 선택 근거
- [ULM-4B Model Comparison Report](reports/ULM_4B_MODEL_COMPARISON.md): 500-item 비교 벤치마크
- [benchmarks/README.md](benchmarks/README.md): benchmark 작성 및 검수 규칙
- [data/README.md](data/README.md): 데이터 거버넌스
- [scripts/README.md](scripts/README.md): 실행 진입점과 실험 스크립트 구분
- [reports/README.md](reports/README.md): 릴리스·벤치마크·연구 보고서 안내

## Known Limitations

- Instruction-trap snapshot은 65%로, 엄격한 출력 형식과 지시 추종에서 추가 개선이 필요합니다.
- Dialect identification accuracy는 release snapshot 기준 32%이며, 울산/타 경상/표준어 분류를 안정적으로 해결했다고 볼 수 없습니다.
- UlsanBench의 semantic/dialectness 지표는 자동 proxy이므로 native speaker human evaluation을 대체하지 않습니다.
- 울산 방언은 지역, 세대, 화자에 따라 변이가 크므로 모델의 한 가지 출력 스타일을 전체 울산 방언으로 일반화하면 안 됩니다.
- 모델은 일반 언어모델과 마찬가지로 사실 오류나 환각을 생성할 수 있습니다.
- Dialect Strength는 현재 prompt-based control이며 강도별 출력 특성이 학습된 control token으로 보장되지 않습니다.
- ULM Live의 음성 억양 품질과 자연스러움은 별도 human listening 검증이 필요합니다.

## Citation

연구, 발표, 보고서에서 이 저장소를 인용할 때는 아래 형식을 사용할 수 있습니다.

```bibtex
@software{ulsanlm_ulm4b_2026,
  author       = {{UlsanLM Lab}},
  title        = {ULM-4B v1.0: Ulsan Dialect Small Language Model},
  version      = {1.0.0},
  year         = {2026},
  url          = {https://github.com/UlsanLM-LAB/ULM-4B},
  note         = {Stable Release; frozen Arm B checkpoint}
}
```

기계 판독 가능한 인용 정보는 [CITATION.cff](CITATION.cff)에 있습니다.

## References

1. Empero. “Qwen3.8-4B-Distill.” Hugging Face model card. https://huggingface.co/empero-ai/Qwen3.8-4B-Distill
2. Hu, E. J., Shen, Y., Wallis, P., et al. “LoRA: Low-Rank Adaptation of Large Language Models.” arXiv:2106.09685, 2021. https://arxiv.org/abs/2106.09685
3. Reimers, N., & Gurevych, I. “Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks.” EMNLP-IJCNLP 2019. https://aclanthology.org/D19-1410/
4. Sentence Transformers. “paraphrase-multilingual-MiniLM-L12-v2.” https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
5. Qwen Team. “Qwen3-TTS.” https://github.com/QwenLM/Qwen3-TTS
6. Hugging Face. “Transformers.” https://github.com/huggingface/transformers
7. Hugging Face. “PEFT: Parameter-Efficient Fine-Tuning.” https://github.com/huggingface/peft

외부 base model, 데이터셋, 평가 모델, TTS 모델은 각 원 프로젝트의 라이선스와 이용 조건을 별도로 확인해야 합니다.

## License

이 저장소의 코드는 [Apache License 2.0](LICENSE)을 따릅니다. Base model, 외부 데이터셋, 평가 모델, 음성 모델 및 파생 가중치에는 각각의 upstream 라이선스와 데이터 이용 조건이 별도로 적용될 수 있습니다.
