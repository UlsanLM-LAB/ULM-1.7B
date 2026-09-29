# ULM-4B v1.0 Model Card

## Model summary

ULM-4B v1.0은 울산 방언의 이해와 생성을 목표로 하는 4B급 소형 언어모델의 Stable Release입니다. 기존 Arm B base model과 LoRA adapter를 그대로 동결하며 추가 학습 계획은 없습니다. rejected Arm B+와 final low-LR 실험은 연구 기록으로만 유지합니다.

- Product: ULM-4B v1.0
- Organization: UlsanLM Lab
- Status: v1.0 Stable Release, 2026-09-29
- Internal checkpoint lineage: Arm B / Instruction Recovery v1 Arm B
- Base: [empero-ai/Qwen3.8-4B-Distill](https://huggingface.co/empero-ai/Qwen3.8-4B-Distill), revision `c83cb7aa2999d2f35c43e9ae0634a30eb8985a1e`
- Fine-tuning: LoRA SFT; existing r8 / alpha16 / dropout0.05
- Runtime: PyTorch + Hugging Face Transformers; FastAPI chat API
- Voice integration: [ULM-LIVE](https://github.com/UlsanLM-LAB/ULM-LIVE) + Qwen3-TTS
- Release contract: [ulm4b-v1.0.json](configs/release/ulm4b-v1.0.json)

Stable identifies the frozen version and verified release code. It does not claim that all historical research quality gates passed or that the model is suitable for every production use.

## Training lineage and frozen weights

```text
Qwen3.8-4B-Distill
  → Dialect Alignment v3
  → Context Repair v1 Arm B
  → Instruction Recovery v1 Arm B
  → ULM-4B v1.0 (Arm B frozen)
```

The final existing Arm B SFT used learning rate `6e-7`, fraction `0.25`, 35 steps and completion-only loss. This release performs no training or weight merge. The original [Arm B release configuration](configs/release/ulm4b-arm-b.json) is unchanged.

Frozen `adapter_model.safetensors` SHA-256:

```text
fee03f872aaff55283be0413536e20b0930334333a20c6ee01cad63f57f43716
```

Weights and adapters are not distributed in this Git repository or attached to this code release. Obtain the authorized existing Arm B artifact separately and check its digest; another adapter or newly downloaded upstream revision is a different artifact.

## Evaluation

Original 2026-09-27 Arm B release snapshot, preserved without retraining:

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

The separate 2026-09-28 neutral 500-item comparison measured generation semantic `0.9841`, generation dialectness `0.6591`, grammar dialectness `0.8909`, context semantic `0.4594` and identification `30%`. Its context-only guard ablation measured context semantic `0.4656` and zero context repetitions. These runs use different decoding conditions from the release snapshot and are not merged into one experiment.

The context benchmark contains conversational-next-turn references and should not be interpreted as a complete measure of open-ended conversation quality. Semantic similarity and dialectness are automated proxies; no new native-speaker human evaluation was conducted for v1.0.

Historical throughput counted padding after EOS and may be overstated. Corrected Arm B runtime verification is recorded separately in the [Stable Release report](reports/ULM_4B_V1_0_STABLE_RELEASE.md). Historical score files and the evaluator rubric remain unchanged.

## Inference policy

The frozen Arm B policy uses non-thinking chat templates and anti-loop decoding:

```text
enable_thinking = false
repetition_penalty = 1.10
no_repeat_ngram_size = 3
```

Existing defaults remain temperature `0.7`, top-p `0.9`, top-k `20`, max new tokens `150` and dialect strength `2`. The HTTP server loads base and adapter separately through `ULM_MODEL_PATH` and `ULM_ADAPTER_PATH`. `/v1/chat/completions`, `/api/chat` and the ULM-LIVE SSE `token` field remain compatible.

API dialect strength is an integer `0–3`: standard, mild, Ulsan, strong. The ULM-LIVE UI exposes `1/2/3`. API and CLI reuse the same prompt helper. Explicit user style and output-format requests take priority over the default style. Prompt control does not guarantee monotonic model outputs.

## Intended use

- 울산 방언 생성·이해·변환 연구
- 지역어 챗봇과 ULM-LIVE 텍스트 응답
- 교육·포트폴리오·연구 데모
- 소형 언어모델의 지역어 적응 및 평가 연구

## Known limitations

- Instruction-trap is 65%; strict output constraints and false-premise rejection remain weak. The historical research instruction gate still fails.
- Identification is 32% in the release snapshot and 30% in the neutral comparison; robust regional classification is not established.
- Automated dialectness and semantic scores do not replace native Ulsan speaker evaluation. Human dialect-naturalness and voice-listening evaluation remain open.
- The model can hallucinate or give factually incorrect responses.
- Ulsan dialect varies by speaker, generation and region; one output style does not represent the whole dialect.
- Dialect strength is prompt-based and does not use a learned control token.
- Full deployed-mode 500-item quality evaluation, live TTS/STT and speech interruption were not revalidated by the v1.0 text smoke.
- Original restricted benchmark inputs and model artifacts are not publicly bundled; published historical scores alone cannot reproduce the experiment.

These are documented scope and quality limitations for the stated research/demo uses. They are not marked as fixed by the release-path code corrections.

## Data, privacy and licenses

Large original datasets, private audio, model weights and adapters are excluded from Git. AI Hub and other data retain their source usage and privacy conditions.

Code is [Apache-2.0](LICENSE). The [base model card](https://huggingface.co/empero-ai/Qwen3.8-4B-Distill) lists Apache-2.0; base, datasets, evaluation encoder, TTS and derivative weights retain their respective upstream notices and terms. A code release is not a blanket license for all artifacts.

## Citation and references

Use [CITATION.cff](CITATION.cff) or the versioned BibTeX in [README](README.md#citation). The repository URL remains `https://github.com/UlsanLM-LAB/ULM-1.7B`; update it after any repository rename.

Architecture and source references are listed in [README References](README.md#references). See the [original Arm B decision](reports/ULM_4B_ARM_B_RELEASE_CANDIDATE.md), [model comparison](reports/ULM_4B_MODEL_COMPARISON.md) and [v1.0 release notes](reports/ULM_4B_V1_0_RELEASE_NOTES.md).
