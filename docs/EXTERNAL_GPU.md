# 외부 GPU 클라우드 학습 가이드 (RunPod, Colab, Vast.ai 등)

이 문서는 외부 GPU 인스턴스(RunPod, Vast.ai, Lambda Labs, Google Colab, AWS, GCP 등)에서 `ULM-1.7B` 알짜 사투리 모델을 학습하는 방법을 안내합니다.

저장소 내에 74,366건의 고밀도 사투리 데이터셋(`data/ulsan_dialect_dense/*.jsonl.gz`)이 8.4MB로 압축되어 깃에 포함되어 있으므로, **별도 데이터 전송 없이 `git clone`만으로 즉시 학습이 가능**합니다.

---

## 1. 예상 학습 시간 및 권장 GPU

| GPU 사양 | VRAM | 예상 학습 시간 (1,428 Steps) | 비고 |
|---|---|---|---|
| **NVIDIA A100 (40G/80G)** | 40GB+ | **약 25 ~ 35분** | 가장 빠름 (초강력 추천) |
| **NVIDIA RTX 4090** | 24GB | **약 40 ~ 50분** | 가성비 최고 |
| **NVIDIA L4** | 24GB | **약 50분 ~ 1시간** | 클라우드 표준 |
| **NVIDIA T4** | 16GB | **약 1시간 30분 ~ 2시간** | 무료/저가 인스턴스 |

---

## 2. 외부 GPU 인스턴스에서 원클릭 실행

클라우드 인스턴스의 터미널(SSH 또는 Jupyter Terminal)에서 아래 3줄만 실행하시면 됩니다:

```bash
# 1. 저장소 클론
git clone https://github.com/UlsanLM-LAB/ULM-1.7B.git
cd ULM-1.7B

# 2. 원클릭 환경 설정 및 학습 자동 실행
bash scripts/train_external.sh
```

---

## 3. 수동 실행 단계 (직접 제어 시)

스크립트 대신 수동으로 실행하고 싶으신 경우:

```bash
# 1. uv 설치
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

# 2. ML 의존성 설치
uv sync --extra ml

# 3. 학습 실행 (대시보드 포함)
uv run python scripts/train_sft.py \
    --config configs/sft/qwen3_1.7b_qlora_dense.yaml \
    --dashboard
```

---

## 4. 백그라운드 무중단 실행 (SSH 연결 끊김 방지)

SSH 창을 닫아도 학습이 끊기지 않게 하려면 `nohup` 또는 `tmux`를 사용하세요:

```bash
# tmux 세션 생성
tmux new -s ulm

# 학습 실행
bash scripts/train_external.sh

# (창 분리: Ctrl + B 누른 뒤 D)
# (다시 접속: tmux attach -t ulm)
```

---

## 5. 학습 완료 후 결과물 로컬로 가져오기

학습이 끝나면 `outputs/qwen3-1.7b-sft-dense/` 폴더에 LoRA 가중치(`adapter_model.safetensors`, 약 140MB)가 생성됩니다.

로컬 컴퓨터 터미널에서:
```bash
# SCP로 다운로드 (서버IP와 포트 번호에 맞게 변경)
scp -P <PORT> -r root@<SERVER_IP>:/workspace/ULM-1.7B/outputs/qwen3-1.7b-sft-dense outputs/
```
