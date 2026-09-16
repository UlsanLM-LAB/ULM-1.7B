#!/usr/bin/env bash
set -euo pipefail

# Hugging Face CAS/XET 에러 방지
export HF_HUB_DISABLE_XET=1

echo "========================================================="
echo "  ULM-1.7B SFT 학습 및 실시간 대시보드를 시작합니다."
echo "========================================================="

uv run python scripts/train_sft.py --config configs/sft/qwen3_1.7b_qlora.yaml --dashboard
