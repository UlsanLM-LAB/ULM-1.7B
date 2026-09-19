#!/usr/bin/env bash
set -euo pipefail

echo "============================================================"
echo "  ULM-1.7B External GPU Training Setup & Runner"
echo "============================================================"

# 1. Check GPU
if command -v nvidia-smi &> /dev/null; then
    echo "[INFO] GPU detected:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    echo "[WARNING] nvidia-smi not found. GPU acceleration may not be available."
fi

# 2. Install uv if not present
if ! command -v uv &> /dev/null; then
    echo "[INFO] Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

# 3. Install ML dependencies
echo "[INFO] Installing dependencies via uv..."
uv sync --extra ml

# 4. Run SFT training
echo "[INFO] Launching training with configs/sft/qwen3_1.7b_qlora_dense.yaml..."
uv run python scripts/train_sft.py \
    --config configs/sft/qwen3_1.7b_qlora_dense.yaml \
    --dashboard
