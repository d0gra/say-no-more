#!/usr/bin/env bash
# Download the BF16 base model (~7.5 GB) and make the 8-bit MLX copy (~4 GB)
# that the server runs. Weights live in models/, which git ignores.
# Resumable: rerun after an interrupted download.
set -euo pipefail
cd "$(dirname "$0")/.."

BASE_ID="Qwen/Qwen3-4B-Thinking-2507"
BASE_DIR="models/Qwen3-4B-Thinking-2507"
MLX_DIR="models/Qwen3-4B-Thinking-2507-mlx-8bit"

.venv/bin/hf download "$BASE_ID" --local-dir "$BASE_DIR"
if [[ ! -d "$MLX_DIR" ]]; then
  .venv/bin/mlx_lm.convert --hf-path "$BASE_DIR" --mlx-path "$MLX_DIR" -q --q-bits 8
fi
