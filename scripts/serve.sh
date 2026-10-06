#!/usr/bin/env bash
# OpenAI-compatible server for Qwen3-4B-Thinking-2507 via MLX (Apple Silicon).
# Endpoint: http://127.0.0.1:${PORT:-8080}/v1  (chat/completions, completions, models)
#
#   scripts/serve.sh                                         # 8-bit MLX model
#   MODEL=models/Qwen3-4B-Thinking-2507 scripts/serve.sh     # BF16 base
#   MODEL=models/<abliterated-dir> scripts/serve.sh          # a modified copy
#   ADAPTER=adapters/my-run scripts/serve.sh                 # with a LoRA adapter
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL="${MODEL:-models/Qwen3-4B-Thinking-2507-mlx-8bit}"
ARGS=()
[[ -n "${ADAPTER:-}" ]] && ARGS+=(--adapter-path "$ADAPTER")

# Sampling defaults follow Qwen's recommendation for the Thinking model;
# per-request values in the API body override them. The thinking trace is
# returned separately in message.reasoning, so max-tokens must leave room for it.
#
# Memory limits sized for a 16 GB Mac. Unbounded, the server keeps the KV
# caches of 10 past requests and decodes 32 at once; long thinking runs then
# exhaust GPU memory and the generation thread dies until restart. Requests
# beyond CONCURRENCY queue rather than fail.
exec .venv/bin/mlx_lm.server \
  --model "$MODEL" \
  --host "${HOST:-127.0.0.1}" --port "${PORT:-8080}" \
  --temp 0.6 --top-p 0.95 --top-k 20 --min-p 0 \
  --max-tokens "${MAX_TOKENS:-32768}" \
  --decode-concurrency "${CONCURRENCY:-2}" \
  --prompt-cache-bytes "$(( ${PROMPT_CACHE_MB:-1024} * 1024 * 1024 ))" \
  ${ARGS[@]+"${ARGS[@]}"} \
  "$@"
