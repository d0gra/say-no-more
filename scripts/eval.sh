#!/usr/bin/env bash
# Run the full refusal suite (39 harmful + 10 benign prompts) against whatever
# scripts/serve.sh is serving, with the settings every run in this repo uses.
# Answers, thinking and keyword verdicts go to results/<run>/raw.json, and the
# report to results/<run>/report.keyword.txt.
#
#   scripts/eval.sh 02-abliterated
set -euo pipefail
cd "$(dirname "$0")/.."

RUN="${1:?usage: scripts/eval.sh <run-name>}"
mkdir -p "results/$RUN"

exec .venv/bin/python eval/refusal_eval.py \
  --base-url "${BASE_URL:-http://127.0.0.1:8080/v1}" --model default_model --api-key none \
  --workers 2 --max-tokens 8192 \
  --output "results/$RUN/report.keyword.txt" \
  --json-output "results/$RUN/raw.json"
