#!/usr/bin/env bash
# Create .venv (Python 3.12) with the pinned dependencies.
# Needs uv:  brew install uv
set -euo pipefail
cd "$(dirname "$0")/.."

uv venv .venv --python 3.12
uv pip install --python .venv/bin/python -r requirements.txt
