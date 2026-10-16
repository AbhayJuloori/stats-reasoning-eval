#!/usr/bin/env bash
# Local part of the run: model reviews of the worked answers, the judge, and the position bias test.
# The two solve sets run on Kaggle GPUs (scripts/kaggle/). Cached and resumable.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
MODELS="ollama/qwen3:8b,ollama/llama3.1:8b,ollama/gemma3:12b,ollama/qwen3:14b"
JUDGE="ollama/phi4:14b"

$PY -m evalset.run_review --models "$MODELS"
$PY -m evalset.judge grade --judge "$JUDGE"
$PY -m evalset.judge pairwise --judge "$JUDGE"
$PY -m evalset.judge grade_v2 --judge "$JUDGE"
