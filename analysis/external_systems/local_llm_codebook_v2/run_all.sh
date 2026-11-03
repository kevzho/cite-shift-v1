#!/usr/bin/env bash
UV_BIN="${UV:-uv}"
export UV_BIN
# Reproduce the local-LLM codebook-v2 feasibility run end to end (localhost ollama only).
set -euo pipefail
cd "$(dirname "$0")"
VENV=~/.cache/citation_corpus/venvs/local_llm_codebook_v2
[ -x "$VENV/bin/python" ] || ${UV_BIN} venv --python 3.12 "$VENV"
PY="$VENV/bin/python"
MODEL=qwen2.5:7b-instruct   # digest 845dbda0ea48ed749caafd9e6037047aa19acfcfd82e704d7ca97d631a0b697e
LOGS=../../results/external_systems/local_llm_codebook_v2/logs; mkdir -p "$LOGS"
$PY run_codebook_v2.py show-prompts
$PY run_codebook_v2.py edges --model $MODEL --run-tag run1 --order forward | tee "$LOGS/run1.log"
ollama stop $MODEL
$PY run_codebook_v2.py edges --model $MODEL --run-tag run2 --order reverse | tee "$LOGS/run2.log"
$PY compare_runs.py run1 run2
$PY run_codebook_v2.py throughput --model $MODEL --run-tag chi200 --sample 200 | tee "$LOGS/throughput_chi200.log"
$PY submit_14_requests_ollama.py   # dry run only
ollama stop $MODEL
