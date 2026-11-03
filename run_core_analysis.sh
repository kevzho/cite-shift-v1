#!/usr/bin/env bash
# Reproduce the active deterministic, human-grounded analysis package.
#
# Historical proxies and provider-backed model workflows are intentionally not
# part of this public artifact or runner.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

cd "$ROOT"

echo "[1/3] HCI–Nature OA comparison using canonical human labels"
"$PYTHON_BIN" -B analysis/measurement/06_cross_domain_comparison.py

echo "[2/3] Compression loss in the background category"
"$PYTHON_BIN" -B analysis/measurement/19_compression_loss.py

echo "[3/3] Deterministic tests and output checks"
"$PYTHON_BIN" -B -m unittest \
  analysis/quality/tests/test_research_rules.py \
  analysis/quality/tests/test_annotation_sampling.py
"$PYTHON_BIN" -B analysis/quality/check_outputs.py --strict

echo
echo "Core analysis complete. Results are in analysis/results/."
echo "Private human-annotation inputs remain a separately controlled workflow."
