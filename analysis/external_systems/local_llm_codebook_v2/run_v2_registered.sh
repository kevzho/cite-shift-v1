#!/bin/bash
CITATION_ROOT="${CITATION_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
PYTHON="${PYTHON:-python3}"
export CITATION_ROOT PYTHON
# Registered local-LLM system (pre-registration §4/§7, OSF osf.io/nfz9c), run 12 Oct 2026: Qwen2.5-7B-Instruct
# (digest 845dbda0), frozen codebook-v2 prompts (pass1 a3140ade…, pass2 3cda7d78…, = Sol's), greedy, seed 20260926.
# 1) all 550 v2 edges (B 300 + A250); 2) two fresh runs of the first 100 of B (same input as Sol's stability runs).
# Resume-safe: rerun this script; finished tags are skipped, partial ones resume from raw_<tag>.jsonl.
cd "$(dirname "$0")"
R=${CITATION_ROOT}
O=$R/analysis/results/external_systems/local_llm_codebook_v2
PY=~/.cache/citation_corpus/venvs/local_llm_codebook_v2/bin/python
M=qwen2.5:7b-instruct
mkdir -p "$O/logs"
for spec in "v2edges:$O/input_v2_edges.csv" "s1:$R/analysis/results/external_systems/hosted_llm_codebook_v2/input_stability_B100.csv" "s2:$R/analysis/results/external_systems/hosted_llm_codebook_v2/input_stability_B100.csv"; do
  tag=${spec%%:*}; inp=${spec#*:}
  if [ -f "$O/edges_$tag.csv" ]; then echo "$(date '+%F %T') $tag exists, skip" >> "$O/logs/run_log_2026-10-12.txt"; continue; fi
  echo "$(date '+%F %T') $tag start" >> "$O/logs/run_log_2026-10-12.txt"
  ollama stop $M 2>/dev/null
  $PY run_codebook_v2.py edges --model $M --run-tag $tag --input "$inp" >> "$O/logs/$tag.log" 2>&1
  echo "$(date '+%F %T') $tag done rc=$?" >> "$O/logs/run_log_2026-10-12.txt"
done
ollama stop $M 2>/dev/null
echo "$(date '+%F %T') QWEN ALL DONE" >> "$O/logs/run_log_2026-10-12.txt"
# Addendum scoring (Qwen into RQ3 + inter-LLM agreement + run-to-run), DEVIATIONS.md #6
cd "$R" && ${PYTHON} analysis/experiments/33_qwen_addendum.py > "$O/logs/addendum.log" 2>&1
echo "$(date '+%F %T') ADDENDUM done rc=$?" >> "$O/logs/run_log_2026-10-12.txt"
