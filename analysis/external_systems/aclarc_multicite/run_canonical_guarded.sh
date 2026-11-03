#!/bin/bash
CITATION_ROOT="${CITATION_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
CITATION_VENVS="${CITATION_VENVS:-$HOME/.cache/citation_corpus/venvs}"
export CITATION_ROOT CITATION_VENVS
# ACL-ARC scaffold (Cohan et al. 2019) on all canonical edges, re-run 21 Oct 2026 (exploratory; the 4 Oct run was
# stopped for low memory at 420/2,929). Same input as SciCite/MultiCite (input_canonical.csv). 2 threads, nice 15;
# memory guard: SIGSTOP when system free memory < 15%, SIGCONT when > 25% (logged).
R=${CITATION_ROOT}
H=$R/analysis/external_systems/aclarc_multicite
O=$R/analysis/results/external_systems/aclarc_multicite
PY=${CITATION_VENVS}/aclarc_multicite/bin/python
LOG=$O/logs/aclarc_canonical_20261021.log
free() { memory_pressure -Q | awk -F': ' '/free percentage/{gsub("%","",$2); print $2}'; }
echo "$(date '+%F %T') start (free $(free)%)" >> "$LOG"
nice -n 15 $PY $H/run_aclarc.py --tag canonical_20261021 --threads 2 \
  --edges $R/analysis/results/external_systems/hosted_llm_codebook_v2/input_canonical.csv >> "$LOG" 2>&1 &
P=$!
paused=0; peak=0
while kill -0 $P 2>/dev/null; do
  f=$(free); rss=$(ps -o rss= -p $P 2>/dev/null | tr -d ' '); [ -n "$rss" ] && [ "$rss" -gt "$peak" ] && peak=$rss
  if [ $paused -eq 0 ] && [ "$f" -lt 15 ]; then kill -STOP $P; paused=1; echo "$(date '+%F %T') PAUSED free ${f}%" >> "$LOG"; fi
  if [ $paused -eq 1 ] && [ "$f" -gt 25 ]; then kill -CONT $P; paused=0; echo "$(date '+%F %T') RESUMED free ${f}%" >> "$LOG"; fi
  sleep 15
done
wait $P; rc=$?
echo "$(date '+%F %T') done rc=$rc peak_rss_MB=$((peak/1024)) lines=$(wc -l < $O/outputs/aclarc_scaffold_canonical_20261021.jsonl 2>/dev/null)" >> "$LOG"
