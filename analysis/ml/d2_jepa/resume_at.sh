#!/bin/bash
CITATION_ROOT="${CITATION_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
export CITATION_ROOT
# Resume paused D2 training at a set time (default 01:00): SIGCONT every pretrain process.
T=${1:-01:00}
L=${CITATION_ROOT}/analysis/results/ml/d2_jepa/runs_mps/run_log.txt
target=$(date -j -f "%Y-%m-%d %H:%M" "$(date +%Y-%m-%d) $T" +%s); now=$(date +%s)
[ $target -le $now ] && target=$((target + 86400))
sleep $((target - now))
for p in $(pgrep -f "d2_jepa/pretrain.py"); do kill -CONT $p; done
echo "$(date '+%F %T') RESUMED (SIGCONT) by resume_at.sh" >> $L
