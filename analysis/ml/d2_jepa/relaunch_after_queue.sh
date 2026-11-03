#!/bin/bash
CITATION_ROOT="${CITATION_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
export CITATION_ROOT
# Wait for the running slow queue to finish, then run it again: finished runs are skipped,
# JEPA seed 1 resumes from its step-500 checkpoint.
O=${CITATION_ROOT}/analysis/results/ml/d2_jepa/runs_mps
until [ $(grep -c "SLOW QUEUE ALL DONE" $O/run_log.txt) -gt 2 ]; do sleep 300; done
echo "$(date '+%F %T') follow-up relaunch (resume JEPA seed 1)" >> $O/run_log.txt
exec ${CITATION_ROOT}/analysis/ml/d2_jepa/mac_run_all_slow.sh
