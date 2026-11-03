#!/bin/bash
CITATION_ROOT="${CITATION_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
export CITATION_ROOT
# Keep every D2 pretraining process at background priority (Kevin wants Chrome/VS Code usable) without
# pausing it: apply taskpolicy -b + nice 10 to each new pretrain process; exit when the fast queue finishes.
L=${CITATION_ROOT}/analysis/results/ml/d2_jepa/runs_mps/run_log.txt
done_pids=" "
until grep -q "FAST QUEUE ALL DONE" "$L"; do
  for p in $(pgrep -f "d2_jepa/pretrain.py"); do
    case "$done_pids" in *" $p "*) ;; *)
      taskpolicy -b -p "$p" && renice -n 10 -p "$p" >/dev/null && done_pids="$done_pids$p "
      echo "$(date '+%F %T') throttled pid $p (background QoS, nice 10)" >> "$L";;
    esac
  done
  sleep 60
done
