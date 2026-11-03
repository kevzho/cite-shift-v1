#!/bin/bash
CITATION_ROOT="${CITATION_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
export CITATION_ROOT
# Give the screen GPU time while D2 pretrains: pause the training process 1 s out of every 3 s
# (SIGSTOP/SIGCONT; training continues, ~1/3 slower). Exits when the stop file appears.
STOP=${CITATION_ROOT}/analysis/results/ml/d2_jepa/runs_mps/STOP_THROTTLE
trap 'for p in $(pgrep -f "d2_jepa/pretrain.py"); do kill -CONT $p 2>/dev/null; done' EXIT
while [ ! -f "$STOP" ]; do
  P=$(pgrep -f "d2_jepa/pretrain.py")
  if [ -n "$P" ]; then kill -STOP $P 2>/dev/null; sleep 1; kill -CONT $P 2>/dev/null; sleep 2; else sleep 10; fi
done
