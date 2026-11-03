#!/bin/bash
CITATION_ROOT="${CITATION_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
PYTHON="${PYTHON:-python3}"
export CITATION_ROOT PYTHON
# D2 (exploratory) on the Mac GPU (MPS): JEPA then MLM for one seed. Effective batch 64 via
# micro-batch 16; 6,000 steps each (same token budget). Usage: mac_run_seed.sh <seed>
S=${1:-1}
R=${CITATION_ROOT}
PY=${PYTHON}
OUT=$R/analysis/results/ml/d2_jepa/runs_mps
mkdir -p $OUT
cd $R
for obj in jepa mlm; do
  echo "$(date '+%F %T') $obj seed $S start" >> $OUT/run_log.txt
  $PY analysis/ml/d2_jepa/pretrain.py --objective $obj --seed $S --batch 64 --micro-batch 16 --device mps \
    --run-dir $OUT/${obj}_seed$S > $OUT/${obj}_seed$S.log 2>&1
  echo "$(date '+%F %T') $obj seed $S done rc=$?" >> $OUT/run_log.txt
done
echo "$(date '+%F %T') seed $S ALL DONE" >> $OUT/run_log.txt
