#!/bin/bash
CITATION_ROOT="${CITATION_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
export CITATION_ROOT
# Wait for the overnight MultiCite-at-scale run to finish (memory), then train D1 for seeds 1-3 on CPU.
R=${CITATION_ROOT}
ML=$R/analysis/results/external_systems/aclarc_multicite/at_scale/run_log.txt
OUT=$R/analysis/results/ml/d1_2026-10-07
mkdir -p $OUT
until grep -q "ALL DONE" $ML 2>/dev/null; do sleep 120; done
cd $R
for s in 1 2 3; do
  echo "$(date '+%F %T') D1 seed $s start" >> $OUT/run_log.txt
  python3 analysis/ml/10_train_distilled.py --seed $s --out $OUT > $OUT/log_seed$s.txt 2>&1
  echo "$(date '+%F %T') D1 seed $s done rc=$?" >> $OUT/run_log.txt
done
echo "$(date '+%F %T') ALL DONE" >> $OUT/run_log.txt
