#!/bin/bash
CITATION_ROOT="${CITATION_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
PYTHON="${PYTHON:-python3}"
export CITATION_ROOT PYTHON
# Anti-collapse weight selection (spec change log, 8 Oct 2026): JEPA seed 1, full 6,000-step schedule,
# stopped at step 300, for w in {0.001, 0.01, 0.1}. Logs every 25 steps (step-50 and step-300 values).
R=${CITATION_ROOT}
PY=${PYTHON}
OUT=$R/analysis/results/ml/d2_jepa/vc_tests
mkdir -p $OUT; cd $R
for w in 0.001 0.01 0.1; do
  L=$OUT/jepa_w${w}.log
  echo "$(date '+%F %T') w=$w start" >> $OUT/run_log.txt
  $PY analysis/ml/d2_jepa/pretrain.py --objective jepa --seed 1 --batch 64 --micro-batch 16 --device mps \
    --vc-weight $w --log-every 25 --eval-every 100000 --ckpt-every 100000 --run-dir $OUT/jepa_w${w} > $L 2>&1 &
  P=$!
  until grep -q '"step": 300,' $L 2>/dev/null || ! kill -0 $P 2>/dev/null; do sleep 10; done
  kill $P 2>/dev/null; wait $P 2>/dev/null
  echo "$(date '+%F %T') w=$w stopped at step 300" >> $OUT/run_log.txt
done
echo "$(date '+%F %T') ALL DONE" >> $OUT/run_log.txt
