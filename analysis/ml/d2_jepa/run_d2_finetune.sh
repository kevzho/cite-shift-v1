#!/bin/bash
CITATION_ROOT="${CITATION_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
PYTHON="${PYTHON:-python3}"
export CITATION_ROOT PYTHON
# D2 fine-tuning (JEPA exploratory spec): D1's multi-head recipe (10_train_distilled.py, defaults, CPU
# deterministic) started from each pretrained backbone; fine-tune seed = pretraining seed. The "None" arm
# (SciBERT as released) is D1 itself (analysis/results/ml/d1_2026-10-07, same script and defaults, seeds 1-3).
# nice 10 so the Mac stays usable. Resume-safe: a seed whose predictions exist is skipped.
R=${CITATION_ROOT}
PY=${PYTHON}
P=$R/analysis/results/ml/d2_jepa/runs_mps
OUT=$R/analysis/results/ml/d2_jepa/finetune_2026-10-11
mkdir -p "$OUT"
cd "$R"
for arm in jepa mlm; do
  for s in 1 2 3; do
    if [ $arm = jepa ]; then init=$P/jepa_w0.001_seed$s/final; else init=$P/mlm_seed$s/final; fi
    d=$OUT/$arm
    if [ -f "$d/d1_frozen_predictions_seed$s.csv" ]; then echo "$(date '+%F %T') $arm seed $s exists, skip" >> "$OUT/run_log.txt"; continue; fi
    echo "$(date '+%F %T') $arm seed $s start (init $init)" >> "$OUT/run_log.txt"
    nice -n 10 $PY analysis/ml/10_train_distilled.py --seed $s --init "$init" --out "$d" > "$OUT/log_${arm}_seed$s.txt" 2>&1
    echo "$(date '+%F %T') $arm seed $s done rc=$?" >> "$OUT/run_log.txt"
  done
done
echo "$(date '+%F %T') D2 FINETUNE ALL DONE" >> "$OUT/run_log.txt"
# Score all three arms against Kevin v2 once all six fine-tunes exist (31_d2_jepa_scoring.py writes the dated note).
cd "$R" && $PY analysis/experiments/31_d2_jepa_scoring.py > "$OUT/scoring.log" 2>&1
echo "$(date '+%F %T') D2 SCORING done rc=$?" >> "$OUT/run_log.txt"
