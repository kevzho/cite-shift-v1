#!/bin/bash
CITATION_ROOT="${CITATION_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
PYTHON="${PYTHON:-python3}"
export CITATION_ROOT PYTHON
# MPS speed test for D2: 30 steps per objective at the planned batch (64) and length (256).
cd ${CITATION_ROOT}
OUT=analysis/results/ml/d2_jepa/speedtest_mps
mkdir -p $OUT
for obj in jepa mlm; do
  s=$(date +%s)
  ${PYTHON} analysis/ml/d2_jepa/pretrain.py --objective $obj --seed 1 --steps 30 --batch 64 --micro-batch 16 --device mps \
    --limit 4000 --log-every 5 --eval-every 100000 --ckpt-every 100000 --run-dir $OUT/$obj > $OUT/$obj.log 2>&1
  rc=$?; e=$(date +%s)
  echo "$obj rc=$rc wall_s=$((e-s))" >> $OUT/summary.txt
done
echo DONE >> $OUT/summary.txt
