#!/bin/bash
CITATION_ROOT="${CITATION_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
PYTHON="${PYTHON:-python3}"
export CITATION_ROOT PYTHON
# D2 (exploratory) remaining runs on the Mac GPU at LOW priority so the Mac stays usable:
# JEPA (w = 0.001, selected by the pre-stated rule) seeds 1-3 and MLM seeds 2-3, interleaved.
# Background QoS (taskpolicy -b) + nice 19; micro-batch 8 (effective batch still 64); MPS memory capped.
R=${CITATION_ROOT}
PY=${PYTHON}
OUT=$R/analysis/results/ml/d2_jepa/runs_mps
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.7   # hard cap on MPS share of RAM
export PYTORCH_MPS_LOW_WATERMARK_RATIO=0.6    # must be <= the high ratio
mkdir -p $OUT; cd $R
run () {  # objective seed extra-args...
  local obj=$1 s=$2; shift 2
  local tag=${obj}_seed$s; [ $obj = jepa ] && tag=jepa_w0.001_seed$s
  [ -d $OUT/$tag/final ] && { echo "$(date '+%F %T') $tag already done, skipping" >> $OUT/run_log.txt; return; }
  local free_gb=$(df -g "$OUT" | awk 'NR==2{print $4}')
  if [ "${free_gb:-0}" -lt 5 ]; then echo "$(date '+%F %T') $tag NOT started: only ${free_gb} GB free on the drive" >> $OUT/run_log.txt; return; fi
  echo "$(date '+%F %T') $tag start (low priority, micro-batch 8; ${free_gb} GB free)" >> $OUT/run_log.txt
  taskpolicy -b nice -n 19 $PY analysis/ml/d2_jepa/pretrain.py --objective $obj --seed $s --batch 64 --micro-batch 8 \
    --device mps --run-dir $OUT/$tag "$@" >> $OUT/$tag.log 2>&1
  local rc=$?
  echo "$(date '+%F %T') $tag done rc=$rc" >> $OUT/run_log.txt
  # Free the ~1.7 GB resume checkpoint once the final backbone is safely saved.
  if [ $rc -eq 0 ] && [ -f "$OUT/$tag/final/model.safetensors" ]; then rm -f "${OUT:?}/${tag:?}/ckpt/last.pt" "${OUT:?}/${tag:?}/ckpt/last.tmp"; fi
}
run jepa 1 --vc-weight 0.001
run mlm 2
run jepa 2 --vc-weight 0.001
run mlm 3
run jepa 3 --vc-weight 0.001
echo "$(date '+%F %T') SLOW QUEUE ALL DONE" >> $OUT/run_log.txt
