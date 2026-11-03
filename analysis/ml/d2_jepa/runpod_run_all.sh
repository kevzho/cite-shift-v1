#!/bin/bash
# D2 (exploratory) on a RunPod RTX 4090 pod: JEPA and MLM, seeds 1-3, run sequentially.
# Spec: private/preregistration/JEPA_EXPLORATORY_SPEC_2026-10-07.md. Same token budget per run
# (6,000 steps x batch 64 x 256 tokens). Micro-batch 32 = gradient accumulation (same effective
# batch). Launched detached (setsid) so it survives SSH drops; progress in /workspace/d2/run_all.log.
set -u
cd /workspace/d2
export HF_HOME=/workspace/hf-cache
LOG=/workspace/d2/run_all.log
{
  echo "$(date -u '+%F %T') start"; nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
  python3 -c "import torch,transformers;print('torch',torch.__version__,'cuda',torch.version.cuda,'transformers',transformers.__version__)"
} >> $LOG 2>&1
for seed in 1 2 3; do
  for obj in jepa mlm; do
    echo "$(date -u '+%F %T') $obj seed $seed start" >> $LOG
    python3 pretrain.py --objective $obj --seed $seed --batch 64 --micro-batch 32 --device cuda \
      --data /workspace/d2/data --run-dir /workspace/d2/runs/${obj}_seed${seed} \
      > /workspace/d2/runs_${obj}_seed${seed}.log 2>&1
    echo "$(date -u '+%F %T') $obj seed $seed done rc=$?" >> $LOG
  done
done
echo "$(date -u '+%F %T') ALL DONE" >> $LOG
