#!/usr/bin/env bash
CITATION_ROOT="${CITATION_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
CITATION_VENVS="${CITATION_VENVS:-$HOME/.cache/citation_corpus/venvs}"
UV_BIN="${UV:-uv}"
export CITATION_ROOT CITATION_VENVS UV_BIN
# End-to-end reproduction of the aclarc_multicite feasibility run (2026-09-26). Run from anywhere.
set -euo pipefail
H=${CITATION_ROOT}/analysis/external_systems/aclarc_multicite
R=${CITATION_ROOT}/analysis/results/external_systems/aclarc_multicite
V=${CITATION_VENVS}/aclarc_multicite
UV=${UV_BIN}
export HF_HOME=$V/hf_home
mkdir -p $R/{inputs,outputs,logs}

# 1. environment (Python 3.11.15)
[ -x $V/bin/python ] || $UV venv --python 3.11 $V
VIRTUAL_ENV=$V $UV pip install torch==2.5.1 transformers==4.46.3 numpy==1.26.4 pandas==2.2.3 \
  huggingface_hub==0.26.5 sentencepiece spacy==3.7.5 "click<8.2" pypdf==5.1.0

# 2. artefacts (public downloads only)
HF_HUB_OFFLINE=0 $V/bin/python -c "
from huggingface_hub import snapshot_download
snapshot_download('allenai/multicite-multilabel-scibert', revision='7696929cf8ba26c21cc1da0daeb76010575dff80', allow_patterns=['*.json','*.txt','*.bin'])"
mkdir -p $V/models/scicite_aclarc $V/data
[ -f $V/models/scicite_aclarc/weights.th ] || (cd $V/models/scicite_aclarc && \
  curl -sO https://s3-us-west-2.amazonaws.com/ai2-s2-research/scicite/models/aclarc.tar.gz && \
  echo "d144c771ca248e7472b7d982705abee8f0a9963b49f8edf13f216493a76c7b9f  aclarc.tar.gz" | shasum -a 256 -c && tar xzf aclarc.tar.gz)
[ -f $V/data/acl-arc/test.jsonl ] || (cd $V/data && \
  curl -sO https://s3-us-west-2.amazonaws.com/ai2-s2-research/scicite/acl-arc.tar.gz && tar xzf acl-arc.tar.gz)
MC=$V/src/multicite
[ -d $MC ] || (git clone -q https://github.com/allenai/multicite $MC && git -C $MC checkout -q 120671924b5ab968b98c7f696c4b67b4f8118948)

# 3. install checks on each system's OWN public test split (no project data)
$V/bin/python $H/sanity_multicite_testset.py $MC gold | tee $R/logs/sanity_multicite_scibert_gold.log
$V/bin/python $H/sanity_multicite_testset.py $MC 1    | tee $R/logs/sanity_multicite_scibert_1.log
$V/bin/python $H/sanity_aclarc_testset.py $V/models/scicite_aclarc $V/data/acl-arc/test.jsonl | tee $R/logs/sanity_aclarc_testset.log

# 4. the 20 test edges, twice each (determinism)
for t in run1 run2; do
  $V/bin/python $H/run_multicite.py --model scibert --tag $t | tee $R/logs/multicite_scibert_$t.log
  $V/bin/python $H/run_aclarc.py --tag $t | tee $R/logs/aclarc_scaffold_$t.log
done
python3 $H/make_table.py | tee $R/logs/make_table.log

# 5. throughput on 200 random CHI/CHI EA edges (seed 12345); outputs discarded
$V/bin/python $H/run_multicite.py --model scibert --throughput 200 | tee $R/logs/throughput_multicite_scibert.log
$V/bin/python $H/run_aclarc.py --throughput 200 | tee $R/logs/throughput_aclarc_scaffold.log
VIRTUAL_ENV=$V $UV pip freeze > $R/env_freeze.txt
