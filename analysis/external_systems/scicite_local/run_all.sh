#!/usr/bin/env bash
CITATION_ROOT="${CITATION_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
CITATION_VENVS="${CITATION_VENVS:-$HOME/.cache/citation_corpus/venvs}"
UV_BIN="${UV:-uv}"
export CITATION_ROOT CITATION_VENVS UV_BIN
# Official SciCite model (allenai/scicite; Cohan et al., NAACL 2019) as an external measured system.
# Runs 100% locally on CPU. Nothing is sent to any hosted API or web service.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
V=${CITATION_VENVS}/scicite_local
R=${CITATION_ROOT}/analysis/results/external_systems/scicite_local
UV=${UV_BIN}
PY=$V/.venv/bin/python
mkdir -p $V/artefacts $R/outputs $R/logs
cd $V/artefacts
# 0. public code, model archive, dataset (sha256 checked)
[ -d scicite ] || { git clone https://github.com/allenai/scicite.git; (cd scicite && git checkout b33a8cd0df5743c9d483b1d0fdce8ec62bc8f135); }
[ -f scicite.tar.gz ] || curl -sS -o scicite.tar.gz https://s3-us-west-2.amazonaws.com/ai2-s2-research/scicite/models/scicite.tar.gz
echo "4eeef0f04f29e0431f549db2e2f840816a09b487ef33ffca84dc69c1d4688e07  scicite.tar.gz" | shasum -a 256 -c
[ -d scicite_model ] || { mkdir scicite_model && tar xzf scicite.tar.gz -C scicite_model; }
[ -f scicite_data.tar.gz ] || curl -sS -o scicite_data.tar.gz https://s3-us-west-2.amazonaws.com/ai2-s2-research/scicite/scicite.tar.gz
echo "711ece2c4e61d116c8ae5bb07e9fbb2ee9ff7bba004b4cab7fbd0ac3af499193  scicite_data.tar.gz" | shasum -a 256 -c
[ -d scicite_data ] || { mkdir -p scicite_data && tar xzf scicite_data.tar.gz -C scicite_data --strip-components 1; }
# 1. env: allennlp 0.8.0 with --no-deps on a modern stack (its pinned deps need Python 3.6 / torch 1.0 / spaCy 2.0)
if [ ! -d $V/.venv ]; then
  $UV venv --python 3.11 $V/.venv
  $UV pip install --python $PY --no-deps allennlp==0.8.0 pytorch-pretrained-bert==0.3.0
  $UV pip install --python $PY torch==2.5.1 numpy==1.26.4 h5py==3.16.0 overrides==3.1.0 spacy==3.7.5 tensorboardx==2.6.5 \
      nltk tqdm requests boto3 editdistance scikit-learn scipy unidecode ftfy parsimonious sqlparse conllu jsonlines python-levenshtein
  $UV pip install --python $PY https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.1/en_core_web_sm-3.7.1-py3-none-any.whl
fi
# (exact versions of every package: env_freeze.txt; `uv pip sync --python $PY env_freeze.txt` reproduces them)
cd "$HERE"
# 2. sanity: reproduce the published SciCite test score (paper macro-F1 84.0) on the public test split (~27 min here)
$PY harness/00_sanity_scicite_test.py > $R/logs/00_sanity_scicite_test.log 2>&1
# 3. score the 20 test edges twice (separate processes) and check determinism
$PY harness/01_score_edges.py --tag run1 > $R/logs/01_score_edges_run1.log 2>&1
$PY harness/01_score_edges.py --tag run2 > $R/logs/01_score_edges_run2.log 2>&1
$PY harness/02_compare_runs.py | tee $R/logs/02_compare_runs.log
# 4. throughput on 200 seeded-random CHI/CHI EA edges (predictions discarded)
$PY harness/03_throughput.py 200 > $R/logs/03_throughput.log 2>&1
