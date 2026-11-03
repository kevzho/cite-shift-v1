#!/usr/bin/env bash
# PST-Bench (THUDM/paper-source-trace) as an external measured system -- reproducible harness.
# Everything stays in this directory. Nothing is sent to any hosted API.
set -euo pipefail
cd "$(dirname "$0")"
UV=$HOME/.local/bin/uv
# 0. code + data (public)
[ -d paper-source-trace ] || git clone https://github.com/THUDM/paper-source-trace.git && (cd paper-source-trace && git checkout d0896c62508ad4bb00d28b28b8411c92034a409d)
mkdir -p paper-source-trace/data logs outputs inputs/tei
[ -f paper-source-trace/data/PST.zip ] || curl -sL -o paper-source-trace/data/PST.zip "https://www.dropbox.com/scl/fi/namx1n55xzqil4zbkd5sv/PST.zip?rlkey=impcbm2acqmqhurv2oj0xxysx&dl=1"
[ -d paper-source-trace/data/PST ] || (cd paper-source-trace/data && unzip -q PST.zip)
# 1. env
[ -d .venv ] || { $UV venv --python 3.11 .venv; $UV pip install --python .venv/bin/python "torch==2.4.1" "transformers==4.44.2" "numpy<2" "scikit-learn==1.5.2" beautifulsoup4 lxml fuzzywuzzy python-Levenshtein tqdm pandas pyarrow duckdb; }
# 2. inputs: GROBID TEI of citing papers (read-only copy from the project's CHI/CHI EA GROBID output)
SRC=/Volumes/kz_extended/SALT-UIUC/citation_iConf/citation_corpus/external_data/chi_chiea_grobid_2026-09-15/pdfs_grobid
for id in $(.venv/bin/python -c "import csv;print(' '.join({r['citing_id'] for r in csv.DictReader(open('../test_edges.csv'))}))"); do
  for v in chi chiea; do [ -f "$SRC/$v/$id.tei.xml" ] && cp "$SRC/$v/$id.tei.xml" inputs/tei/ || true; done
done
# 3. sanity: official rule baseline on released labelled PST split (expect MAP ~0.065; paper test 0.0616)
.venv/bin/python harness/00_sanity_rule_on_pst.py > logs/00_sanity_rule_on_pst.log 2>&1
# 4. score test edges (rule baseline; SciBERT columns fill only if a trained checkpoint exists)
.venv/bin/python harness/03_score_edges.py --out outputs/pst_edge_scores.csv
.venv/bin/python harness/04_dump_rule_detail.py
# 5. OPTIONAL, NOT RUN TO COMPLETION (cost: >=~14 h CPU on this machine, see README):
#    .venv/bin/python harness/01_prepare_train.py   # official SciBERT training contexts (4 min)
#    .venv/bin/python harness/02_train_scibert.py   # official SciBERT fine-tune (20 ep, patience 5)
#    then re-run step 4 to fill scibert_* columns.
