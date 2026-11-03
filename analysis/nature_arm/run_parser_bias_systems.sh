#!/bin/bash
# Exploratory parser-bias check (6 Oct 2026): MultiCite then SciCite on the JATS and GROBID versions of the
# 200 parity papers. Sequential (one CPU-heavy job at a time). Outputs refuse to overwrite.
set -u
CITATION_ROOT="${CITATION_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
CITATION_VENVS="${CITATION_VENVS:-$HOME/.cache/citation_corpus/venvs}"
N=${NATURE_ARM_DIR:-${CITATION_ROOT}/external_data/nature_arm_2026-10-04}
L=$N/parser_bias/run_log.txt
MCV=${CITATION_VENVS}/aclarc_multicite/bin/python
SCV=${CITATION_VENVS}/scicite_local/.venv/bin/python
H=${CITATION_ROOT}/analysis/external_systems
for v in jats grobid; do
  echo "$(date '+%F %T') multicite $v start" >> $L
  ( cd $H/aclarc_multicite && HF_HOME=${CITATION_VENVS}/aclarc_multicite/hf_home \
    nice -n 5 $MCV run_multicite.py --model scibert --tag nature_parity_$v --edges $N/parser_bias/input_$v.csv ) >> $L 2>&1
  echo "$(date '+%F %T') multicite $v done rc=$?" >> $L
done
for v in jats grobid; do
  echo "$(date '+%F %T') scicite $v start" >> $L
  ( cd $H/scicite_local/harness && nice -n 5 $SCV 01_score_edges.py --edges $N/parser_bias/input_$v.csv --tag nature_parity_$v ) >> $L 2>&1
  echo "$(date '+%F %T') scicite $v done rc=$?" >> $L
done
echo "$(date '+%F %T') ALL DONE" >> $L
