#!/bin/bash
CITATION_ROOT="${CITATION_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
CITATION_VENVS="${CITATION_VENVS:-$HOME/.cache/citation_corpus/venvs}"
export CITATION_ROOT CITATION_VENVS
# EXPLORATORY (6 Oct 2026): MultiCite (SciBERT, official checkpoint) at scale on CHI (all edges) and the
# Nature arm (failed its registered parity gate -> exploratory only). Parser-bias bound from
# docs/results/2026-10-06_nature_parser_bias.md: uptake differences < ~2 points are not interpretable.
set -u
R=${CITATION_ROOT}
IN=$R/analysis/results/external_systems/aclarc_multicite/at_scale
L=$IN/run_log.txt
MCV=${CITATION_VENVS}/aclarc_multicite/bin/python
cd $R/analysis/external_systems/aclarc_multicite
for c in nature_arm chi; do
  echo "$(date '+%F %T') multicite $c start" >> $L
  HF_HOME=${CITATION_VENVS}/aclarc_multicite/hf_home \
    nice -n 5 $MCV run_multicite.py --model scibert --tag at_scale_$c --edges $IN/input_$c.csv >> $L 2>&1
  echo "$(date '+%F %T') multicite $c done rc=$?" >> $L
done
echo "$(date '+%F %T') ALL DONE" >> $L
