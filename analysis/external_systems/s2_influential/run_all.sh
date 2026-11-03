#!/usr/bin/env bash
CITATION_ROOT="${CITATION_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
CITATION_VENVS="${CITATION_VENVS:-$HOME/.cache/citation_corpus/venvs}"
UV_BIN="${UV:-uv}"
export CITATION_ROOT CITATION_VENVS UV_BIN
# Semantic Scholar isInfluential / intents / contexts as an external measured system -- reproducible harness.
# Sends ONLY public CorpusIds to api.semanticscholar.org (S2's own precomputed labels). No citance or text is sent.
# The API key is read from citation_corpus/.env inside s2_client.py; it is never printed or written.
set -euo pipefail
cd "$(dirname "$0")/harness"
UV=${UV_BIN}
V=${CITATION_VENVS}/s2_influential
[ -d $V ] || { $UV venv --python 3.11 $V; VIRTUAL_ENV=$V $UV pip install "requests==2.34.2" "pypdf==6.19.0"; }
P=$V/bin/python
L=${CITATION_ROOT}/analysis/results/external_systems/s2_influential/logs
mkdir -p $L
TAG=${1:-$(date -u +%Y%m%dT%H%M%SZ)}
# 1. freeze a snapshot (two independent fetches for the determinism check)
$P 01_fetch_snapshot.py ${TAG}_A > $L/01_fetch_${TAG}_A.log 2>&1
$P 01_fetch_snapshot.py ${TAG}_B > $L/01_fetch_${TAG}_B.log 2>&1
# 2. per-edge outputs from each frozen snapshot (offline, pure function of the snapshot)
$P 02_extract.py ${TAG}_A > $L/02_extract_${TAG}_A.log
$P 02_extract.py ${TAG}_B > $L/02_extract_${TAG}_B.log
# 3. determinism: fetch A vs fetch B, and re-extraction of A
$P 03_compare_runs.py ${TAG}_A ${TAG}_B > $L/03_compare_${TAG}.log
# 4. OPTIONAL throughput probe on 200 seeded CHI lines (responses discarded; ~10-20 min under a shared key)
#    $P 04_throughput_chi.py 200 42 > $L/04_throughput_chi.log 2>&1
