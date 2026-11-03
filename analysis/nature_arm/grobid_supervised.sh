#!/bin/bash
# Run GROBID over the Nature arm PDFs on a memory-constrained Mac, resuming until done.
# Same image and settings as CHI (grobid/grobid:0.9.0-crf, consolidateHeader=0, consolidateCitations=0).
# Keeps GROBID alive: a watcher stops the runner if GROBID stops answering; the loop restarts
# Docker/GROBID and resumes (finished TEIs are skipped). Run under `caffeinate -i`.
# Usage: caffeinate -i bash grobid_supervised.sh [concurrency=1] [memory=4g] [rounds=20]
CONC=${1:-1}; MEM=${2:-4g}; ROUNDS=${3:-20}
CITATION_ROOT="${CITATION_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
N="${NATURE_ARM_DIR:-$CITATION_ROOT/external_data/nature_arm_2026-10-04}"
cd "$CITATION_ROOT/analysis/extraction/corpus_ingest/upstream" || exit 1
LOG=$N/logs/grobid_supervised.log
export UV_LINK_MODE=copy

ensure_grobid () {
  docker info >/dev/null 2>&1 || { open -a Docker; for i in $(seq 1 60); do docker info >/dev/null 2>&1 && break; sleep 5; done; }
  curl -s --max-time 10 localhost:8070/api/isalive | grep -q true && return 0
  docker rm -f grobid >/dev/null 2>&1
  docker run -d -p 8070:8070 --memory "$MEM" --name grobid grobid/grobid:0.9.0-crf >/dev/null
  for i in $(seq 1 60); do curl -s --max-time 10 localhost:8070/api/isalive | grep -q true && return 0; sleep 5; done
  return 1
}

for round in $(seq 1 "$ROUNDS"); do
  echo "$(date '+%F %T') round $round (concurrency $CONC, memory $MEM)" >> "$LOG"
  ensure_grobid || { echo "$(date '+%T') grobid would not start" >> "$LOG"; sleep 60; continue; }
  ( while sleep 30; do
      curl -s --max-time 20 localhost:8070/api/isalive | grep -q true || {
        echo "$(date '+%T') grobid down; stopping runner" >> "$LOG"; pkill -f "parse_grobid_corpus.py run"; break; }
    done ) &
  WATCH=$!
  uv run python parse_grobid_corpus.py run --input-root $N/pdfs_by_journal --output-root $N/pdfs_grobid \
    --concurrency "$CONC" --retry-failed > "$N/logs/grobid_round_${round}.log" 2>&1
  kill $WATCH 2>/dev/null
  left=$(uv run python parse_grobid_corpus.py status --input-root $N/pdfs_by_journal --output-root $N/pdfs_grobid 2>/dev/null \
         | grep -o 'remaining: [0-9]*' | grep -o '[0-9]*')
  echo "$(date '+%F %T') remaining=$left" >> "$LOG"
  [ "$left" = "0" ] && { echo "$(date '+%F %T') DONE" >> "$LOG"; break; }
done
