#!/bin/bash
# Create the per-system virtualenvs that the expensive and provider tiers call, from the frozen package lists in
# reproduce/envs/. Requires uv (https://docs.astral.sh/uv/). Location: $CITATION_VENVS (default ~/.cache/citation_corpus/venvs).
#   bash reproduce/setup_envs.sh                 # all four
#   bash reproduce/setup_envs.sh scicite_local   # one
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENVS="${CITATION_VENVS:-$HOME/.cache/citation_corpus/venvs}"
UV_BIN="${UV:-uv}"
names="${*:-aclarc_multicite scicite_local s2_influential local_llm_codebook_v2}"
for n in $names; do
  case "$n" in
    aclarc_multicite)      py=3.11.15; dest="$VENVS/aclarc_multicite" ;;
    scicite_local)         py=3.11.15; dest="$VENVS/scicite_local/.venv" ;;
    s2_influential)        py=3.11.15; dest="$VENVS/s2_influential" ;;
    local_llm_codebook_v2) py=3.12.13; dest="$VENVS/local_llm_codebook_v2" ;;  # standard library only; talks to Ollama over HTTP
    *) echo "unknown environment: $n" >&2; exit 2 ;;
  esac
  echo "== $n -> $dest (Python $py)"
  "$UV_BIN" venv --python "$py" "$dest"
  if [ -s "$HERE/envs/$n.txt" ]; then "$UV_BIN" pip install --python "$dest/bin/python" -r "$HERE/envs/$n.txt"; fi
done
echo "MultiCite downloads its checkpoint into HF_HOME=$VENVS/aclarc_multicite/hf_home on first run; ACL-ARC expects its"
echo "archive in $VENVS/aclarc_multicite/models/scicite_aclarc (see reproduce/README.md)."
