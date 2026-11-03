#!/usr/bin/env bash
# End-to-end reproduction of the open_llm_intent feasibility run (Koloveas et al. TPDL 2025, released fine-tuned
# Qwen2.5-14B-CIC-SciCite, GGUF Q4_K_M, llama.cpp/Metal, fully local). Usage: bash run_all.sh [scicite|acl-arc]
set -euo pipefail
DS="${1:-scicite}"
UV=$HOME/.local/bin/uv
CACHE=$HOME/.cache/citation_corpus/venvs/open_llm_intent
HERE="$(cd "$(dirname "$0")" && pwd)"
RES=/Volumes/kz_extended/SALT-UIUC/citation_iConf/citation_corpus/analysis/results/external_systems/open_llm_intent
mkdir -p "$CACHE/artefacts" "$CACHE/models" "$RES/outputs" "$RES/logs" "$RES/inputs"

# 1. environment (llama-cpp-python built from source with Metal; the prebuilt metal wheel index served a corrupt wheel)
if [ ! -x "$CACHE/.venv/bin/python" ]; then
  "$UV" venv --python 3.11 "$CACHE/.venv"
  "$UV" pip install --python "$CACHE/.venv/bin/python" cmake==4.4.3 ninja==1.13.2
  PATH="$CACHE/.venv/bin:$PATH" CMAKE_ARGS="-DGGML_METAL=on -DGGML_NATIVE=on" \
    "$UV" pip install --python "$CACHE/.venv/bin/python" --no-binary llama-cpp-python \
    llama-cpp-python==0.3.16 jinja2==3.1.6 pandas==3.0.6 numpy==2.4.6 scikit-learn==1.9.1
fi
PY="$CACHE/.venv/bin/python"

# 2. official code (prompts, label cleanup, public test split + the authors' published predictions)
if [ ! -d "$CACHE/artefacts/CitationIntentOpenLLM" ]; then
  git clone https://github.com/athenarc/CitationIntentOpenLLM "$CACHE/artefacts/CitationIntentOpenLLM"
fi
git -C "$CACHE/artefacts/CitationIntentOpenLLM" checkout -q 3034b803e19eae8da5b04c8af47eb438d8c6f40f

# 3. weights: mradermacher static Q4_K_M quants of the sknow-lab merged checkpoints, pinned by repo revision + sha256
if [ "$DS" = scicite ]; then
  F=Qwen2.5-14B-CIC-SciCite.Q4_K_M.gguf; REPO=mradermacher/Qwen2.5-14B-CIC-SciCite-GGUF
  REV=231348503f4081969af6202288af5a9ccecec8b0; SHA=2d5112eda113ee4acae88fe73064a62047d3919004f3cd0bf5f9cb806cd3eab8
else
  F=Qwen2.5-14B-CIC-ACLARC.Q4_K_M.gguf; REPO=mradermacher/Qwen2.5-14B-CIC-ACLARC-GGUF
  REV=20a630ae7d3b1d58ff59a8ee3d839db6ae2abbdd; SHA=9457ca206c3069ddfc75a05ef141d0a9b4974baeed3b0faabf0bfb7c609577fa
fi
[ -f "$CACHE/models/$F" ] || curl -L -o "$CACHE/models/$F" "https://huggingface.co/$REPO/resolve/$REV/$F"
echo "$SHA  $CACHE/models/$F" | shasum -a 256 -c -

# 4. run (stderr = llama.cpp/Metal init chatter)
T=$([ "$DS" = scicite ] && echo sc || echo aa)
cd "$HERE/harness"
"$PY" 01_sanity_public_test.py --dataset "$DS" --n 300 2> "$RES/logs/sanity_$DS.stderr.log" | tee "$RES/logs/sanity_$DS.log"
for r in 1 2; do
  "$PY" 02_score_edges.py --dataset "$DS" --run $r 2> "$RES/logs/score_${DS}_run$r.stderr.log" | tee "$RES/logs/score_${DS}_run$r.log"
done
"$PY" 03_compare_runs.py --dataset "$DS"
"$PY" 04_throughput.py --dataset "$DS" --n 200 2> "$RES/logs/throughput_$DS.stderr.log" | tee "$RES/logs/throughput_$DS.log"
