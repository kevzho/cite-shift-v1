#!/bin/bash
CITATION_ROOT="${CITATION_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PYTHON="${PYTHON:-python3}"
export CITATION_ROOT PYTHON
# Registered D3 (soft vs hard labels on citation_function, pre-registration §7). Waits for the D2 pretraining
# queue to finish (memory), then runs D3 at normal priority (CPU) and writes the dated results note.
R=${CITATION_ROOT}
PY=${PYTHON}
Q=$R/analysis/results/ml/d2_jepa/runs_mps/run_log.txt
OUT=$R/analysis/results/ml/d3_2026-10
mkdir -p "$OUT"
echo "$(date '+%F %T') D3 chain waiting for GENTLE QUEUE ALL DONE" >> "$OUT/chain_log.txt"
until grep -q "GENTLE QUEUE ALL DONE" "$Q" 2>/dev/null; do sleep 300; done
while pgrep -f "d2_jepa/pretrain.py" >/dev/null; do sleep 60; done
cd "$R"
echo "$(date '+%F %T') D3 start" >> "$OUT/chain_log.txt"
$PY analysis/ml/11_d3_soft_labels.py > "$OUT/main_run.log" 2>&1
rc=$?
echo "$(date '+%F %T') D3 done rc=$rc" >> "$OUT/chain_log.txt"
[ $rc -eq 0 ] || exit $rc
$PY - <<'EOF'
import json, time
from pathlib import Path
R = Path(__import__("os").environ["CITATION_ROOT"])
r = json.loads((R / "analysis/results/ml/d3_2026-10/main/d3_results.json").read_text())
d = time.strftime("%Y-%m-%d")
A, sm = r["arms"], r["soft_minus_hard"]
rows = "\n".join(
    f"| {a} | {A[a]['ece_seed_mean']} ± {A[a]['ece_seed_sd']} | {A[a]['brier_seed_mean']} ± {A[a]['brier_seed_sd']} | "
    f"{A[a]['macro_f1_vs_kevin']} | {A[a]['macro_f1_vs_canonical']} |" for a in ("soft", "hard_kevin", "hard_canonical"))
diffs = "\n".join(
    f"| soft − {h} | {sm[h]['ece']['diff']} [{sm[h]['ece']['ci95'][0]}, {sm[h]['ece']['ci95'][1]}] | "
    f"{sm[h]['brier']['diff']} [{sm[h]['brier']['ci95'][0]}, {sm[h]['brier']['ci95'][1]}] |" for h in sm)
note = f"""# D3 — soft vs hard labels on citation function ({d})

**Registered analysis** (pre-registration §7 D3, OSF osf.io/nfz9c): canonical + Kevin v2 labels on
`citation_function` for the sample-B edges (Pass 1: stratum + rest; stratum Pass 1 is valid, deviation 3a excluded
only stratum Pass 2). Mechanical crosswalk (canonical background_bulk → background, etc.). Hard-label vs soft-label
heads (soft = agree → one-hot, disagree → 0.5/0.5), SciBERT, CPU deterministic, cross-validated by citation-graph
component, {r['folds']} folds, seeds {r['seeds']}, {r['epochs']} epochs. Primary: ECE (15 equal-width bins; accuracy =
soft-label mass on the predicted class) and Brier against the soft label distribution; secondary: macro-F1 (classes
with support ≥ 5) against each human record. `unclear` and Kevin `other` excluded (§10): {r['excluded']}.
Script `analysis/ml/11_d3_soft_labels.py`; output `analysis/results/ml/d3_2026-10/main/`.

n = {r['n_edges']} edges; Kevin–canonical agreement {r['agreement_rate_kevin_canonical']}.

| Arm | ECE (mean ± sd over seeds) | Brier (mean ± sd) | macro-F1 vs Kevin | macro-F1 vs canonical |
|---|---|---|---|---|
{rows}

Soft − hard differences (seed-averaged probabilities; cluster bootstrap over components, 10,000 resamples, seed
20261003; negative = soft better calibrated):

| Comparison | ΔECE [95% CI] | ΔBrier [95% CI] |
|---|---|---|
{diffs}

**Verdict vs the registered hard arm (Kevin v2): {r['verdict_vs_registered_hard_arm']}.** Decision rule fixed in the
script before the run: soft improves calibration only if the 95% CIs of both ΔECE and ΔBrier lie below 0. The
hard_canonical arm is a sensitivity arm, not registered as primary. Runtime {r['wall_s']} s.
"""
(R / f"docs/results/{d}_d3_soft_labels.md").write_text(note, encoding="utf-8")
print("wrote", f"docs/results/{d}_d3_soft_labels.md")
EOF
echo "$(date '+%F %T') D3 results note written" >> "$OUT/chain_log.txt"
