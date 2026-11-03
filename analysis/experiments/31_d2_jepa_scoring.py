#!/usr/bin/env python3
"""D2 JEPA, exploratory (private/preregistration/JEPA_EXPLORATORY_SPEC_2026-10-07.md): score the three
pretraining arms, each fine-tuned with D1's recipe (10_train_distilled.py, defaults, seeds 1-3), against
Kevin's locked v2 labels.

Arms: jepa  analysis/results/ml/d2_jepa/finetune_2026-10-11/jepa   (init runs_mps/jepa_w0.001_seed{s}/final)
      mlm   analysis/results/ml/d2_jepa/finetune_2026-10-11/mlm    (init runs_mps/mlm_seed{s}/final)
      none  analysis/results/ml/d1_2026-10-07                      (SciBERT as released = D1 itself)
Primary (spec): macro-F1 on credits_contribution and uptake_mode vs Kevin v2 B-rest (240 Pass 2 edges), mean
over 3 seeds with per-seed values; JEPA - MLM difference with a cluster bootstrap over components.
Prediction: JEPA > MLM on uptake_mode; counts against it if the JEPA - MLM interval includes 0 or favours MLM.
Secondary: the other four v2 fields on B (Pass 1 on 300 edges), all Pass 2 fields on A250.
Statistics, references, unclear handling and the bootstrap (10,000, seed 20261003; the same resamples for every
arm, so differences are paired) are those of 29_confirmatory_2026_10_08.py, imported unchanged.
Outputs (refuse to overwrite): analysis/results/ml/d2_jepa/scoring_2026-10-11/d2_scores.json and a dated note
in docs/results/.
"""
from __future__ import annotations

import importlib.util
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("conf", ROOT / "analysis/experiments/29_confirmatory_2026_10_08.py")
C = importlib.util.module_from_spec(spec)
spec.loader.exec_module(C)
S = C.S

FT = ROOT / "analysis/results/ml/d2_jepa/finetune_2026-10-11"
ARMS = {"jepa": FT / "jepa", "mlm": FT / "mlm", "none": ROOT / "analysis/results/ml/d1_2026-10-07"}
OUT = ROOT / "analysis/results/ml/d2_jepa/scoring_2026-10-11"
FIELDS = C.PASS1 + C.PASS2
PRIMARY = ("credits_contribution", "uptake_mode")


def load(cw, alias):
    preds = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))   # arm -> seed -> field -> {edge: label}
    for arm, d in ARMS.items():
        for s in (1, 2, 3):
            for r in C.read(d / f"d1_frozen_predictions_seed{s}.csv"):
                for f in FIELDS:
                    v = (r.get(f) or "").strip()
                    if v and v != "not_applicable":
                        preds[arm][s][f][alias.get(r["edge_id"], r["edge_id"])] = S.norm(f, v, cw)
    return preds


def score(ref, edges, boot, seeds, field):
    per = []
    for s, fields in sorted(seeds.items()):
        pred = fields.get(field, {})
        pairs = [(e, ref[e], pred[e]) for e in edges if e in ref and e in pred
                 and not (field == "uptake_degree" and pred[e] not in ("full", "partial"))]
        per.append(C.macro_f1(pairs, boot))
    return per


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    out_json = OUT / "d2_scores.json"
    if out_json.exists():
        raise SystemExit(f"refusing to overwrite {out_json}")
    cw = S.crosswalk()
    alias, cluster = S.components()
    canon, v2b_raw, v2a_raw, _ = C.references(cw)
    v2b, _ = C.drop_unclear(v2b_raw)
    v2a, _ = C.drop_unclear(v2a_raw)
    B1, B2, A = sorted(v2b_raw["citation_function"]), sorted(v2b_raw["uptake_mode"]), sorted(v2a_raw["uptake_mode"])
    bootB, bootA = C.Boot(B1, cluster), C.Boot(A, cluster)
    preds = load(cw, alias)

    res, boots = {}, {}
    for refname, ref, boot, fields in (("kevin_v2@B", v2b, bootB, FIELDS), ("kevin_v2@A250", v2a, bootA, C.PASS2)):
        for field in fields:
            edges = (B1 if field in C.PASS1 else B2) if refname == "kevin_v2@B" else A
            for arm in ARMS:
                per = [m for m in score(ref[field], edges, boot, preds[arm], field) if m]
                if not per:
                    continue
                bf1 = np.nanmean(np.stack([m["boot_f1"] for m in per]), axis=0)
                boots[(refname, field, arm)] = bf1
                res[f"{refname}|{field}|{arm}"] = {
                    "n": per[0]["n"], "macro_f1_seed_mean": C.r4(np.mean([m["macro_f1"] for m in per])),
                    "macro_f1_per_seed": [C.r4(m["macro_f1"]) for m in per], "ci95": C.ci(bf1),
                    "kappa_seed_mean": C.r4(np.mean([m["kappa"] for m in per])), "withheld": per[0]["withheld"]}
    diffs = {}
    for (refname, field, arm), b in boots.items():
        if arm != "jepa":
            continue
        for other in ("mlm", "none"):
            if (refname, field, other) in boots:
                full = res[f"{refname}|{field}|jepa"]["macro_f1_seed_mean"] - res[f"{refname}|{field}|{other}"]["macro_f1_seed_mean"]
                diffs[f"{refname}|{field}|jepa-{other}"] = {"diff": C.r4(full), "ci95": C.ci(b - boots[(refname, field, other)])}
        if (refname, field, "mlm") in boots and (refname, field, "none") in boots:
            full = res[f"{refname}|{field}|mlm"]["macro_f1_seed_mean"] - res[f"{refname}|{field}|none"]["macro_f1_seed_mean"]
            diffs[f"{refname}|{field}|mlm-none"] = {"diff": C.r4(full),
                                                   "ci95": C.ci(boots[(refname, field, "mlm")] - boots[(refname, field, "none")])}
    d = diffs["kevin_v2@B|uptake_mode|jepa-mlm"]
    verdict = ("SUPPORTED (JEPA > MLM on uptake_mode, interval excludes 0)" if d["ci95"][0] > 0 else
               "NOT SUPPORTED (interval favours MLM)" if d["ci95"][1] < 0 else "NOT SUPPORTED (interval includes 0)")
    out = {"spec": "private/preregistration/JEPA_EXPLORATORY_SPEC_2026-10-07.md", "n_boot": C.N_BOOT, "seed": C.SEED,
           "edges": {"B_pass1": len(B1), "B_pass2": len(B2), "A250": len(A)}, "scores": res, "differences": diffs,
           "verdict_uptake_mode": verdict}
    out_json.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")

    def row(refname, field):
        cells = []
        for arm in ARMS:
            r = res.get(f"{refname}|{field}|{arm}")
            cells.append(f"{r['macro_f1_seed_mean']} ({', '.join(map(str, r['macro_f1_per_seed']))})" if r else "—")
        dd = [diffs.get(f"{refname}|{field}|{k}") for k in ("jepa-mlm", "jepa-none", "mlm-none")]
        cells += [f"{x['diff']:+} [{x['ci95'][0]:+}, {x['ci95'][1]:+}]" if x else "—" for x in dd]
        n = res.get(f"{refname}|{field}|jepa", {}).get("n", "")
        return f"| {field} | {n} | " + " | ".join(cells) + " |"

    head = ("| Field | n | JEPA mean (s1, s2, s3) | MLM mean (s1, s2, s3) | None (SciBERT) mean (s1, s2, s3) | "
            "JEPA − MLM [95% CI] | JEPA − None | MLM − None |\n|---|---|---|---|---|---|---|---|")
    day = time.strftime("%Y-%m-%d")
    note = f"""# D2 JEPA vs masked-LM vs SciBERT, scored against Kevin v2 ({day})

**Exploratory** (not registered on OSF), run exactly as `private/preregistration/JEPA_EXPLORATORY_SPEC_2026-10-07.md`
fixed before any code ran. Script `analysis/experiments/31_d2_jepa_scoring.py`; output `{out_json.relative_to(ROOT)}`.
Arms differ only in the pretraining objective; each is fine-tuned with D1's recipe (`10_train_distilled.py`,
defaults, CPU deterministic, seeds 1–3; fine-tune seed = pretraining seed). None = SciBERT as released = D1 itself.
JEPA = variance-hinge regulariser w = 0.001 (selected by the pre-stated rule); seed 1's predictor collapsed after
step ~2,000 (encoder spread kept) — reported, not re-tuned. Macro-F1 over classes with reference support ≥ 5;
cluster bootstrap over components, {C.N_BOOT:,} resamples, seed {C.SEED}, identical resamples for every arm (paired).

## Primary — Kevin v2, sample B (Pass 2 on {len(B2)} B-rest edges)

{head}
{row("kevin_v2@B", "credits_contribution")}
{row("kevin_v2@B", "uptake_mode")}

**Prediction (spec): JEPA > MLM on uptake_mode → {verdict}.**

## Secondary — other fields on B, and A250

{head}
""" + "\n".join(row("kevin_v2@B", f) for f in FIELDS if f not in PRIMARY) + f"""

A250 ({len(A)} edges, Pass 2):

{head}
""" + "\n".join(row("kevin_v2@A250", f) for f in C.PASS2) + """

Not yet run (spec, secondary): linear-probe separability of credits vs uptake on frozen embeddings; compute cost.
"""
    (ROOT / f"docs/results/{day}_d2_jepa_scoring.md").write_text(note, encoding="utf-8")
    print(note)


if __name__ == "__main__":
    main()
