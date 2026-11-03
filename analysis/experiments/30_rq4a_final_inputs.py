#!/usr/bin/env python3
"""RQ4(a) final lineup (11 Oct 2026): long-format inputs for 24_rq4a_hci_vs_nature.py --extra.

Adds the systems missing from the 4 Oct run (docs/results/2026-10-04_local_systems_and_scoring.md §3):
  * Fine-tuned encoder (registered system; SciBERT, pooled train, CPU deterministic, seeds 1-5), task realization,
    unconditioned (primary) and contribution-conditioned arms. Only its OUT-OF-SAMPLE predictions exist on
    canonical edges: the 1,631 frozen-partition edges (--predict-frozen, 4 Oct). Train/dev edges would be
    in-sample (circular) and its test-split predictions were not retained, so the encoder is scored on the
    frozen subset only -- as S2 is scored only where it has context. One series per seed (mean and spread, §4).
  * canonical@frozen: the canonical realized label on the same frozen subset, so the encoder's Δ has a
    same-edge human comparison.
  * D1 (exploratory; not a registered RQ4 system, §4 "own aim"): frozen-edge predictions, seeds 1-3, credits /
    uptake / claim. Trained on Sol silver labels only, so every frozen edge is out of sample.
No-model baseline: one majority class for both corpora, so Δ = 0 by construction; reported in the note, not run.
Writes analysis/results/external_systems/scoring_inputs_rq4a_final_20261011/ (refuses to overwrite).
"""
import csv
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "analysis/results/external_systems/scoring_inputs_rq4a_final_20261011"
FP = ROOT / "analysis/results/ml/frozen_predictions/realization"
D1 = ROOT / "analysis/results/ml/d1_2026-10-07"
COLS = ["edge_id", "field", "label", "system", "corpus"]


def write(name, rows):
    p = OUT / name
    if p.exists():
        raise SystemExit(f"refusing to overwrite {p}")
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        w.writerows(rows)
    print(f"{name}: {len(rows)} rows")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    frozen = set()
    for arm in ("uncond", "cond"):
        for s in range(1, 6):
            rows = []
            for c in ("hci", "nature_oa"):
                for r in csv.DictReader(open(FP / f"scibert_pooled_{arm}_{c}_seed{s}.csv", encoding="utf-8")):
                    frozen.add(r["edge_id_v1"])
                    rows.append({"edge_id": r["edge_id_v1"], "field": "realization", "label": r["y_pred"],
                                 "system": f"encoder_{arm}_s{s}", "corpus": c})
            write(f"encoder_{arm}_seed{s}_realization.csv", rows)
    rows = []
    for c in ("hci", "nature_oa"):
        for r in csv.DictReader(open(ROOT / f"analysis/derived/{c}_contribution_coding_human_reviewed.csv",
                                     encoding="utf-8-sig")):
            e = hashlib.sha256("\x1f".join((c, r["cited_id"].strip(), r["citing_id"].strip())).encode()).hexdigest()[:20]
            if e in frozen:
                rows.append({"edge_id": e, "field": "realization", "label": r["realizes_contribution"].strip(),
                             "system": "canonical@frozen", "corpus": c})
    write("canonical_frozen_subset_realization.csv", rows)
    for s in (1, 2, 3):
        rows = []
        for r in csv.DictReader(open(D1 / f"d1_frozen_predictions_seed{s}.csv", encoding="utf-8")):
            for f in ("credits_contribution", "uptake_mode", "claim_dependent"):
                rows.append({"edge_id": r["edge_id"], "field": f, "label": r[f],
                             "system": f"d1_exploratory_s{s}", "corpus": r["corpus"]})
        write(f"d1_seed{s}_exploratory.csv", rows)


if __name__ == "__main__":
    main()
