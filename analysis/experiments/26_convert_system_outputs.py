#!/usr/bin/env python3
"""Convert frozen external-system outputs into the scorer's long format, applying the registered
crosswalks (pre-registration §5). Writes one CSV per system and variant under
analysis/results/external_systems/scoring_inputs_<tag>/ (refuses to overwrite).

citation_function (scorable against canonical):
  SciCite  edge_label (argmax of the mean citance probability, harness rule fixed 26 Sep):
           background -> background, method -> uses_method, result -> compares_against.
  S2       intents (deployed, per pair): a single intent maps as SciCite (methodology -> uses_method);
           a pair with several intents makes no single claim and is excluded (count reported);
           no intent (no S2 context) is missing, not a prediction.
  Coarsening (§5): reference labels with no system class (applies_framework, builds_on_finding,
  reuses_artifact, other, unclear) are EXCLUDED from these systems' primary score (`_coarse` files,
  restricted to edges whose canonical function is background / uses_method / compares_against);
  `_all` files keep every edge (the registered sensitivity analysis: counted as errors).

Uptake-type constructs (no canonical field; used by RQ4(a), and later against Kevin v2):
  MultiCite (§5): highest-probability eligible label over edge_per_citance_max_probs, eligible =
           uses, extends, motivation, differences, background; uses -> method|artifact (merged),
           extends -> extension, motivation -> framing, differences -> contrast, background -> none;
           similarities / future_work make no claim. Using the paper's decision rule (sigmoid > 0.5 per
           class), the edge's claim is the highest-probability eligible label among those > 0.5; if none
           passes, the edge makes no claim (count reported).
  ACL-ARC  NOT registered (exploratory only): edge_mean_argmax; Uses/Extends/Motivation/
           CompareOrContrast -> uptake (as harness TO_V2), Background/Future -> none.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
R = ROOT / "analysis/results/external_systems"
spec = importlib.util.spec_from_file_location("scorer", ROOT / "analysis/experiments/22_score_systems.py")
S = importlib.util.module_from_spec(spec)
spec.loader.exec_module(S)

COARSE = {"background", "uses_method", "compares_against"}
SCICITE = {"background": "background", "method": "uses_method", "result": "compares_against"}
S2 = {"background": "background", "methodology": "uses_method", "result": "compares_against"}
MC = {"uses": "method|artifact", "extends": "extension", "motivation": "framing",
      "differences": "contrast", "background": "none"}
ACL = {"Uses": "method|artifact", "Extends": "extension", "Motivation": "framing",
       "CompareOrContrast": "contrast", "Background": "none", "Future": "none"}


def write(path, rows):
    if path.exists():
        raise SystemExit(f"refusing to overwrite {path}")
    with path.open("w", newline="", encoding="utf-8") as h:
        w = csv.DictWriter(h, fieldnames=["edge_id", "field", "label", "system", "corpus"])
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="canonical_20261004")
    a = ap.parse_args()
    out = R / f"scoring_inputs_{a.tag}"
    out.mkdir(exist_ok=True)
    cw = S.crosswalk()
    canon_fn = S.ref_canonical(cw)["citation_function"]
    alias, _ = S.components()
    notes = {}

    # SciCite
    rows = list(csv.DictReader(open(R / f"scicite_local/outputs/scicite_edge_preds_{a.tag}.csv")))
    pred = [{"edge_id": r["edge_id"], "field": "citation_function", "label": SCICITE[r["edge_label"]],
             "system": "scicite", "corpus": r["corpus"]} for r in rows]
    write(out / "scicite_function_all.csv", pred)
    write(out / "scicite_function_coarse.csv",
          [p for p in pred if canon_fn.get(alias.get(p["edge_id"], p["edge_id"])) in COARSE])
    notes["scicite"] = {"edges": len(rows), "labels": dict(Counter(p["label"] for p in pred)),
                        "edges_no_citance_passing_filter": sum(r["edge_agg_basis"] != "filtered" for r in rows)}

    # S2 intents
    rows = list(csv.DictReader(open(R / "s2_influential/outputs/canonical_20261004/s2_edge_labels.csv")))
    pred, multi, none = [], 0, 0
    for r in rows:
        ints = [i for i in r["intents"].split("|") if i]
        if not ints:
            none += 1
        elif len(ints) > 1:
            multi += 1
        else:
            pred.append({"edge_id": r["edge_id"], "field": "citation_function", "label": S2[ints[0]],
                         "system": "s2_intents", "corpus": r["corpus"]})
    write(out / "s2_intents_function_all.csv", pred)
    write(out / "s2_intents_function_coarse.csv",
          [p for p in pred if canon_fn.get(alias.get(p["edge_id"], p["edge_id"])) in COARSE])
    notes["s2_intents"] = {"pairs": len(rows), "single_intent_scored": len(pred),
                           "multi_intent_excluded": multi, "no_intent_missing": none}

    # S2 isInfluential (claim-dependent construct), only where S2 has contexts
    pred = [{"edge_id": r["edge_id"], "field": "claim_dependent",
             "label": "yes" if r["isInfluential"] == "True" else "no", "system": "s2_influential",
             "corpus": r["corpus"]} for r in rows if r["edge_evidence"] == "has_contexts"]
    write(out / "s2_influential_claim.csv", pred)
    notes["s2_influential"] = {"with_contexts": len(pred), "missing": len(rows) - len(pred),
                               "true": sum(p["label"] == "yes" for p in pred)}

    # MultiCite
    # §10: citances whose target marker is ambiguous (or absent) are excluded; an edge with no
    # unambiguously marked citance is excluded from MultiCite altogether (count reported).
    recs = [json.loads(l) for l in open(R / f"aclarc_multicite/outputs/multicite_scibert_{a.tag}.jsonl")]
    pred, noclaim, excluded, cit_dropped = [], 0, 0, 0
    for r in recs:
        keep = [c["probs"] for c, res in zip(r["per_citance"], r["marker_resolution"])
                if res in ("single_group", "intersection")]
        cit_dropped += len(r["per_citance"]) - len(keep)
        if not keep:
            excluded += 1
            continue
        p = {k: max(c[k] for c in keep) for k in keep[0]}
        elig = [(v, k) for k, v in p.items() if k in MC and v > 0.5]
        if not elig:
            noclaim += 1
            continue
        lab = MC[max(elig)[1]]
        pred.append({"edge_id": r["edge_id"], "field": "uptake_mode", "label": lab,
                     "system": "multicite", "corpus": r["corpus"]})
    write(out / "multicite_uptake.csv", pred)
    notes["multicite"] = {"edges": len(recs), "excluded_no_unambiguous_marker": excluded,
                          "citances_dropped_ambiguous_or_no_marker": cit_dropped,
                          "with_claim": len(pred), "no_claim": noclaim,
                          "labels": dict(Counter(p["label"] for p in pred))}

    # ACL-ARC (exploratory; optional until its full run exists)
    acl = R / f"aclarc_multicite/outputs/aclarc_scaffold_{a.tag}.jsonl"
    if acl.exists():
        recs = [json.loads(l) for l in open(acl)]
        pred = [{"edge_id": r["edge_id"], "field": "uptake_mode", "label": ACL[r["edge_mean_argmax"]],
                 "system": "aclarc_exploratory", "corpus": r["corpus"]} for r in recs]
        write(out / "aclarc_uptake_exploratory.csv", pred)
        notes["aclarc"] = {"edges": len(recs), "labels": dict(Counter(r["edge_mean_argmax"] for r in recs))}
    else:
        notes["aclarc"] = "not run to completion (run stopped by the system for low memory, 4 Oct)"

    (out / "conversion_notes.json").write_text(json.dumps(notes, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(notes, indent=1))


if __name__ == "__main__":
    main()
