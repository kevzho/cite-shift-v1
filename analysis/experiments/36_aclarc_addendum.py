#!/usr/bin/env python3
"""Exploratory addendum (21 Oct 2026; DEVIATIONS.md, same date): ACL-ARC scaffold (Cohan et al. 2019) on the canonical
edges. Crosswalk = ACL in 26_convert_system_outputs.py (fixed 4 Oct). Scoring = 29_confirmatory_2026_10_08.py machinery,
with ACL-ARC treated like MultiCite (method/artifact merged; primary restricted to expressible classes). Also the
RQ4(a) Δ via 24_rq4a_hci_vs_nature.py. Enters no registered verdict. Refuses to overwrite.
"""
from __future__ import annotations

import csv
import importlib.util
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def mod(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


C = mod("conf", "analysis/experiments/29_confirmatory_2026_10_08.py")
CONV = mod("conv", "analysis/experiments/26_convert_system_outputs.py")
S = C.S
RES = ROOT / "analysis/results"
SRC = RES / "external_systems/aclarc_multicite/outputs/aclarc_scaffold_canonical_20261021.jsonl"
INP = RES / "external_systems/scoring_inputs_aclarc_20261021"
OUT = RES / "exploratory/2026-10-21_aclarc_addendum.json"


def main():
    if OUT.exists():
        raise SystemExit(f"refusing to overwrite {OUT}")
    recs = [json.loads(l) for l in open(SRC, encoding="utf-8")]
    INP.mkdir(parents=True, exist_ok=True)
    long_path = INP / "aclarc_uptake_exploratory.csv"
    if long_path.exists():
        raise SystemExit(f"refusing to overwrite {long_path}")
    with open(long_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["edge_id", "field", "label", "system", "corpus"])
        w.writeheader()
        for r in recs:
            w.writerow({"edge_id": r["edge_id"], "field": "uptake_mode", "label": CONV.ACL[r["edge_mean_argmax"]],
                        "system": "aclarc_exploratory", "corpus": r["corpus"]})
    rq4 = RES / "exploratory/2026-10-21_aclarc_rq4a.json"
    subprocess.run([sys.executable, str(ROOT / "analysis/experiments/24_rq4a_hci_vs_nature.py"), "--out", str(rq4),
                    "--extra", str(long_path)], check=True, capture_output=True)
    rq4_row = [x for x in json.loads(rq4.read_text())["results"] if x["source"] == "aclarc_exploratory"]

    cw = S.crosswalk()
    alias, cluster = S.components()
    _, v2b_raw, v2a_raw, _ = C.references(cw)
    v2b, _ = C.drop_unclear(v2b_raw)
    v2a, _ = C.drop_unclear(v2a_raw)
    B1, B2, A = sorted(v2b_raw["citation_function"]), sorted(v2b_raw["uptake_mode"]), sorted(v2a_raw["uptake_mode"])
    pred = {alias.get(r["edge_id"], r["edge_id"]): CONV.ACL[r["edge_mean_argmax"]] for r in recs}
    scores = {}
    for name, ref, edges, boot in (("kevin_v2@B", v2b, B2, C.Boot(B1, cluster)), ("kevin_v2@A250", v2a, A, C.Boot(A, cluster))):
        for variant in ("primary", "sensitivity_all"):
            pairs = []
            for e in edges:
                if e in ref["uptake_mode"] and e in pred:
                    ep = C.eligible_pair("multicite", "uptake_mode", ref["uptake_mode"][e], pred[e], variant)
                    if ep:
                        pairs.append((e, *ep))
            m = C.macro_f1(pairs, boot)
            scores[f"{name}|uptake_mode|{variant}"] = {"n": m["n"], "macro_f1": C.r4(m["macro_f1"]),
                                                        "ci95": C.ci(m["boot_f1"]), "kappa": C.r4(m["kappa"]),
                                                        "withheld": m["withheld"]}
    bootB = C.Boot(B1, cluster)
    crB, upB, clB = v2b_raw["credits_contribution"], v2b_raw["uptake_mode"], v2b_raw["claim_dependent"]
    kb = {"credits_specific": {e: crB[e] == "specific" for e in B2 if crB.get(e) in ("specific", "generic")},
          "uptake": {e: upB[e] != "none" for e in B2 if upB.get(e) not in (None, "unclear")},
          "claim_dependent_yes": {e: clB[e] == "yes" for e in B2 if clB.get(e) in ("yes", "no")}}
    sb = {e: v != "none" for e, v in pred.items()}
    recovery = {}
    for k, kv in kb.items():
        pairs = [(e, sb[e], kv[e]) for e in B2 if e in sb and e in kv]
        (kap, nmi), _ = C.binary_stats(pairs, bootB)
        recovery[k] = {"n": len(pairs), "kappa": C.r4(kap), "nmi_geometric": C.r4(nmi)}
    # rank among the registered uptake_mode systems on B (8 Oct scores, unchanged)
    old = json.loads((RES / "confirmatory_2026-10-08/rq3.json").read_text())["rankings_macro_f1"]["kevin_v2@B|uptake_mode"]
    out = {"status": "EXPLORATORY; enters no registered verdict", "edges_scored": len(recs),
           "label_counts": {k: sum(r["edge_mean_argmax"] == k for r in recs) for k in CONV.ACL},
           "rq4a_uptake_delta": rq4_row, "uptake_mode_vs_kevin_v2": scores, "construct_recovery_B": recovery,
           "registered_uptake_mode_ranking_B_8_oct": old}
    OUT.write_text(json.dumps(out, indent=2, default=float) + "\n")
    print(json.dumps(out, indent=2, default=float))


if __name__ == "__main__":
    main()
