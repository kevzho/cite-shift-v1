#!/usr/bin/env python3
"""Exploratory, post hoc (private/preregistration/RQ3_LIKE_FOR_LIKE_SPEC_2026-11-02.md, Part 2, fixed before running):
RQ3 between-construct reversals when both systems are scored on one common edge set for both fields, against
classes with support >= 5 in Kevin v2 (B300) on that set.

    python3 analysis/experiments/69_rq3_constructs_like_for_like.py [--out PATH]
Default output: analysis/results/rq3_constructs_like_for_like_2026-11-03.json (refuses to overwrite).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("conf", ROOT / "analysis/experiments/29_confirmatory_2026_10_08.py")
C = importlib.util.module_from_spec(spec)
spec.loader.exec_module(C)
spec68 = importlib.util.spec_from_file_location("l4l", ROOT / "analysis/experiments/68_rq3_like_for_like.py")
L = importlib.util.module_from_spec(spec68)
spec68.loader.exec_module(L)
S = C.S
THIRD = C.V2 / "received/B_stratum_pass2_as_submitted_2026-10-05_2223.csv"     # as in 38_sensitivity_B300.py
REGISTERED = ROOT / "analysis/results/sensitivity_2026-10-23_B300/rq3.json"
MERGE = {"method": "method|artifact", "artifact": "method|artifact"}
D1 = "d1_distilled"


def kevin_b300(cw):
    _, v2b, _, _ = C.references(cw)
    edge = {k["item"]: k["edge_id"] for k in C.read(C.V2 / "B_stratum_pass2_key.csv")}
    for row in C.read(THIRD):
        for f in C.PASS2:
            v = (row.get(f) or "").strip()
            if v and v != "not_applicable":
                v2b[f][edge[row["item"]]] = v
    return v2b


def field_view(field, pair, kref, preds):
    """(eligible edge set, ref, per-system per-seed preds) after the field's registered eligibility, applied alike."""
    ref = dict(kref)
    preds = {s: [dict(p) for p in ps] for s, ps in preds.items()}
    E = {e for e in ref if all(e in p for ps in preds.values() for p in ps)}
    if field == "citation_function" and set(pair) & L.COARSE_SYS:
        E = {e for e in E if ref[e] in C.COARSE_FN}
    if field == "uptake_mode" and "multicite" in pair:
        ref = {e: MERGE.get(v, v) for e, v in ref.items()}
        preds = {s: [{e: MERGE.get(v, v) for e, v in p.items()} for p in ps] for s, ps in preds.items()}
        E = {e for e in E if ref[e] in C.MC_CLASSES}
    if field == "uptake_degree":
        E = {e for e in E if ref[e] in ("full", "partial")
             and all(p[e] in ("full", "partial") for ps in preds.values() for p in ps)}
    return E, ref, preds


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, default=ROOT / "analysis/results/rq3_constructs_like_for_like_2026-11-03.json")
    a = ap.parse_args()
    if a.out.exists():
        sys.exit(f"refusing to overwrite {a.out}")
    cw = S.crosswalk()
    alias, cluster = S.components()
    v2b, _ = C.drop_unclear(kevin_b300(cw))
    boot = C.Boot(sorted(set(C.references(cw)[1]["citation_function"])), cluster)   # registered bootB (300 edges)
    sysp = C.load_systems(cw, alias)
    registered = json.load(open(REGISTERED))["flip_probabilities_between_fields"]
    out = {"spec": "private/preregistration/RQ3_LIKE_FOR_LIKE_SPEC_2026-11-02.md (Part 2)", "pairs": []}
    for reg in registered:
        f1, f2 = reg["fields"]
        s1, s2 = reg["pair"]
        views = {}
        for f in (f1, f2):
            preds = {s: [fl[f] for fl in sysp[s].values() if f in fl] for s in (s1, s2)}
            views[f] = field_view(f, (s1, s2), v2b[f], preds)
        E = sorted(views[f1][0] & views[f2][0])
        row = {"fields": [f1, f2], "pair": [s1, s2], "n_edges": len(E), "registered_p_flip": reg["p_flip"]}
        f1s = {}
        for f in (f1, f2):
            _, ref, preds = views[f]
            sup = Counter(ref[e] for e in E)
            K = sorted(l for l, c in sup.items() if c >= S.MIN_SUPPORT)
            row[f"classes|{f}"] = K
            if len(K) < 2:
                row["status"] = f"not scored ({f}: fewer than 2 classes with support >= 5)"
                break
            for s in (s1, s2):
                f1s[(f, s)] = L.boot_f1(E, ref, preds[s], K, boot)
        if "status" not in row:
            x = np.sign(f1s[(f1, s1)][1] - f1s[(f1, s2)][1])
            y = np.sign(f1s[(f2, s1)][1] - f1s[(f2, s2)][1])
            row.update({"macro_f1": {f"{f}|{s}": round(v[0], 4) for (f, s), v in f1s.items()},
                        "p_flip": round(float(np.mean(x != y)), 4)})
        out["pairs"].append(row)
    scored = [r for r in out["pairs"] if "p_flip" in r]
    no_d1 = [r for r in scored if D1 not in r["pair"]]
    out["summary"] = {
        "registered_B300": "14/48 (9/17 without D1)",
        "pairs_scored": f"{len(scored)}/{len(out['pairs'])}",
        "reversals_gt_0.5": f"{sum(r['p_flip'] > 0.5 for r in scored)}/{len(scored)}",
        "reversals_gt_0.5_without_D1": f"{sum(r['p_flip'] > 0.5 for r in no_d1)}/{len(no_d1)}",
        "reversing_pairs": [r["fields"] + r["pair"] + [r["p_flip"], r["n_edges"]] for r in scored if r["p_flip"] > 0.5]}
    out["verdict"] = ("SURVIVES (at least one pair without D1 reverses with p > 0.5 like-for-like)"
                      if any(r["p_flip"] > 0.5 for r in no_d1) else
                      "DOES NOT SURVIVE (no pair without D1 reverses with p > 0.5 like-for-like)")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out["summary"], indent=1), "\n", out["verdict"])


if __name__ == "__main__":
    main()
