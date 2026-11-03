#!/usr/bin/env python3
"""Exploratory, post hoc (private/preregistration/RQ3_LIKE_FOR_LIKE_SPEC_2026-11-02.md, fixed before running): RQ3
reference reversals when both systems of a pair are scored on identical edges against identical reference classes.

Per field and system pair: common edges E = B edges with a non-unclear label in canonical and in Kevin v2, and a
prediction from every seed of both systems (and, if either system is SciCite or S2 intents, both reference labels in
the coarse function space). Common classes K = labels with support >= 5 in both references on E; pairs with |K| < 2
are not scored. Macro-F1 over K, seed mean, registered cluster bootstrap (10,000 draws, seed 20261003).

    python3 analysis/experiments/68_rq3_like_for_like.py [--out PATH]
Default output: analysis/results/rq3_like_for_like_2026-11-02.json (refuses to overwrite).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections import Counter
from itertools import combinations
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("conf", ROOT / "analysis/experiments/29_confirmatory_2026_10_08.py")
C = importlib.util.module_from_spec(spec)
spec.loader.exec_module(C)
S = C.S
REGISTERED = ROOT / "analysis/results/confirmatory_2026-10-08/rq3.json"
COARSE_SYS = {"scicite", "s2_intents"}
D1 = "d1_distilled"


def boot_f1(edges, ref, preds_by_seed, K, boot):
    """Seed-mean macro-F1 over classes K: (full, bootstrap vector)."""
    labels = sorted(set(K) | {ref[e] for e in edges} | {p[e] for p in preds_by_seed for e in edges})
    rep = np.array([l in K for l in labels])
    full, bv = [], []
    for p in preds_by_seed:
        f, b = boot.confusions([(e, ref[e], p[e]) for e in edges], labels)
        full.append(S.metrics_from_confusion(f, rep)[0])
        bv.append(S.metrics_from_confusion(b, rep)[0])
    return float(np.mean(full)), np.nanmean(np.stack(bv), axis=0)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, default=ROOT / "analysis/results/rq3_like_for_like_2026-11-02.json")
    a = ap.parse_args()
    if a.out.exists():
        sys.exit(f"refusing to overwrite {a.out}")
    cw = S.crosswalk()
    alias, cluster = S.components()
    canon, v2b_raw, _, _ = C.references(cw)
    v2b, _ = C.drop_unclear(v2b_raw)
    B1 = sorted(set(v2b_raw["citation_function"]))
    boot = C.Boot(B1, cluster)
    sysp = C.load_systems(cw, alias)
    registered = {(x["field"], tuple(x["pair"])): x["p_flip"]
                  for x in json.load(open(REGISTERED))["flip_probabilities_between_references"]}
    out = {"spec": "private/preregistration/RQ3_LIKE_FOR_LIKE_SPEC_2026-11-02.md", "pairs": [], "summary": {}}

    for field in C.PASS1:
        cref, kref = canon[field], v2b[field]
        systems = sorted(s for s in sysp if any(field in f for f in sysp[s].values()))
        for s1, s2 in combinations(systems, 2):
            if (field, (s1, s2)) not in registered:
                continue                                   # only pairs the registered test compared
            p1 = [f[field] for f in sysp[s1].values() if field in f]
            p2 = [f[field] for f in sysp[s2].values() if field in f]
            E = [e for e in B1 if e in kref and cref.get(e) not in (None, "unclear")
                 and all(e in p for p in p1 + p2)]
            if {s1, s2} & COARSE_SYS and field == "citation_function":
                E = [e for e in E if cref[e] in C.COARSE_FN and kref[e] in C.COARSE_FN]
            sc, sk = Counter(cref[e] for e in E), Counter(kref[e] for e in E)
            K = sorted(l for l in set(sc) | set(sk) if sc[l] >= S.MIN_SUPPORT and sk[l] >= S.MIN_SUPPORT)
            row = {"field": field, "pair": [s1, s2], "n_edges": len(E), "classes": K,
                   "registered_p_flip": registered[(field, (s1, s2))]}
            if len(K) < 2:
                row["status"] = "not scored (fewer than 2 common classes)"
                out["pairs"].append(row)
                continue
            f = {}
            for refname, ref in (("canonical", cref), ("kevin_v2", kref)):
                for s, p in ((s1, p1), (s2, p2)):
                    f[(refname, s)] = boot_f1(E, ref, p, K, boot)
            x = np.sign(f[("canonical", s1)][1] - f[("canonical", s2)][1])
            y = np.sign(f[("kevin_v2", s1)][1] - f[("kevin_v2", s2)][1])
            row.update({"macro_f1": {f"{r}|{s}": round(v[0], 4) for (r, s), v in f.items()},
                        "p_flip": round(float(np.mean(x != y)), 4)})
            out["pairs"].append(row)

    scored = [r for r in out["pairs"] if "p_flip" in r]
    no_d1 = [r for r in scored if D1 not in r["pair"]]
    out["summary"] = {
        "registered": "5/27 (2/18 without D1)",
        "pairs_compared": len(out["pairs"]), "pairs_scored": len(scored),
        "pairs_not_scored": [r["pair"] + [r["field"]] for r in out["pairs"] if "p_flip" not in r],
        "reversals_gt_0.5": f"{sum(r['p_flip'] > 0.5 for r in scored)}/{len(scored)}",
        "reversals_gt_0.5_without_D1": f"{sum(r['p_flip'] > 0.5 for r in no_d1)}/{len(no_d1)}",
        "reversing_pairs": [[r["field"]] + r["pair"] + [r["p_flip"], r["n_edges"]] for r in scored if r["p_flip"] > 0.5]}
    out["verdict"] = ("SURVIVES (at least one pair without D1 reverses with p > 0.5 like-for-like)"
                      if any(r["p_flip"] > 0.5 for r in no_d1) else
                      "DOES NOT SURVIVE (no pair without D1 reverses with p > 0.5 like-for-like)")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out["summary"], indent=1), "\n", out["verdict"])


if __name__ == "__main__":
    main()
