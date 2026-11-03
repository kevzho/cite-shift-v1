#!/usr/bin/env python3
"""Exploratory (spec private/preregistration/S2_CONSEQUENCE_SPEC_2026-10-12.md, fixed before this ran): does a
published group comparison by S2 influential-citation share (Cunningham & Smyth, CACM 2025) survive a valid reference?
Δ = HCI - Nature OA; cluster bootstrap over components (10,000, seed 20261003). Refuses to overwrite.
"""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("conf", ROOT / "analysis/experiments/29_confirmatory_2026_10_08.py")
C = importlib.util.module_from_spec(spec)
spec.loader.exec_module(C)
S = C.S
RES = ROOT / "analysis/results"
OUT = RES / "exploratory/2026-10-12_s2_consequence.json"


def main():
    if OUT.exists():
        raise SystemExit(f"refusing to overwrite {OUT}")
    cw = S.crosswalk()
    alias, cluster = S.components()
    a = lambda e: alias.get(e, e)
    s2 = {}
    for r in C.read(RES / "external_systems/s2_influential/outputs/canonical_20261004/s2_edge_labels.csv"):
        if r["isInfluential"] in ("True", "False"):
            s2[a(r["edge_id"])] = (r["corpus"], r["isInfluential"] == "True", r["edge_evidence"] == "has_contexts")
    edges = sorted(s2)
    corpus = {e: s2[e][0] for e in edges}
    boot = C.Boot(edges, cluster)
    idx = np.array([boot.idx(e) for e in edges])
    W = boot.W[:, idx].astype(float)                                   # (n_boot, n_edges)
    pos = {e: i for i, e in enumerate(edges)}

    def delta(series):
        """series {edge: bool} -> rates, Δ, CI (edges outside `edges` ignored)."""
        m = np.zeros(len(edges)); v = np.zeros(len(edges))
        for e, x in series.items():
            if e in pos:
                m[pos[e]] = 1; v[pos[e]] = float(x)
        h = np.array([corpus[e] == "hci" for e in edges]) * m
        n = np.array([corpus[e] == "nature_oa" for e in edges]) * m
        rate = lambda w, g: (w * g * v).sum(-1) / (w * g).sum(-1)
        rh, rn = rate(np.ones(len(edges)), h), rate(np.ones(len(edges)), n)
        b = rate(W, h) - rate(W, n)
        ci = C.ci(b)
        return {"n_hci": int(h.sum()), "n_nature": int(n.sum()), "rate_hci": C.r4(rh), "rate_nature": C.r4(rn),
                "delta": C.r4(rh - rn), "ci95": ci,
                "sign": "HCI>Nature" if ci[0] > 0 else "Nature>HCI" if ci[1] < 0 else "n.s."}

    canon = {}
    for c in ("hci", "nature_oa"):
        for r in csv.DictReader(open(ROOT / f"analysis/derived/{c}_contribution_coding_human_reviewed.csv", encoding="utf-8-sig")):
            e = hashlib.sha256("\x1f".join((c, r["cited_id"].strip(), r["citing_id"].strip())).encode()).hexdigest()[:20]
            v = r["realizes_contribution"].strip()
            if v in ("no", "partial", "yes"):
                canon[a(e)] = v in ("partial", "yes")
    _, v2b_raw, _, _ = C.references(cw)
    kev_up = {a(e): u != "none" for e, u in v2b_raw["uptake_mode"].items() if u != "unclear"}
    kev_cr = {a(e): u == "specific" for e, u in v2b_raw["credits_contribution"].items() if u in ("specific", "generic")}
    sol_up = {}
    for r in C.read(RES / "external_systems/hosted_llm_codebook_v2/edges_canonical.csv"):
        if r["uptake_mode"] and r["uptake_mode"] != "unclear":
            sol_up[a(r["edge_id"])] = r["uptake_mode"] != "none"
    ctx = {e for e in edges if s2[e][2]}

    res = {"coverage_with_s2_context": delta({e: s2[e][2] for e in edges}),
           "flag_all_citations_missing_as_false": delta({e: s2[e][1] for e in edges}),
           "flag_context_only": delta({e: s2[e][1] for e in ctx}),
           "canonical_realized_same_edges": delta({e: canon[e] for e in edges if e in canon}),
           "canonical_realized_context_edges": delta({e: canon[e] for e in ctx if e in canon}),
           "sol_uptake_same_edges": delta({e: sol_up[e] for e in edges if e in sol_up}),
           "kevin_v2_uptake_B": delta({e: kev_up[e] for e in edges if e in kev_up}),
           "kevin_v2_credits_B": delta({e: kev_cr[e] for e in edges if e in kev_cr})}

    def validity(ref, subset):
        out = {}
        for c in ("hci", "nature_oa", "pooled"):
            es = [e for e in subset if e in ref and (c == "pooled" or corpus[e] == c)]
            f = np.array([s2[e][1] for e in es], float); h = np.array([ref[e] for e in es], float)
            ii = np.array([pos[e] for e in es])
            def stats(w):
                tp, fp, fn = (w * f * h).sum(-1), (w * f * (1 - h)).sum(-1), (w * (1 - f) * h).sum(-1)
                n = w.sum(-1); po = (w * (f == h)).sum(-1) / n
                pf, ph = (w * f).sum(-1) / n, (w * h).sum(-1) / n
                pe = pf * ph + (1 - pf) * (1 - ph)
                with np.errstate(invalid="ignore", divide="ignore"):
                    return tp / (tp + fp), tp / (tp + fn), (po - pe) / (1 - pe)
            p, r, k = stats(np.ones(len(es)))
            bp, br, bk = stats(W[:, ii])
            out[c] = {"n": len(es), "human_positive_rate": C.r4(h.mean()), "flag_rate": C.r4(f.mean()),
                      "precision": C.r4(p), "precision_ci95": C.ci(bp), "recall": C.r4(r), "recall_ci95": C.ci(br),
                      "kappa": C.r4(k), "kappa_ci95": C.ci(bk)}
        return out

    res["validity_vs_canonical_realized_context_edges"] = validity(canon, sorted(ctx))
    res["validity_vs_kevin_v2_uptake_B_context_edges"] = validity(kev_up, sorted(ctx))
    fa, fc = res["flag_all_citations_missing_as_false"], res["flag_context_only"]
    human = [res[k] for k in ("canonical_realized_same_edges", "sol_uptake_same_edges")]
    changes = any(fa["sign"] != "n.s." and h["sign"] != "n.s." and fa["sign"] != h["sign"] for h in human) or \
        (fa["sign"] != "n.s." and fc["sign"] == "n.s.")
    confound = res["coverage_with_s2_context"]["sign"] != "n.s." and fa["sign"] != fc["sign"]
    res["reading"] = {"conclusion_changes": changes, "coverage_confound": confound,
                      "rule": "spec §Pre-stated reading"}
    OUT.write_text(json.dumps(res, indent=2) + "\n")
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
