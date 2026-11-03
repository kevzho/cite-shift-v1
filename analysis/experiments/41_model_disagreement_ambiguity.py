#!/usr/bin/env python3
"""Exploratory Part A of private/preregistration/AMBIGUITY_AND_RETEST_SPEC_2026-10-25.md (fixed before running):
model disagreement among the three LLM judges (Sol, Jev, Qwen; identical codebook-v2 prompts) as a probe of where
codebook v2 is hard to apply, against Kevin v2 on sample B (registered sets). D1 excluded (trained on Sol).
Cluster bootstrap over components, 10,000 resamples, seed 20261003. Refuses to overwrite.
"""
from __future__ import annotations

import importlib.util
import json
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("conf", ROOT / "analysis/experiments/29_confirmatory_2026_10_08.py")
C = importlib.util.module_from_spec(spec)
spec.loader.exec_module(C)
S = C.S
RES = ROOT / "analysis/results"
OUT = RES / "exploratory/2026-10-25_model_disagreement_ambiguity.json"
SYSTEMS = {"gpt-6.1-sol": RES / "external_systems/hosted_llm_codebook_v2/edges_canonical.csv",
           "jev-1.13.0": RES / "external_systems/jev_codebook_v2/edges_canonical.csv",
           "qwen2.5-7b": RES / "external_systems/local_llm_codebook_v2/edges_v2edges.csv"}


def main():
    if OUT.exists():
        raise SystemExit(f"refusing to overwrite {OUT}")
    cw = S.crosswalk()
    alias, cluster = S.components()
    _, v2b_raw, _, _ = C.references(cw)
    ref, _ = C.drop_unclear(v2b_raw)
    B1, B2 = sorted(v2b_raw["citation_function"]), sorted(v2b_raw["uptake_mode"])
    boot = C.Boot(B1, cluster)
    lab = {}
    for name, path in SYSTEMS.items():
        d = {f: {} for f in C.PASS1 + C.PASS2}
        for r in C.read(path):
            for f in d:
                v = (r.get(f) or "").strip()
                if v and v != "not_applicable":
                    d[f][alias.get(r["edge_id"], r["edge_id"])] = S.norm(f, v, cw)
        lab[name] = d
    names = list(SYSTEMS)

    def wrate(edges, vals):
        """point rate and bootstrap CI of a 0/1 per-edge vector."""
        if not edges:
            return None
        idx = np.array([boot.idx(e) for e in edges]); v = np.array(vals, float)
        W = boot.W[:, idx].astype(float)
        b = (W * v).sum(1) / W.sum(1)
        return {"n": len(edges), "rate": C.r4(v.mean()), "ci95": C.ci(b)}

    fields = {}
    for f in C.PASS1 + C.PASS2:
        edges = (B1 if f in C.PASS1 else B2)
        rows = []
        for e in edges:
            if e not in ref[f] or any(e not in lab[n][f] for n in names):
                continue
            if f == "uptake_degree" and any(lab[n][f][e] not in ("full", "partial") for n in names):
                continue
            ms = [lab[n][f][e] for n in names]
            rows.append((e, ref[f][e], ms))
        agree = [(e, h, ms) for e, h, ms in rows if len(set(ms)) == 1]
        dis = [(e, h, ms) for e, h, ms in rows if len(set(ms)) > 1]
        hm = lambda sub: [np.mean([h == m for m in ms]) for _, h, ms in sub]
        fields[f] = {"llms_disagree": wrate([e for e, _, _ in rows], [len(set(ms)) > 1 for _, _, ms in rows]),
                     "human_llm_agreement_when_llms_agree": wrate([e for e, _, _ in agree], hm(agree)),
                     "human_llm_agreement_when_llms_disagree": wrate([e for e, _, _ in dis], hm(dis))}

    def binary(name, fn, edges, field, cond=None):
        rows = []
        for e in edges:
            if e not in ref[field] or any(e not in lab[n][field] for n in names):
                continue
            if cond and not cond(e):
                continue
            h = fn(ref[field][e]); ms = [fn(lab[n][field][e]) for n in names]
            rows.append((e, h, ms))
        maj = lambda ms: Counter(ms).most_common(1)[0][0]
        return {"inter_llm_disagreement": wrate([e for e, _, _ in rows], [len(set(ms)) > 1 for _, _, ms in rows]),
                "human_vs_llm_majority_disagreement": wrate([e for e, _, _ in rows], [h != maj(ms) for _, h, ms in rows])}

    up_all = lambda e: (ref["uptake_mode"].get(e, "none") != "none" and
                        all(lab[n]["uptake_mode"].get(e, "none") not in ("none", "unclear") for n in names))
    bins = {"credits_specific": binary("credits", lambda v: v == "specific", B2, "credits_contribution"),
            "uptake_any": binary("uptake", lambda v: v not in ("none",), B2, "uptake_mode"),
            "claim_dependent_yes": binary("claim", lambda v: v == "yes", B2, "claim_dependent"),
            "degree_partial": binary("degree", lambda v: v == "partial", B2, "uptake_degree", up_all)}
    named = {"(i) crediting vs uptake": ["credits_specific", "uptake_any"],
             "(ii) uptake vs claim dependence": ["uptake_any", "claim_dependent_yes"],
             "(iii) full vs partial": ["degree_partial"]}
    rank = sorted(bins, key=lambda k: -(bins[k]["human_vs_llm_majority_disagreement"] or {"rate": -1})["rate"])
    hardest = max(named, key=lambda k: max(bins[b]["human_vs_llm_majority_disagreement"]["rate"] for b in named[k]))
    out = {"spec": "private/preregistration/AMBIGUITY_AND_RETEST_SPEC_2026-10-25.md (Part A)",
           "judges": names, "per_field": fields, "binaries": bins,
           "rank_binaries_by_human_vs_llm_majority_disagreement": rank,
           "hardest_named_distinction": hardest,
           "caveat": "cannot separate instrument ambiguity from model limitations"}
    OUT.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
