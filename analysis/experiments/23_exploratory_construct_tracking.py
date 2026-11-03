#!/usr/bin/env python3
"""EXPLORATORY preview of RQ1(b) with measured systems standing in for the human v2 coder.

Registered RQ1(b) uses Kevin's v2 labels on sample B. Those do not exist yet. This script asks the
same question of each LLM's v2 labels on all canonical edges: does canonical `realizes_contribution`
(realized = {partial, yes}) agree more with the system's `credits_contribution = specific` (crediting)
or with `uptake_mode not in {none, unclear}` (use)? It is labelled exploratory everywhere it is
reported and never substitutes for the registered test. No v2 human label is read.

Cohen's kappa for each pairing, the difference (crediting - uptake), and a cluster bootstrap over
citation-graph components (analysis/ml/splits) for the difference. Also restricted to canonical
background citations, where the canonical record most often says `partial`.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SYSTEMS = {
    "gpt-6.1-sol": ROOT / "analysis/results/external_systems/hosted_llm_codebook_v2/edges_canonical.csv",
    "jev-1.13.0": ROOT / "analysis/results/external_systems/jev_codebook_v2/edges_canonical.csv",
}


def edge_id(corpus, cited, citing):
    return hashlib.sha256("\x1f".join((corpus, cited.strip(), citing.strip())).encode()).hexdigest()[:20]


def kappa(pairs):
    n = len(pairs)
    if not n:
        return None
    po = sum(a == b for a, b in pairs) / n
    pa = sum(a for a, _ in pairs) / n
    pb = sum(b for _, b in pairs) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return (po - pe) / (1 - pe) if pe < 1 else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20261003)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    canon, comp = {}, {}
    for c in ("hci", "nature_oa"):
        for r in csv.DictReader(open(ROOT / f"analysis/derived/{c}_contribution_coding_human_reviewed.csv",
                                     encoding="utf-8-sig")):
            canon[edge_id(c, r["cited_id"], r["citing_id"])] = (c, r["citance_function"].strip(),
                                                               r["realizes_contribution"].strip())
        for s in csv.DictReader(open(ROOT / f"analysis/ml/splits/{c}_splits.csv", encoding="utf-8")):
            comp[s["edge_id_v1"]] = f"{c}:{s['component_key']}"

    out = {"status": "EXPLORATORY - systems stand in for the human v2 coder; not the registered RQ1(b)",
           "realized": "canonical realizes_contribution in {partial, yes}", "systems": {}}
    for name, path in SYSTEMS.items():
        rows = []
        for r in csv.DictReader(open(path, encoding="utf-8")):
            e = r["edge_id"]
            if e not in canon or not r["credits_contribution"] or not r["uptake_mode"]:
                continue
            if r["credits_contribution"] == "unclear" or r["uptake_mode"] == "unclear":
                continue
            corpus, fn, real = canon[e]
            rows.append({"e": e, "corpus": corpus, "bg": fn == "background_bulk",
                         "realized": real in ("partial", "yes"),
                         "credits": r["credits_contribution"] == "specific",
                         "uptake": r["uptake_mode"] != "none"})
        res = {}
        for subset, sel in (("all", lambda x: True), ("canonical_background", lambda x: x["bg"])):
            sub = [x for x in rows if sel(x)]
            kc = kappa([(x["realized"], x["credits"]) for x in sub])
            ku = kappa([(x["realized"], x["uptake"]) for x in sub])
            by = defaultdict(list)
            for x in sub:
                by[comp.get(x["e"], x["e"])].append(x)
            keys = sorted(by)
            rng = random.Random(a.seed)
            diffs = []
            for _ in range(a.n_boot):
                bs = [x for k in rng.choices(keys, k=len(keys)) for x in by[k]]
                c1 = kappa([(x["realized"], x["credits"]) for x in bs])
                u1 = kappa([(x["realized"], x["uptake"]) for x in bs])
                if c1 is not None and u1 is not None:
                    diffs.append(c1 - u1)
            diffs.sort()
            res[subset] = {
                "n": len(sub),
                "canonical_realized_rate": round(sum(x["realized"] for x in sub) / len(sub), 3),
                "system_credits_specific_rate": round(sum(x["credits"] for x in sub) / len(sub), 3),
                "system_uptake_rate": round(sum(x["uptake"] for x in sub) / len(sub), 3),
                "kappa_realized_vs_credits": round(kc, 3),
                "kappa_realized_vs_uptake": round(ku, 3),
                "difference_credits_minus_uptake": round(kc - ku, 3),
                "difference_95ci": [round(diffs[int(0.025 * len(diffs))], 3),
                                    round(diffs[int(0.975 * len(diffs)) - 1], 3)],
            }
        out["systems"][name] = res
    if a.out.exists():
        raise SystemExit(f"refusing to overwrite {a.out}")
    a.out.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
