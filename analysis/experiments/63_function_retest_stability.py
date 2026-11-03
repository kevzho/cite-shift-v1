#!/usr/bin/env python3
"""Registered stability (pre-registration §7): the 150-item function-only retest against the coder's first v1 coding.
Cohen's kappa and raw agreement on citation_function; never called inter-rater reliability. 'unclear' in either
coding is excluded (registered field-by-field rule); the all-items figures are reported alongside. 95% intervals:
bootstrap over items, 10,000 draws, seed 20261031.

    python3 analysis/experiments/63_function_retest_stability.py [--out PATH]
Default output: analysis/results/stability/function_retest_2026-10-31.json (refuses to overwrite).
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RT = ROOT / "annotations/generated/retest_2026-10-24"
V1 = ROOT / "annotations/coders/expansion_2026-09-26/annotator_kz_expansion_multidimensional.csv"
SEED, NB = 20261031, 10000


def read(p):
    with open(p, newline="", encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def kappa(pairs):
    n = len(pairs)
    if not n:
        return None
    po = sum(a == b for a, b in pairs) / n
    ca, cb = Counter(a for a, _ in pairs), Counter(b for _, b in pairs)
    pe = sum(ca[k] * cb[k] for k in set(ca) | set(cb)) / (n * n)
    return (po - pe) / (1 - pe) if pe < 1 else None


def summary(pairs, rng):
    k = kappa(pairs)
    raw = sum(a == b for a, b in pairs) / len(pairs)
    ks, rs = [], []
    for _ in range(NB):
        s = [pairs[rng.randrange(len(pairs))] for _ in pairs]
        kk = kappa(s)
        if kk is not None:
            ks.append(kk)
        rs.append(sum(a == b for a, b in s) / len(s))
    ks.sort(), rs.sort()
    ci = lambda v: [round(v[int(.025 * len(v))], 4), round(v[int(.975 * len(v)) - 1], 4)]
    return {"n": len(pairs), "kappa": round(k, 4), "kappa_ci95": ci(ks), "raw_agreement": round(raw, 4),
            "raw_ci95": ci(rs)}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, default=ROOT / "analysis/results/stability/function_retest_2026-10-31.json")
    a = ap.parse_args()
    if a.out.exists():
        sys.exit(f"refusing to overwrite {a.out}")
    key = {r["item"]: r["edge_id"] for r in read(RT / "function_retest_key.csv")}
    retest = {key[r["item"]]: r["citation_function"].strip() for r in read(RT / "function_retest_coded.csv")}
    first = {r["edge_id"].strip(): r["citation_function"].strip() for r in read(V1)}
    pairs_all = [(first[e], retest[e]) for e in sorted(retest)]
    pairs = [(f, s) for f, s in pairs_all if "unclear" not in (f, s)]
    rng = random.Random(SEED)
    conf = Counter(f"{f} -> {s}" for f, s in pairs_all if f != s)
    out = {"design": "same coder, v1 citation_function, retest on 31 Oct 2026 vs first coding (file saved 26 Sep 2026)",
           "primary_unclear_excluded": summary(pairs, rng), "all_items": summary(pairs_all, rng),
           "unclear": {"first": sum(f == "unclear" for f, _ in pairs_all), "retest": sum(s == "unclear" for _, s in pairs_all)},
           "first_distribution": dict(Counter(f for f, _ in pairs_all)), "retest_distribution": dict(Counter(s for _, s in pairs_all)),
           "changes": dict(conf.most_common())}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
