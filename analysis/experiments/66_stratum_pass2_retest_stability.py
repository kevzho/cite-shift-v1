#!/usr/bin/env python3
"""Registered stability (pre-registration §7): the 60-edge stratum Pass 2 retest against the coder's first v2 Pass 2
coding of the stratum (5 Oct 22:23, the file used in the B300 primary analysis). Cohen's kappa and raw agreement per field; never called inter-rater reliability. 'unclear' in either
coding is excluded (registered field-by-field rule); the all-items figures are reported alongside. strongest_citance is
compared as a citance number (a leading "[n]" in the text is read as n), and also on the items with more than one citance (single-citance items agree trivially).
uptake_mode is also collapsed to any uptake vs none (the RQ1 construct). uptake_degree is compared only where both
codings give a degree (it is blank when the mode is none or unclear).
95% intervals: bootstrap over items, 10,000 draws, seed 20261031.

    python3 analysis/experiments/66_stratum_pass2_retest_stability.py [--out PATH]
Default output: analysis/results/stability/stratum_pass2_retest_2026-11-02.json (refuses to overwrite).
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RT = ROOT / "annotations/generated/retest_2026-10-31"
V2 = ROOT / "annotations/generated/v2_coding_2026-10-02"
FIELDS = ["strongest_citance", "credits_contribution", "uptake_mode", "any_uptake", "uptake_degree", "claim_dependent"]
SEED, NB = 20261031, 10000


def read(p):
    with open(p, newline="", encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def norm(field, row):
    if field == "any_uptake":
        m = row["uptake_mode"].strip()
        return m if m in ("none", "unclear") else "uptake"
    v = row[field].strip()
    if field == "strongest_citance" and v:
        m = re.match(r"\s*\[?(\d+(?:\.0)?)\]?", v)  # the first coding wrote some as the citance text "[n] ..."
        return str(int(float(m.group(1)))) if m else v
    return v


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
    ci = lambda v: [round(v[int(.025 * len(v))], 4), round(v[int(.975 * len(v)) - 1], 4)] if v else None  # None: kappa undefined
    return {"n": len(pairs), "kappa": None if k is None else round(k, 4), "kappa_ci95": ci(ks),
            "raw_agreement": round(raw, 4), "raw_ci95": ci(rs)}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, default=ROOT / "analysis/results/stability/stratum_pass2_retest_2026-11-02.json")
    a = ap.parse_args()
    if a.out.exists():
        sys.exit(f"refusing to overwrite {a.out}")
    rkey = {r["item"]: r["edge_id"] for r in read(RT / "stratum_pass2_retest_key.csv")}
    fkey = {r["item"]: r["edge_id"] for r in read(V2 / "B_stratum_pass2_key.csv")}
    retest = {rkey[r["item"]]: r for r in read(RT / "stratum_pass2_retest_coded.csv")}
    first = {fkey[r["item"]]: r for r in read(V2 / "received/B_stratum_pass2_as_submitted_2026-10-05_2223.csv")}
    assert set(retest) == set(first), "retest and first coding cover different edges"
    edges = sorted(retest)
    rng = random.Random(SEED)
    out = {"design": "same coder, v2 Pass 2 fields, stratum retest coded 2 Nov 2026 12:00-14:10 vs first coding (5 Oct 2026 22:23)",
           "fields": {}}
    for f in FIELDS:
        pairs_all = [(norm(f, first[e]), norm(f, retest[e])) for e in edges]
        if f == "uptake_degree":
            pairs_all = [(x, y) for x, y in pairs_all if x and y]
        pairs = [(x, y) for x, y in pairs_all if "unclear" not in (x, y)]
        rec = {"primary_unclear_excluded": summary(pairs, rng), "all_items": summary(pairs_all, rng),
               "unclear": {"first": sum(x == "unclear" for x, _ in pairs_all),
                           "retest": sum(y == "unclear" for _, y in pairs_all)},
               "changes": dict(Counter(f"{x} -> {y}" for x, y in pairs_all if x != y).most_common())}
        if f == "strongest_citance":
            multi = [(norm(f, first[e]), norm(f, retest[e])) for e in edges if int(retest[e]["n_citances"]) > 1]
            rec = {"all_items": rec["all_items"], "multi_citance_items": summary(multi, rng)}
        else:
            rec["first_distribution"] = dict(Counter(x for x, _ in pairs_all))
            rec["retest_distribution"] = dict(Counter(y for _, y in pairs_all))
        out["fields"][f] = rec
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
