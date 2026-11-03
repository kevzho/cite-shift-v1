#!/usr/bin/env python3
"""EXPLORATORY: MultiCite uptake at scale, CHI (all edges) vs the Nature arm.

The Nature arm FAILED its registered parity gate (docs/results/2026-10-06_nature_arm_parity.md), so
RQ4(b) is "not run"; this is exploratory only. The parser-bias check
(docs/results/2026-10-06_nature_parser_bias.md) bounds GROBID's effect on MultiCite uptake at about
-1.7 points [-2.5, -0.8]: differences smaller than ~2-3 points are not interpretable.

Edge rule = the registered RQ4(a) MultiCite construct (26_convert_system_outputs.py): drop citances
whose target marker is ambiguous/absent; edge prob = max over kept citances; claim = highest-probability
eligible label with sigmoid > 0.5 (uses/extends/motivation/differences -> uptake; background -> none;
similarities / future_work make no claim). Uptake share = uptake / (edges with a claim).
Bootstrap over citing papers (2,000 resamples).
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
D = ROOT / "analysis/results/external_systems/aclarc_multicite"
MC = {"uses": "method|artifact", "extends": "extension", "motivation": "framing",
      "differences": "contrast", "background": "none"}
GROUPS = {"chi": ["chi"], "nature_arm": ["nature_arm"]}


def load(corpus):
    paper = {}
    venue = {}
    for r in csv.DictReader(open(D / f"at_scale/input_{corpus}.csv", encoding="utf-8")):
        paper[r["edge_id"]] = r["citing_paper_id"]
        venue[r["edge_id"]] = r["venue"]
    by_paper = defaultdict(list)
    counts = Counter()
    for line in open(D / f"outputs/multicite_scibert_at_scale_{corpus}.jsonl", encoding="utf-8"):
        r = json.loads(line)
        counts["edges"] += 1
        keep = [c["probs"] for c, res in zip(r["per_citance"], r["marker_resolution"])
                if res in ("single_group", "intersection")]
        if not keep:
            counts["excluded_no_marker"] += 1
            continue
        p = {k: max(c[k] for c in keep) for k in keep[0]}
        elig = [(v, k) for k, v in p.items() if k in MC and v > 0.5]
        if not elig:
            counts["no_claim"] += 1
            continue
        lab = MC[max(elig)[1]]
        counts[lab] += 1
        by_paper[(venue[r["edge_id"]], paper[r["edge_id"]])].append(lab != "none")
    return by_paper, counts


def share(groups):
    n = sum(len(v) for v in groups)
    return sum(sum(v) for v in groups) / n if n else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20261003)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    data, out = {}, {"status": "EXPLORATORY - Nature arm failed its registered parity gate; RQ4(b) not run",
                     "parser_bias_bound_uptake_points": [-2.5, -0.8], "corpora": {}}
    for c in ("chi", "nature_arm"):
        bp, counts = load(c)
        data[c] = bp
        out["corpora"][c] = {"counts": dict(counts), "citing_papers": len(bp),
                             "uptake_share": round(share(list(bp.values())), 4)}
        if c == "chi":
            for v in ("chi", "chiea"):
                sub = [x for (ven, _), x in bp.items() if ven == v]
                out["corpora"][c][f"uptake_share_{v}"] = round(share(sub), 4)
    rng = random.Random(a.seed)
    keys = {c: list(data[c].values()) for c in data}
    diffs = []
    for _ in range(a.n_boot):
        s = {c: share(rng.choices(keys[c], k=len(keys[c]))) for c in keys}
        diffs.append(s["chi"] - s["nature_arm"])
    diffs.sort()
    d = out["corpora"]["chi"]["uptake_share"] - out["corpora"]["nature_arm"]["uptake_share"]
    out["delta_chi_minus_nature"] = {"delta": round(d, 4),
                                     "ci95": [round(diffs[int(.025 * len(diffs))], 4),
                                              round(diffs[int(.975 * len(diffs)) - 1], 4)]}
    if a.out.exists():
        raise SystemExit(f"refusing to overwrite {a.out}")
    a.out.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
