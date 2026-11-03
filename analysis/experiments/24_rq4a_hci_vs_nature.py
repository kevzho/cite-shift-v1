#!/usr/bin/env python3
"""RQ4(a), registered: HCI - Nature OA difference in "realized" rate, per system and per construct,
on the canonical labelled edges, with cluster-bootstrap CIs (components from analysis/ml/splits).

Report the SPREAD of the difference across systems and constructs, never one headline rate
(pre-registration §7). Constructs per source:
  canonical  realized = realizes_contribution in {partial, yes}
  <system>   credits  = credits_contribution == specific
             uptake   = uptake_mode not in {none, unclear}
             claim    = claim_dependent == yes
`unclear` is excluded field by field (count reported). Systems are added as their outputs exist.

--extra takes long-format files (edge_id, field, label, system, corpus) from
26_convert_system_outputs.py: field uptake_mode -> construct `uptake` (label != none; MultiCite's
merged "method|artifact" counts as uptake), field claim_dependent -> construct `claim`. Edges a system
makes no claim on are absent, i.e. missing. Added 11 Oct (final lineup): field credits_contribution -> `credits`
(label == specific); field realization (no/partial/yes) -> `realized`, as canonical (30_rq4a_final_inputs.py).
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=20261003)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--extra", type=Path, nargs="*", default=[])
    a = ap.parse_args()

    comp, series = {}, defaultdict(dict)   # series[(source, construct)][edge] = (corpus, bool)
    for c in ("hci", "nature_oa"):
        for s in csv.DictReader(open(ROOT / f"analysis/ml/splits/{c}_splits.csv", encoding="utf-8")):
            comp[s["edge_id_v1"]] = f"{c}:{s['component_key']}"
        for r in csv.DictReader(open(ROOT / f"analysis/derived/{c}_contribution_coding_human_reviewed.csv",
                                     encoding="utf-8-sig")):
            e = edge_id(c, r["cited_id"], r["citing_id"])
            v = r["realizes_contribution"].strip()
            if v in ("no", "partial", "yes"):
                series[("canonical", "realized")][e] = (c, v in ("partial", "yes"))
    for name, path in SYSTEMS.items():
        for r in csv.DictReader(open(path, encoding="utf-8")):
            e, c = r["edge_id"], r["corpus"]
            if r["credits_contribution"] in ("specific", "generic"):
                series[(name, "credits")][e] = (c, r["credits_contribution"] == "specific")
            if r["uptake_mode"] and r["uptake_mode"] != "unclear":
                series[(name, "uptake")][e] = (c, r["uptake_mode"] != "none")
            if r["claim_dependent"] in ("yes", "no"):
                series[(name, "claim")][e] = (c, r["claim_dependent"] == "yes")

    for path in a.extra:
        for r in csv.DictReader(open(path, encoding="utf-8")):
            if r["field"] == "uptake_mode" and r["label"] != "unclear":
                series[(r["system"], "uptake")][r["edge_id"]] = (r["corpus"], r["label"] != "none")
            elif r["field"] == "claim_dependent" and r["label"] in ("yes", "no"):
                series[(r["system"], "claim")][r["edge_id"]] = (r["corpus"], r["label"] == "yes")
            elif r["field"] == "credits_contribution" and r["label"] in ("specific", "generic"):
                series[(r["system"], "credits")][r["edge_id"]] = (r["corpus"], r["label"] == "specific")
            elif r["field"] == "realization" and r["label"] in ("no", "partial", "yes"):
                series[(r["system"], "realized")][r["edge_id"]] = (r["corpus"], r["label"] in ("partial", "yes"))

    def delta(items):
        h = [v for c, v in items if c == "hci"]
        n = [v for c, v in items if c == "nature_oa"]
        return (sum(h) / len(h) - sum(n) / len(n)) if h and n else None, h, n

    out = {"analysis": "RQ4(a) registered; Δ = HCI rate − Nature OA rate", "results": []}
    for (src, con), d in sorted(series.items()):
        by = defaultdict(list)
        for e, cv in d.items():
            by[comp.get(e, e)].append(cv)
        keys = sorted(by)
        rng = random.Random(a.seed)
        boots = []
        for _ in range(a.n_boot):
            x = delta([cv for k in rng.choices(keys, k=len(keys)) for cv in by[k]])[0]
            if x is not None:
                boots.append(x)
        boots.sort()
        dlt, h, n = delta(list(d.values()))
        lo, hi = boots[int(0.025 * len(boots))], boots[int(0.975 * len(boots)) - 1]
        out["results"].append({"source": src, "construct": con, "n_hci": len(h), "n_nature": len(n),
                               "rate_hci": round(sum(h) / len(h), 3), "rate_nature": round(sum(n) / len(n), 3),
                               "delta": round(dlt, 3), "ci95": [round(lo, 3), round(hi, 3)],
                               "sign": "HCI>Nature" if lo > 0 else "Nature>HCI" if hi < 0 else "n.s."})
    ds = [r["delta"] for r in out["results"]]
    out["spread"] = {"min_delta": min(ds), "max_delta": max(ds),
                     "signs": sorted({r["sign"] for r in out["results"]})}
    if a.out.exists():
        raise SystemExit(f"refusing to overwrite {a.out}")
    a.out.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"{'source':14s}{'construct':10s}{'HCI':>7s}{'Nature':>8s}{'Δ':>8s}  95% CI           sign")
    for r in out["results"]:
        print(f"{r['source']:14s}{r['construct']:10s}{r['rate_hci']:7.3f}{r['rate_nature']:8.3f}{r['delta']:8.3f}"
              f"  [{r['ci95'][0]:+.3f}, {r['ci95'][1]:+.3f}]  {r['sign']}")
    print("spread:", out["spread"])


if __name__ == "__main__":
    main()
