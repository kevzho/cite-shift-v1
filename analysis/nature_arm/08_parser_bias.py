#!/usr/bin/env python3
"""EXPLORATORY parser-bias check for the Nature arm (6 Oct 2026). The arm FAILED its registered parity
gate (citance recovery 86.1%, edge match 82.5%); this script does not change that.

Question: does GROBID's loss of citances (vs the publisher JATS of the same 200 parity papers) change what
citation classifiers say? SciCite and MultiCite were run on both versions (code/run_parser_bias_systems.sh).

Compared, per system:
  (a) citance-level label distribution, JATS vs GROBID (all citances);
  (b) edge-level label distribution, using the project's edge rules:
        SciCite   edge_label = argmax of mean citance probability (harness rule, 26 Sep)
        MultiCite RQ4(a) rule (analysis/experiments/26_convert_system_outputs.py): citances with ambiguous or
                  no target marker dropped; highest-probability eligible label > 0.5 among uses/extends/
                  motivation/differences/background; uptake = any label other than background ("none");
  (c) on PAIRED edges only (parity/parity_gate_edges.csv, located != none): per-edge agreement and kappa.
Uncertainty: bootstrap over papers (paired: both versions of a resampled paper go in together), 2,000
resamples, seed 20261003, percentile 95% CIs, for every share difference (GROBID - JATS) and for the
total-variation distance. Decision aid only: compares shifts with ~0.035, the CI half-width of the RQ4(a)
HCI-Nature differences.
"""
from __future__ import annotations

import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

N = Path(__import__("os").environ.get("NATURE_ARM_DIR", Path(__file__).resolve().parents[2] / "external_data/nature_arm_2026-10-04"))
ROOT = Path(__import__("os").environ.get("CITATION_ROOT", Path(__file__).resolve().parents[2]))
RES = ROOT / "analysis/results/external_systems"
OUT = N / "parser_bias"
SEED, NB, BAR = 20261003, 2000, 0.035
MC = {"uses": "method|artifact", "extends": "extension", "motivation": "framing",
      "differences": "contrast", "background": "none"}
VERSIONS = ("jats", "grobid")


def paper(edge_id):
    return edge_id.split(":")[1]


def load_multicite(v):
    cit, edge = [], {}
    for line in open(RES / f"aclarc_multicite/outputs/multicite_scibert_nature_parity_{v}.jsonl"):
        r = json.loads(line)
        p = paper(r["edge_id"])
        for c, res in zip(r["per_citance"], r["marker_resolution"]):
            cit.append((p, tuple(sorted(c["labels"])), res))
        keep = [c["probs"] for c, res in zip(r["per_citance"], r["marker_resolution"])
                if res in ("single_group", "intersection")]
        if not keep:
            edge[r["edge_id"]] = (p, "excluded_marker")
            continue
        mx = {k: max(c[k] for c in keep) for k in keep[0]}
        elig = [(val, k) for k, val in mx.items() if k in MC and val > 0.5]
        edge[r["edge_id"]] = (p, MC[max(elig)[1]] if elig else "no_claim")
    return cit, edge


def load_scicite(v):
    cit, edge = [], {}
    base = RES / "scicite_local/outputs"
    eid = {}
    for r in csv.DictReader(open(base / f"scicite_edge_preds_nature_parity_{v}.csv")):
        edge[r["edge_id"]] = (paper(r["edge_id"]), r["edge_label"])
    for r in csv.DictReader(open(base / f"scicite_citance_preds_nature_parity_{v}.csv")):
        cit.append((paper(r["edge_id"]), r["model_prediction"]))
    return cit, edge


def shares(items, classes):
    n = len(items)
    c = Counter(items)
    return {k: (c[k] / n if n else 0.0) for k in classes}, n


def boot_compare(by_paper, classes, positive=None):
    """by_paper[version][paper] -> list of category labels (one per unit).
    positive: optional set of labels whose union share is also compared (e.g. uptake)."""
    papers = sorted(set(by_paper["jats"]) | set(by_paper["grobid"]))

    def stat(ps):
        res = {}
        for v in VERSIONS:
            items = [x for p in ps for x in by_paper[v].get(p, [])]
            s, n = shares(items, classes)
            if positive is not None:
                s["__pos__"] = sum(s[k] for k in classes if k in positive)
            res[v] = (s, n)
        d = {k: res["grobid"][0][k] - res["jats"][0][k] for k in res["jats"][0]}
        tv = 0.5 * sum(abs(res["grobid"][0][k] - res["jats"][0][k]) for k in classes)
        return res, d, tv

    point, dpt, tvpt = stat(papers)
    rng = random.Random(SEED)
    bd, btv = defaultdict(list), []
    for _ in range(NB):
        ps = rng.choices(papers, k=len(papers))
        _, d, tv = stat(ps)
        for k, x in d.items():
            bd[k].append(x)
        btv.append(tv)

    def ci(xs):
        xs = sorted(xs)
        return [round(xs[int(0.025 * len(xs))], 4), round(xs[int(0.975 * len(xs)) - 1], 4)]

    out = {"n_jats": point["jats"][1], "n_grobid": point["grobid"][1],
           "classes": {k: {"jats": round(point["jats"][0][k], 4), "grobid": round(point["grobid"][0][k], 4),
                           "diff_grobid_minus_jats": round(dpt[k], 4), "ci95": ci(bd[k])}
                       for k in point["jats"][0]},
           "total_variation": {"value": round(tvpt, 4), "ci95": ci(btv)}}
    if "__pos__" in out["classes"]:
        out["positive_share"] = out["classes"].pop("__pos__")
    return out


def kappa(pairs):
    n = len(pairs)
    if not n:
        return None
    po = sum(a == b for a, b in pairs) / n
    ca, cb = Counter(a for a, _ in pairs), Counter(b for _, b in pairs)
    pe = sum(ca[k] * cb[k] for k in set(ca) | set(cb)) / n / n
    return round((po - pe) / (1 - pe), 4) if pe < 1 else None


def main():
    pairs = [r for r in csv.DictReader(open(N / "parity/parity_gate_edges.csv")) if r["located"] != "none"]
    result = {"status": "EXPLORATORY - the Nature arm failed its registered gate; this does not change that",
              "papers": 200, "bootstrap": {"unit": "paper", "resamples": NB, "seed": SEED},
              "paired_edges": len(pairs), "systems": {}}

    # ---- MultiCite
    cit, edge = {}, {}
    for v in VERSIONS:
        cit[v], edge[v] = load_multicite(v)
    mc_labels = sorted({l for v in VERSIONS for _, labs, _ in cit[v] for l in labs})
    # citance level: multi-label -> share of citances carrying each label (one binary unit per label)
    mc_cit = {}
    for lab in mc_labels:
        bp = {v: defaultdict(list) for v in VERSIONS}
        for v in VERSIONS:
            for p, labs, _ in cit[v]:
                bp[v][p].append("yes" if lab in labs else "no")
        mc_cit[lab] = boot_compare(bp, ["yes", "no"])["classes"]["yes"]
    marker_res = {v: dict(Counter(res for _, _, res in cit[v])) for v in VERSIONS}
    classes = sorted({lab for v in VERSIONS for _, lab in edge[v].values()})
    bp = {v: defaultdict(list) for v in VERSIONS}
    for v in VERSIONS:
        for p, lab in edge[v].values():
            bp[v][p].append(lab)
    mc_edge_all = boot_compare(bp, classes)
    # uptake share among edges WITH a claim (the RQ4(a) construct)
    claim_cls = [c for c in classes if c not in ("excluded_marker", "no_claim")]
    bpc = {v: {p: [x for x in xs if x in claim_cls] for p, xs in bp[v].items()} for v in VERSIONS}
    mc_uptake = boot_compare(bpc, claim_cls, positive={c for c in claim_cls if c != "none"})
    pe = [(edge["jats"].get(f"jats:{r['pmcid']}:{r['jats_ref']}"), edge["grobid"].get(f"grobid:{r['pmcid']}:{r['grobid_ref']}"))
          for r in pairs]
    pe = [(a[1], b[1]) for a, b in pe if a and b]
    pe_claim = [(a, b) for a, b in pe if a in claim_cls and b in claim_cls]
    up = [(a != "none", b != "none") for a, b in pe_claim]
    result["systems"]["multicite"] = {
        "marker_resolution": marker_res,
        "citance_level_label_share": mc_cit,
        "edge_level_all_categories": mc_edge_all,
        "edge_level_uptake_among_claims": mc_uptake,
        "paired_edges": {"n_both_present": len(pe), "agreement_all_categories": round(sum(a == b for a, b in pe) / len(pe), 4),
                         "kappa_all_categories": kappa(pe), "n_both_claim": len(pe_claim),
                         "agreement_label_both_claim": round(sum(a == b for a, b in pe_claim) / len(pe_claim), 4) if pe_claim else None,
                         "kappa_label_both_claim": kappa(pe_claim),
                         "agreement_uptake_both_claim": round(sum(a == b for a, b in up) / len(up), 4) if up else None,
                         "kappa_uptake_both_claim": kappa(up)},
    }

    # ---- SciCite
    cit, edge = {}, {}
    for v in VERSIONS:
        cit[v], edge[v] = load_scicite(v)
    sc_cls = ["background", "method", "result"]
    bpc = {v: defaultdict(list) for v in VERSIONS}
    bpe = {v: defaultdict(list) for v in VERSIONS}
    for v in VERSIONS:
        for p, lab in cit[v]:
            bpc[v][p].append(lab)
        for p, lab in edge[v].values():
            bpe[v][p].append(lab)
    pe = [(edge["jats"].get(f"jats:{r['pmcid']}:{r['jats_ref']}"), edge["grobid"].get(f"grobid:{r['pmcid']}:{r['grobid_ref']}"))
          for r in pairs]
    pe = [(a[1], b[1]) for a, b in pe if a and b]
    result["systems"]["scicite"] = {
        "citance_level": boot_compare(bpc, sc_cls),
        "edge_level": boot_compare(bpe, sc_cls, positive={"method"}),
        "paired_edges": {"n_both_present": len(pe), "agreement": round(sum(a == b for a, b in pe) / len(pe), 4),
                         "kappa": kappa(pe),
                         "method_agreement": round(sum((a == "method") == (b == "method") for a, b in pe) / len(pe), 4)},
    }

    # ---- decision aid
    shifts = []
    for name, block in (("multicite edge uptake share", result["systems"]["multicite"]["edge_level_uptake_among_claims"]["positive_share"]),
                        ("scicite edge method share", result["systems"]["scicite"]["edge_level"]["positive_share"])):
        shifts.append((name, block["diff_grobid_minus_jats"], block["ci95"]))
    for k, b in result["systems"]["scicite"]["edge_level"]["classes"].items():
        shifts.append((f"scicite edge {k}", b["diff_grobid_minus_jats"], b["ci95"]))
    for k, b in result["systems"]["multicite"]["edge_level_all_categories"]["classes"].items():
        shifts.append((f"multicite edge {k}", b["diff_grobid_minus_jats"], b["ci95"]))
    big = [s for s in shifts if abs(s[1]) > BAR]
    result["decision_aid"] = {
        "bar": BAR, "basis": "CI half-width of the RQ4(a) HCI-Nature differences",
        "shifts_over_bar": [{"what": n, "diff": d, "ci95": c} for n, d, c in big],
        "max_abs_shift": max(shifts, key=lambda s: abs(s[1])),
    }
    OUT.mkdir(exist_ok=True)
    path = OUT / "parser_bias_result.json"
    if path.exists():
        raise SystemExit(f"refusing to overwrite {path}")
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
