#!/usr/bin/env python3
"""Dated B300 companion evaluations for the manuscript revision authorized 23 Oct.

Uses frozen predictions only: no model calls or training. Keeps prior outputs intact.
The B300 reference includes the third stratum submission reported hand-coded by
the author. This scoring exercise makes no independent authorship determination.
Outputs: reanalysis_2026-10-23_B300_companions/.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "analysis/results"
OUT = RES / "reanalysis_2026-10-23_B300_companions"


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


C = module("b300_conf", "analysis/experiments/29_confirmatory_2026_10_08.py")
ORIGINAL_REFERENCES = C.references
THIRD = C.V2 / "received/B_stratum_pass2_as_submitted_2026-10-05_2223.csv"


def b300_references(cw):
    canonical, b, a, v1 = ORIGINAL_REFERENCES(cw)
    keys = {r["item"]: r["edge_id"] for r in C.read(C.V2 / "B_stratum_pass2_key.csv")}
    rows = C.read(THIRD)
    assert len(rows) == 60 and {r["item"] for r in rows} == set(keys)
    for row in rows:
        for field in C.PASS2:
            value = (row.get(field) or "").strip()
            if value and value != "not_applicable":
                b[field][keys[row["item"]]] = value
    assert len(b["uptake_mode"]) == 300
    return canonical, b, a, v1


C.references = b300_references


def write(path, data):
    if path.exists():
        raise SystemExit(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def calibration():
    path = OUT / "calibration.json"
    if path.exists():
        return
    M = module("b300_calibration", "analysis/experiments/32_calibration.py")
    M.C = C
    cw = C.S.crosswalk()
    alias, cluster = C.S.components()
    _, raw, _, _ = C.references(cw)
    ref, dropped = C.drop_unclear(raw)
    b1, b2 = sorted(raw["citation_function"]), sorted(raw["uptake_mode"])
    boot = C.Boot(b1, cluster)
    probs = M.load_probs(cw, alias)
    results = {}
    for system, seeds in probs.items():
        for field in M.FIELDS:
            if any(field in fields for fields in seeds.values()):
                value = M.evaluate(seeds, ref[field], b1 if field in C.PASS1 else b2, field, boot)
                if value:
                    results[f"kevin_v2@B|{field}|{system}"] = value
    # The D3 reference is entirely Pass 1 and must remain unchanged.
    prior = json.loads((RES / "calibration_2026-10-12/calibration.json").read_text())
    for key, value in results.items():
        if key.split("|")[1] in C.PASS1:
            assert value == prior["results"][key], key
    write(path, {"date": "2026-10-23", "reference": "B300", "bins": M.BINS,
                 "n_boot": C.N_BOOT, "seed": C.SEED, "unclear_excluded_B": dropped,
                 "results": results})
    print("wrote B300 calibration", flush=True)


def acl_arc():
    path = OUT / "aclarc.json"
    if path.exists():
        return
    rows = C.read(RES / "external_systems/scoring_inputs_aclarc_20261021/aclarc_uptake_exploratory.csv")
    cw = C.S.crosswalk()
    alias, cluster = C.S.components()
    _, raw, _, _ = C.references(cw)
    ref, _ = C.drop_unclear(raw)
    edges = sorted(raw["uptake_mode"])
    boot = C.Boot(sorted(raw["citation_function"]), cluster)
    pred = {alias.get(r["edge_id"], r["edge_id"]): r["label"] for r in rows}
    scores = {}
    for variant in ("primary", "sensitivity_all"):
        pairs = []
        for e in edges:
            if e in ref["uptake_mode"] and e in pred:
                pair = C.eligible_pair("multicite", "uptake_mode", ref["uptake_mode"][e], pred[e], variant)
                if pair:
                    pairs.append((e, *pair))
        m = C.macro_f1(pairs, boot)
        scores[variant] = {"n": m["n"], "macro_f1": C.r4(m["macro_f1"]),
                           "ci95": C.ci(m["boot_f1"]), "kappa": C.r4(m["kappa"]), "withheld": m["withheld"]}
    recovery = {}
    for target, field in (("credits_specific", "credits_contribution"), ("uptake", "uptake_mode"),
                          ("claim_dependent_yes", "claim_dependent")):
        positive = "specific" if target == "credits_specific" else "yes"
        pairs = [(e, pred[e] != "none", ref[field][e] != "none" if target == "uptake" else ref[field][e] == positive)
                 for e in edges if e in pred and e in ref[field]]
        (kap, nmi), _ = C.binary_stats(pairs, boot)
        recovery[target] = {"n": len(pairs), "kappa": C.r4(kap), "nmi_geometric": C.r4(nmi)}
    write(path, {"date": "2026-10-23", "status": "exploratory", "reference": "B300",
                 "uptake_mode": scores, "construct_recovery_B": recovery,
                 "limitations": "No target-citation marker; system-specific expressible subset."})
    print("wrote B300 ACL-ARC", flush=True)


def d2():
    path = OUT / "d2_scores.json"
    if path.exists():
        return
    M = module("b300_d2", "analysis/experiments/31_d2_jepa_scoring.py")
    M.C = C
    cw = C.S.crosswalk()
    alias, cluster = C.S.components()
    _, raw, _, _ = C.references(cw)
    ref, _ = C.drop_unclear(raw)
    b1, b2 = sorted(raw["citation_function"]), sorted(raw["uptake_mode"])
    boot = C.Boot(b1, cluster)
    preds = M.load(cw, alias)
    scores, boots = {}, {}
    for field in M.FIELDS:
        for arm in M.ARMS:
            per = [m for m in M.score(ref[field], b1 if field in C.PASS1 else b2, boot, preds[arm], field) if m]
            bf1 = np.nanmean(np.stack([m["boot_f1"] for m in per]), axis=0)
            boots[(field, arm)] = bf1
            scores[f"kevin_v2@B|{field}|{arm}"] = {
                "n": per[0]["n"], "macro_f1_seed_mean": C.r4(np.mean([m["macro_f1"] for m in per])),
                "macro_f1_per_seed": [C.r4(m["macro_f1"]) for m in per], "ci95": C.ci(bf1),
                "kappa_seed_mean": C.r4(np.mean([m["kappa"] for m in per])), "withheld": per[0]["withheld"]}
    differences = {}
    for field in M.FIELDS:
        for left, right in (("jepa", "mlm"), ("jepa", "none"), ("mlm", "none")):
            key = lambda arm: f"kevin_v2@B|{field}|{arm}"
            differences[f"kevin_v2@B|{field}|{left}-{right}"] = {
                "diff": C.r4(scores[key(left)]["macro_f1_seed_mean"] - scores[key(right)]["macro_f1_seed_mean"]),
                "ci95": C.ci(boots[(field, left)] - boots[(field, right)])}
    prior = json.loads((RES / "ml/d2_jepa/scoring_2026-10-11/d2_scores.json").read_text())
    for key, value in scores.items():
        if key.split("|")[1] in C.PASS1:
            assert value == prior["scores"][key], key
    write(path, {"date": "2026-10-23", "status": "exploratory; B300 re-evaluation",
                 "n_boot": C.N_BOOT, "seed": C.SEED, "scores": scores, "differences": differences})
    print("wrote B300 JEPA/MLM scoring", flush=True)


def qwen():
    target = OUT / "with_qwen"
    if (target / "rq3.json").exists():
        return
    original_load = C.load_systems
    Q = module("b300_qwen", "analysis/experiments/33_qwen_addendum.py")

    def load(cw, alias):
        systems = original_load(cw, alias)
        rows = Q.qwen_rows("v2edges")
        for field in Q.FIELDS:
            systems[Q.QWEN][0][field].update(Q.labels(rows, field, cw, alias))
        return systems

    C.load_systems = load
    C.OUT = target
    C.main()
    C.load_systems = original_load
    for name in ("rq1b", "rq2", "d1"):
        actual = json.loads((target / f"{name}.json").read_text())
        prior = json.loads((RES / "sensitivity_2026-10-23_B300" / f"{name}.json").read_text())
        assert actual == prior, name
    p = target / "rq1b.json"
    d = json.loads(p.read_text())
    d["B"]["sample"] = "B300 (300 Pass 2 records; unclear excluded; 23 Oct reanalysis)"
    p.write_text(json.dumps(d, indent=2) + "\n")
    print("verified B300 RQ1/RQ2/D1 reproduce exactly with Qwen added", flush=True)


def robustness():
    path = OUT / "robustness.json"
    if path.exists():
        return
    cw = C.S.crosswalk()
    alias, _ = C.S.components()
    canon, raw, _, _ = C.references(cw)
    ref, _ = C.drop_unclear(raw)
    systems = C.load_systems(cw, alias)
    degenerate = set()
    for reference, data in (("canonical@B", canon), ("kevin_v2@B", ref)):
        for field in C.PASS1 + C.PASS2:
            if field not in data:
                continue
            for system, seeds in systems.items():
                seed = min(seeds)
                pred = seeds[seed].get(field, {})
                support = Counter()
                edges = sorted(raw["citation_function"] if field in C.PASS1 else raw["uptake_mode"])
                if reference == "canonical@B" and field in C.PASS2:
                    continue
                for e in edges:
                    if e in data[field] and e in pred:
                        if field == "uptake_degree" and pred[e] not in ("full", "partial"):
                            continue
                        pair = C.eligible_pair(system, field, data[field][e], pred[e], "primary")
                        if pair:
                            support[pair[0]] += 1
                if sum(n >= 5 for n in support.values()) == 1:
                    degenerate.add((reference, field, system))
    base = json.loads((RES / "sensitivity_2026-10-23_B300/rq3.json").read_text())
    refs = [x for x in base["flip_probabilities_between_references"]
            if not any((r, x["field"], s) in degenerate for r in ("canonical@B", "kevin_v2@B") for s in x["pair"])]
    fields = [x for x in base["flip_probabilities_between_fields"]
              if not any(("kevin_v2@B", f, s) in degenerate for f in x["fields"] for s in x["pair"])]
    write(path, {"reference": "B300", "lineup": "October 8", "degenerate_cells_one_reportable_class": sorted(degenerate),
                 "reference_flips": {"count": sum(x["p_flip"] > .5 for x in refs), "pairs": len(refs)},
                 "field_flips": {"count": sum(x["p_flip"] > .5 for x in fields), "pairs": len(fields)}})


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    # Refuse to silently reuse another source file on a resumed run.
    manifest = {"date": "2026-10-23", "third_submission_sha256": hashlib.sha256(THIRD.read_bytes()).hexdigest(),
                "reason": "Author-authorized full-B manuscript presentation after viewing the B300 results.",
                "original_outputs": "sensitivity_2026-10-23_B300", "n_boot": C.N_BOOT, "seed": C.SEED,
                "authorship": "Author reports hand coding; scoring does not independently determine authorship."}
    mp = OUT / "manifest.json"
    if mp.exists():
        assert json.loads(mp.read_text()) == manifest
    else:
        write(mp, manifest)
    qwen()
    calibration()
    acl_arc()
    d2()
    robustness()
    print("Completed companion evaluations; all archived outputs preserved.", flush=True)


if __name__ == "__main__":
    main()
