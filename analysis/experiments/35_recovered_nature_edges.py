#!/usr/bin/env python3
"""Exploratory recovered-edge sensitivity analysis; never reinstates RQ4(b).

Use only the 200 Nature parity papers, and select a JATS edge if the frozen
parity audit located its GROBID edge and recovered every JATS citance.
MultiCite outputs already exist for JATS, GROBID, and CHI. Uncertainty is
resampled by citing paper. The comparison with CHI is descriptive because
the Nature arm failed the registered gate and the corpora are unmatched.
"""
from __future__ import annotations

import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from importlib.machinery import SourceFileLoader

ROOT = Path(__file__).resolve().parents[2]
NATURE = ROOT / "external_data/nature_arm_2026-10-04"
OUTPUTS = ROOT / "analysis/results/external_systems/aclarc_multicite/outputs"
OUT = ROOT / "analysis/results/exploratory/2026-10-12_recovered_nature_edges.json"
SCALE = SourceFileLoader("scale_summary", str(Path(__file__).with_name("28_multicite_at_scale_summary.py"))).load_module()
SEED, N_BOOT = 20261012, 2000
ELIGIBLE = {"uses", "extends", "motivation", "differences", "background"}


def classify(record):
    keep = [c["probs"] for c, resolution in zip(record["per_citance"], record["marker_resolution"])
            if resolution in ("single_group", "intersection")]
    if not keep:
        return "excluded_marker"
    probs = {label: max(c[label] for c in keep) for label in keep[0]}
    eligible = [(value, label) for label, value in probs.items() if label in ELIGIBLE and value > .5]
    if not eligible:
        return "no_claim"
    return "none" if max(eligible)[1] == "background" else "uptake"


def load_parity(version):
    path = OUTPUTS / f"multicite_scibert_nature_parity_{version}.jsonl"
    return {r["edge_id"]: classify(r) for r in map(json.loads, path.open())}


def tally(by_paper, papers=None):
    keys = list(by_paper) if papers is None else papers
    counts = Counter()
    for paper in keys:
        counts.update(by_paper.get(paper, ()))
    claims = counts["none"] + counts["uptake"]
    return {"edges": sum(counts.values()), "papers": len(set(keys)), "excluded_marker": counts["excluded_marker"],
            "no_claim": counts["no_claim"], "claims": claims, "uptake": counts["uptake"],
            "uptake_share": counts["uptake"] / claims if claims else None}


def ci(values):
    values = sorted(values)
    return [round(values[int(.025 * len(values))], 4), round(values[int(.975 * len(values)) - 1], 4)]


def main():
    audit = list(csv.DictReader((NATURE / "parity/parity_gate_edges.csv").open()))
    jats = load_parity("jats")
    grobid = load_parity("grobid")
    all_papers = sorted({row["pmcid"] for row in audit})
    candidates = [row for row in audit if row["fully_matched"] == "1" and row["located"] != "none"]
    grobid_targets = Counter((row["pmcid"], row["grobid_ref"]) for row in candidates)
    selected = [row for row in candidates if grobid_targets[(row["pmcid"], row["grobid_ref"])] == 1]
    selected_keys = {(row["pmcid"], row["jats_ref"]) for row in selected}
    selected_papers = {row["pmcid"] for row in selected}
    ja, js, gs, ga = (defaultdict(list) for _ in range(4))
    missing_predictions = Counter()
    for row in audit:
        p = row["pmcid"]
        jk = f"jats:{p}:{row['jats_ref']}"
        gk = f"grobid:{p}:{row['grobid_ref']}"
        if jk in jats:
            ja[p].append(jats[jk])
        else:
            missing_predictions["jats_all"] += 1
        if row["located"] != "none" and gk in grobid:
            ga[p].append(grobid[gk])
        if (p, row["jats_ref"]) in selected_keys:
            if jk in jats and gk in grobid:
                js[p].append(jats[jk])
                gs[p].append(grobid[gk])
            else:
                missing_predictions["selected"] += 1
    # Existing CHI output and frozen edge-to-paper map; same classifier rule.
    chi_by_venue_paper, chi_counts = SCALE.load("chi")
    chi = {f"{venue}:{paper}": ["uptake" if v else "none" for v in values]
           for (venue, paper), values in chi_by_venue_paper.items()}
    chi_papers = sorted(chi)
    point = {"nature_jats_all": tally(ja, all_papers),
             "nature_jats_recovered": tally(js, all_papers),
             "nature_grobid_recovered": tally(gs, all_papers),
             "nature_grobid_located": tally(ga, all_papers),
             "chi_claims_only": tally(chi, chi_papers)}
    rng = random.Random(SEED)
    diffs = defaultdict(list)
    for _ in range(N_BOOT):
        nature_sample = rng.choices(all_papers, k=len(all_papers))
        chi_sample = rng.choices(chi_papers, k=len(chi_papers))
        rates = {"ja": tally(ja, nature_sample)["uptake_share"],
                 "js": tally(js, nature_sample)["uptake_share"],
                 "gs": tally(gs, nature_sample)["uptake_share"],
                 "chi": tally(chi, chi_sample)["uptake_share"]}
        if all(value is not None for value in rates.values()):
            diffs["selected_minus_all_jats"].append(rates["js"] - rates["ja"])
            diffs["grobid_minus_jats_selected"].append(rates["gs"] - rates["js"])
            diffs["chi_minus_nature_recovered"].append(rates["chi"] - rates["gs"])
    d = {"selected_minus_all_jats": point["nature_jats_recovered"]["uptake_share"] - point["nature_jats_all"]["uptake_share"],
         "grobid_minus_jats_selected": point["nature_grobid_recovered"]["uptake_share"] - point["nature_jats_recovered"]["uptake_share"],
         "chi_minus_nature_recovered": point["chi_claims_only"]["uptake_share"] - point["nature_grobid_recovered"]["uptake_share"]}
    result = {"status": "EXPLORATORY; registered RQ4(b) remains not run",
              "selection": "parity row fully_matched=1 and located!=none; all JATS citances recovered; unique GROBID edge mapping",
              "bootstrap": {"unit": "citing paper", "resamples": N_BOOT, "seed": SEED},
              "audit": {"papers": len(all_papers), "jats_edges": len(audit),
                        "selected_edges": len(selected), "excluded_edges": len(audit) - len(selected),
                        "ambiguous_duplicate_mapping_edges_excluded": len(candidates) - len(selected),
                        "selected_papers": len(selected_papers), "missing_predictions": dict(missing_predictions)},
              "rates": point, "contrasts": {key: {"difference": round(value, 4), "ci95": ci(diffs[key])} for key, value in d.items()},
              "limitations": ["Nature subset is the 200-paper parity sample, not all 1,200 Nature papers.",
                              "Selection by successful recovery can change the distribution of citation roles.",
                              "CHI and Nature papers are unmatched; this is not a field-causal estimate.",
                              "The CHI uptake rate is calculated among edges with a claim, matching the existing at-scale estimand; excluded-marker and no-claim CHI edges are outside that denominator."]}
    if OUT.exists():
        raise SystemExit(f"refusing to overwrite {OUT}")
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
