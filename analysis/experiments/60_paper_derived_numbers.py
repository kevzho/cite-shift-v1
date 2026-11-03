#!/usr/bin/env python3
"""Numbers reported in the iConference manuscript that were previously computed by hand or held only in notes
(27 Oct 2026). Each entry is recomputed from the frozen inputs or result files; nothing here is typed in.

    python3 analysis/experiments/60_paper_derived_numbers.py [--out PATH]

Default output: analysis/results/paper_numbers/derived_numbers.json (refuses to overwrite).
Reads private inputs under annotations/generated/ and external_data/ (see reproduce/README.md).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import subprocess
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "analysis/results"
V2 = ROOT / "annotations/generated/v2_coding_2026-10-02"
ND = ROOT / "external_data/nature_arm_2026-10-04"
ES = RES / "external_systems"
PASS2 = ("credits_contribution", "uptake_mode", "uptake_degree", "claim_dependent")


def read(p: Path):
    with p.open(newline="", encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def load(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))


def wilson(k: int, n: int, z: float = 1.959964):
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [round(c - h, 4), round(c + h, 4)]


def norm(v: str | None) -> str:
    v = (v or "").strip()
    return "" if v in ("", "not_applicable", "n/a", "na") else v


# ---------------------------------------------------------------- RQ1 rationale audit (3/50)
def rationale_audit():
    rows = read(ROOT / "annotations/generated/v2_prep/rationale_check_50.csv")
    k = sum(r["asserts_use"].strip().lower() == "yes" for r in rows)
    return {"n": len(rows), "asserts_use_yes": k, "share": round(k / len(rows), 4), "wilson95": wilson(k, len(rows))}


# ---------------------------------------------------------------- RQ3 flip counts
def flips():
    out = {}
    for tag, path in (("b300", RES / "sensitivity_2026-10-23_B300/rq3.json"),
                      ("b300_with_qwen", RES / "reanalysis_2026-10-23_B300_companions/with_qwen/rq3.json")):
        d = load(path)
        res = {}
        for kind in ("references", "fields"):
            pairs = d[f"flip_probabilities_between_{kind}"]
            no_d1 = [p for p in pairs if "d1_distilled" not in p["pair"]]
            res[kind] = {"flips": sum(p["p_flip"] > 0.5 for p in pairs), "pairs": len(pairs),
                         "flips_without_d1": sum(p["p_flip"] > 0.5 for p in no_d1), "pairs_without_d1": len(no_d1),
                         "ties_at_0.5": sum(p["p_flip"] == 0.5 for p in pairs)}
        out[tag] = res
    return out


# ---------------------------------------------------------------- Qwen pairing-rule failures (197/550)
def qwen_pairing():
    v = load(RES / "confirmatory_2026-10-12_with_qwen/qwen_addendum.json")["qwen_validity"]
    return {"edges": v["v2edges_n"], "degree_rule_ok": v["v2edges_degree_rule_ok"],
            "degree_rule_violated": v["v2edges_n"] - v["v2edges_degree_rule_ok"]}


# ---------------------------------------------------------------- B300 RQ1-eligible edges by corpus (147/150)
def b300_eligible_by_corpus():
    corpus, labels = {}, {}
    for part in ("B_stratum_pass2", "B_rest_pass2"):
        key = {r["item"]: r for r in read(V2 / f"{part}_key.csv")}
        src = (V2 / "received/B_stratum_pass2_as_submitted_2026-10-05_2223.csv") if part == "B_stratum_pass2" \
            else V2 / f"{part}.csv"
        for r in read(src):
            k = key[r["item"]]
            corpus[k["edge_id"]] = k["corpus"]
            labels[k["edge_id"]] = r
    ok = [e for e, r in labels.items()
          if norm(r["credits_contribution"]) not in ("", "unclear") and norm(r["uptake_mode"]) not in ("", "unclear")]
    return {"edges": len(labels), "eligible": len(ok), "eligible_by_corpus": dict(Counter(corpus[e] for e in ok))}


# ---------------------------------------------------------------- split overlap found on 29 Sep (281/61/61/97)
def split_overlap():
    exp = {r["edge_id"] for r in read(ROOT / "annotations/generated/expansion_2026-09-16/expansion_batch_kz.csv")}
    pre = ROOT / "analysis/ml/splits_pre_expansion_2026-09-29"
    best = None
    for idcol in ("edge_id_v1", "edge_id_v2"):
        where = {}
        for c in ("hci", "nature_oa"):
            for r in read(pre / f"{c}_splits.csv"):
                where[r[idcol]] = r["split"]
        counts = Counter(where.get(e, "not_in_splits") for e in exp)
        if best is None or sum(v for k, v in counts.items() if k != "not_in_splits") > \
                sum(v for k, v in best[1].items() if k != "not_in_splits"):
            best = (idcol, counts)
    return {"expansion_edges": len(exp), "id_column": best[0], "split_of_expansion_edges": dict(best[1])}


# ---------------------------------------------------------------- A250 seeded draw (seed 20261005)
def a250_draw():
    a500 = read(V2 / "A_pass2_key.csv")
    a250 = [r["item"] for r in read(V2 / "A250_pass2_key.csv")]
    tried = {}
    for name, draw in (
        ("per_corpus_sample_in_file_order", lambda rng: [r["item"] for c in ("hci", "nature_oa") for r in
                                                         sorted(rng.sample([x for x in a500 if x["corpus"] == c], 125),
                                                                key=a500.index)]),
        ("per_corpus_sample", lambda rng: [r["item"] for c in ("hci", "nature_oa") for r in
                                           rng.sample([x for x in a500 if x["corpus"] == c], 125)]),
    ):
        got = draw(random.Random(20261005))
        tried[name] = {"same_set": set(got) == set(a250), "same_order": got == a250}
    return {"a500": len(a500), "a250": len(a250), "by_corpus": dict(Counter(r["corpus"] for r in read(V2 / "A250_pass2_key.csv"))),
            "subset_of_a500": set(a250) <= {r["item"] for r in a500},
            "same_relative_order_as_a500": a250 == [r["item"] for r in a500 if r["item"] in set(a250)],
            "reconstruction_attempts_seed_20261005": tried}


# ---------------------------------------------------------------- stratum Pass 2 provenance overlap (52/60 vs 16-18/60)
def stratum_overlap():
    key = {r["item"]: r["edge_id"] for r in read(V2 / "B_stratum_pass2_key.csv")}
    rec = V2 / "received"
    files = {"third_submission": rec / "B_stratum_pass2_as_submitted_2026-10-05_2223.csv",
             "llm_2026-10-05": rec / "B_stratum_pass2_LLM_GENERATED_INADMISSIBLE_2026-10-05.csv",
             "llm_2026-10-04": rec / "B_stratum_pass2_LLM_GENERATED_INADMISSIBLE_2026-10-04.csv"}
    lab = {n: {key[r["item"]]: tuple(norm(r[f]) for f in PASS2) for r in read(p)} for n, p in files.items()}
    for n, p in (("sol", ES / "hosted_llm_codebook_v2/edges_canonical.csv"), ("jev", ES / "jev_codebook_v2/edges_canonical.csv")):
        rows = {r["edge_id"]: r for r in read(p)}
        lab[n] = {e: tuple(norm(rows[e][f]) for f in PASS2) for e in key.values() if e in rows}
    names = list(lab)
    out = {}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            common = set(lab[a]) & set(lab[b])
            out[f"{a}~{b}"] = {"edges": len(common), "all_four_identical": sum(lab[a][e] == lab[b][e] for e in common),
                               "claim_dependent_identical": sum(lab[a][e][3] == lab[b][e][3] for e in common)}
    return out


# ---------------------------------------------------------------- calibration critical cells (5/8/11)
def calibration_cells():
    p = subprocess.run([sys.executable, str(ROOT / "analysis/experiments/21_make_v2_coding_files.py"), "cells"],
                       capture_output=True, text=True, cwd=ROOT)
    return {"returncode": p.returncode, "stdout": p.stdout.strip().splitlines()}


# ---------------------------------------------------------------- D1 and hosted-model cost
def costs():
    per_edge = [load(RES / f"ml/d1_2026-10-07/d1_meta_seed{s}.json")["inference_cpu_s_per_edge"] for s in (1, 2, 3)]
    m = sum(per_edge) / len(per_edge)
    est = (ES / "hosted_llm_codebook_v2/estimate_canonical.txt").read_text(encoding="utf-8")
    return {"d1_cpu_s_per_edge_by_seed": per_edge, "d1_cpu_s_per_edge_mean": round(m, 4), "chi_edges": 226793,
            "d1_cpu_hours_for_chi": round(m * 226793 / 3600, 2),
            "hosted_estimate_file": est.strip().splitlines()[-5:],
            "note": "the hosted estimate's per-token prices were passed on the command line and were not recorded"}


# ---------------------------------------------------------------- rerun consistency of the local and S2 systems
def rerun_consistency():
    out = {}
    sc = ES / "scicite_local/outputs"
    a, b = read(sc / "scicite_edge_preds_canonical_20261004.csv"), read(sc / "scicite_edge_preds_stab100_20261004.csv")
    lab = next(c for c in b[0] if "label" in c.lower() or "pred" in c.lower())
    ia = {r["edge_id"]: r[lab] for r in a}
    out["scicite"] = {"edges": len(b), "column": lab, "identical": sum(ia.get(r["edge_id"]) == r[lab] for r in b)}
    mc = ES / "aclarc_multicite/outputs"
    ca = {r["edge_id"]: r for r in map(json.loads, (mc / "multicite_scibert_canonical_20261004.jsonl").open())}
    cb = [json.loads(x) for x in (mc / "multicite_scibert_stab100_20261004.jsonl").open()]
    pick = lambda r: json.dumps({k: r.get(k) for k in ("edge_label", "labels", "marker_resolution") if k in r}, sort_keys=True)
    out["multicite"] = {"edges": len(cb), "identical": sum(pick(ca[r["edge_id"]]) == pick(r) for r in cb if r["edge_id"] in ca)}
    s2 = ES / "s2_influential/outputs"
    stab = sorted((s2 / "stab100_20261004").glob("*.csv"))
    out["s2"] = {"files": [p.name for p in stab]}
    canon = {p.name: p for p in (s2 / "canonical_20261004").glob("*.csv")}
    for p in stab:
        if p.name in canon:
            rb = read(p)
            ra = {r.get("edge_id"): r for r in read(canon[p.name])}
            cols = ["isInfluential", "intents", "context_intents", "n_contexts", "edge_evidence"]  # labels, not fetch metadata
            out["s2"][p.name] = {"edges": len(rb), "columns": cols, "identical": sum(all(ra.get(r["edge_id"], {}).get(c) == r[c] for c in cols)
                                                                   for r in rb)}
    return out


# ---------------------------------------------------------------- recovered Nature subset: ambiguous matches (10 / 5)
def recovered_ambiguity():
    audit = read(ND / "parity/parity_gate_edges.csv")
    cand = [r for r in audit if r["fully_matched"] == "1" and r["located"] != "none"]
    targets = Counter((r["pmcid"], r["grobid_ref"]) for r in cand)
    amb = [r for r in cand if targets[(r["pmcid"], r["grobid_ref"])] > 1]
    return {"candidates": len(cand), "ambiguous_jats_edges": len(amb),
            "ambiguous_grobid_targets": len({(r["pmcid"], r["grobid_ref"]) for r in amb}),
            "selected_jats_edges": len(cand) - len(amb), "all_jats_edges": len(audit)}


# ---------------------------------------------------------------- ACL-ARC run status
def aclarc_status():
    out = ES / "aclarc_multicite/outputs"
    return {p.name: sum(1 for _ in p.open()) for p in sorted(out.glob("aclarc_scaffold_canonical_*.jsonl"))}


SECTIONS = {"rationale_audit": rationale_audit, "rq3_flips": flips, "qwen_pairing_rule": qwen_pairing,
            "b300_eligible_by_corpus": b300_eligible_by_corpus, "split_overlap_2026-09-29": split_overlap,
            "a250_draw": a250_draw, "stratum_pass2_overlap": stratum_overlap, "calibration_cells": calibration_cells,
            "costs": costs, "rerun_consistency": rerun_consistency, "recovered_nature_ambiguity": recovered_ambiguity,
            "aclarc_runs": aclarc_status}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, default=RES / "paper_numbers/derived_numbers.json")
    a = ap.parse_args()
    if a.out.exists():
        sys.exit(f"refusing to overwrite {a.out}")
    out = {}
    for name, fn in SECTIONS.items():
        try:
            out[name] = fn()
        except Exception as e:  # report and continue; a missing private input should not hide the rest
            out[name] = {"error": f"{type(e).__name__}: {e}"}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=1)[:6000])


if __name__ == "__main__":
    main()
