#!/usr/bin/env python3
"""Registered confirmatory analyses (OSF https://osf.io/nfz9c; STS_PREREG_2026-10-03.md §6-§7, §10),
applying private/preregistration/DEVIATIONS.md. Run 8 Oct 2026, after all human v2 coding was locked.

Human references (never pooled)
  canonical  analysis/derived (Yiren Liu), restricted to the edges being compared
  v2B        Kevin v2, sample B: Pass 1 = B_stratum_pass1 + B_rest_pass1 (300 edges);
             Pass 2 = B_rest_pass2 only (240) -- stratum Pass 2 excluded (deviation 3a)
  v2A        Kevin v2, sample A250 (deviation 4): Pass 2 only; A's function = Kevin's v1 label
`unclear` in a reference is excluded field by field (count reported, §10). uptake_degree is scored
only where both reference and system have uptake (§10).

Outputs (refuses to overwrite): analysis/results/confirmatory_2026-10-08/*.json
"""
from __future__ import annotations

import csv
import importlib.util
import json
import math
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("scorer", ROOT / "analysis/experiments/22_score_systems.py")
S = importlib.util.module_from_spec(spec)
spec.loader.exec_module(S)

OUT = ROOT / "analysis/results/confirmatory_2026-10-08"
V2 = ROOT / "annotations/generated/v2_coding_2026-10-02"
RES = ROOT / "analysis/results"
N_BOOT, SEED = 10_000, 20261003
PASS1 = ("cited_contribution_type", "citation_function")
PASS2 = ("credits_contribution", "uptake_mode", "uptake_degree", "claim_dependent")
COARSE_FN = {"background", "uses_method", "compares_against"}
MC_CLASSES = {"method|artifact", "extension", "framing", "contrast", "none"}
V1_NMI = 0.700


def read(p):
    return S.read_csv(Path(p))


# ----------------------------------------------------------------------------------- references

def coded(stem, fields):
    edge = {k["item"]: k["edge_id"] for k in read(V2 / f"{stem}_key.csv")}
    out = defaultdict(dict)
    for row in read(V2 / f"{stem}.csv"):
        for f in fields:
            v = (row.get(f) or "").strip()
            if v and v != "not_applicable":
                out[f][edge[row["item"]]] = "background" if v == "motivates_background" else v
    return out


def references(cw):
    v2b = defaultdict(dict)
    for stem, fields in (("B_stratum_pass1", PASS1), ("B_rest_pass1", PASS1), ("B_rest_pass2", PASS2)):
        for f, d in coded(stem, fields).items():
            v2b[f].update(d)
    v2a = coded("A250_pass2", PASS2)
    v1 = S.ref_v1(cw)
    v2a["citation_function"] = {e: v1["citation_function"][e] for e in
                                set().union(*[set(d) for d in v2a.values()]) if e in v1["citation_function"]}
    canon = S.ref_canonical(cw)
    return canon, v2b, v2a, v1


def drop_unclear(ref):
    out, dropped = {}, {}
    for f, d in ref.items():
        out[f] = {e: v for e, v in d.items() if v != "unclear"}
        dropped[f] = sum(v == "unclear" for v in d.values())
    return out, dropped


# ----------------------------------------------------------------------------------- systems

def load_systems(cw, alias):
    sysp = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))   # system -> seed -> field -> {edge: label}

    def put(system, seed, field, e, v):
        v = (v or "").strip()
        if v and v != "not_applicable":
            sysp[system][seed][field][alias.get(e, e)] = S.norm(field, v, cw)

    for name, path in (("gpt-6.1-sol", RES / "external_systems/hosted_llm_codebook_v2/edges_canonical.csv"),
                       ("jev-1.13.0", RES / "external_systems/jev_codebook_v2/edges_canonical.csv")):
        for r in read(path):
            for f in PASS1 + PASS2:
                put(name, 0, f, r["edge_id"], r.get(f))
    for s in (1, 2, 3):
        for r in read(RES / f"ml/d1_2026-10-07/d1_frozen_predictions_seed{s}.csv"):
            for f in PASS1 + PASS2:
                put("d1_distilled", s, f, r["edge_id"], r.get(f))
    for s in (1, 2, 3, 4, 5):
        for c in ("hci", "nature_oa"):
            for r in read(RES / f"ml/frozen_predictions/function/scibert_pooled_uncond_{c}_seed{s}.csv"):
                put("encoder_scibert", s, "citation_function", r["edge_id_v1"], r["y_pred"])
    conv = RES / "external_systems/scoring_inputs_canonical_20261004"
    for name, fname, field in (("scicite", "scicite_function_all.csv", "citation_function"),
                               ("s2_intents", "s2_intents_function_all.csv", "citation_function"),
                               ("multicite", "multicite_uptake.csv", "uptake_mode"),
                               ("s2_influential", "s2_influential_claim.csv", "claim_dependent")):
        for r in read(conv / fname):
            put(name, 0, field, r["edge_id"], r["label"])
    for r in read(RES / "external_systems/no_model_baseline/no_model_baseline.csv"):
        if r["field"] in PASS1:
            put("no_model_baseline", 0, r["field"], r["edge_id"], r["label"])
    return sysp


def eligible_pair(system, field, ref_label, pred, variant):
    """(ref, pred) after registered coarsening, or None if the edge is excluded for this system."""
    if system in ("scicite", "s2_intents") and field == "citation_function":
        if variant == "primary" and ref_label not in COARSE_FN:
            return None
        return ref_label, pred
    if system == "multicite" and field == "uptake_mode":
        r = "method|artifact" if ref_label in ("method", "artifact") else ref_label
        if variant == "primary" and r not in MC_CLASSES:
            return None
        return r, pred
    return ref_label, pred


# ----------------------------------------------------------------------------------- statistics

class Boot:
    def __init__(self, edges, cluster):
        self.cl = sorted({cluster.get(e, f"singleton:{e}") for e in edges})
        self.ci = {c: i for i, c in enumerate(self.cl)}
        self.cluster = cluster
        rng = np.random.default_rng(SEED)
        draws = rng.integers(0, len(self.cl), size=(N_BOOT, len(self.cl)))
        self.W = np.stack([np.bincount(d, minlength=len(self.cl)) for d in draws])

    def idx(self, e):
        return self.ci[self.cluster.get(e, f"singleton:{e}")]

    def confusions(self, pairs, labels):
        li = {l: i for i, l in enumerate(labels)}
        per = np.zeros((len(self.cl), len(labels), len(labels)))
        for e, t, p in pairs:
            per[self.idx(e), li[t], li[p]] += 1
        return per.sum(0), np.tensordot(self.W, per, axes=(1, 0))


def macro_f1(pairs, boot):
    """Registered macro-F1 (support >= 5), kappa, accuracy; full + bootstrap vectors."""
    if not pairs:
        return None
    labels = sorted({t for _, t, _ in pairs} | {p for _, _, p in pairs})
    support = Counter(t for _, t, _ in pairs)
    rep = np.array([support[l] >= S.MIN_SUPPORT for l in labels])
    full, b = boot.confusions(pairs, labels)
    m, a, k = S.metrics_from_confusion(full, rep)
    bm, ba, bk = S.metrics_from_confusion(b, rep)
    return {"n": len(pairs), "macro_f1": float(m), "kappa": float(k), "accuracy": float(a),
            "boot_f1": bm, "boot_kappa": bk,
            "withheld": [l for l, r in zip(labels, rep) if not r],
            "majority_rate": max(support.values()) / len(pairs)}


def ci(x):
    x = np.asarray(x)
    x = x[~np.isnan(x)]
    return [round(float(np.percentile(x, 2.5)), 4), round(float(np.percentile(x, 97.5)), 4)] if len(x) else None


def binary_stats(pairs, boot):
    """kappa and NMI (geometric) for two binary series; full + bootstrap."""
    full, b = boot.confusions([(e, int(x), int(y)) for e, x, y in pairs], [0, 1])
    return kappa_nmi(full), np.array([kappa_nmi(m) for m in b])


def kappa_nmi(M):
    n = M.sum()
    if n == 0:
        return (np.nan, np.nan)
    po = np.trace(M) / n
    r, c = M.sum(1) / n, M.sum(0) / n
    pe = (r * c).sum()
    kap = (po - pe) / (1 - pe) if pe < 1 else np.nan
    return kap, nmi_from_counts(M)


def nmi_from_counts(M):
    n = M.sum()
    p = M / n
    px, py = p.sum(1), p.sum(0)
    hx = -sum(v * math.log(v) for v in px if v > 0)
    hy = -sum(v * math.log(v) for v in py if v > 0)
    mi = sum(p[i, j] * math.log(p[i, j] / (px[i] * py[j]))
             for i in range(M.shape[0]) for j in range(M.shape[1]) if p[i, j] > 0)
    return mi / math.sqrt(hx * hy) if hx > 0 and hy > 0 else np.nan


def nmi_multi(pairs, boot):
    xs = sorted({x for _, x, _ in pairs})
    ys = sorted({y for _, _, y in pairs})
    xi, yi = {v: i for i, v in enumerate(xs)}, {v: i for i, v in enumerate(ys)}
    per = np.zeros((len(boot.cl), len(xs), len(ys)))
    for e, x, y in pairs:
        per[boot.idx(e), xi[x], yi[y]] += 1
    full = per.sum(0)
    b = np.tensordot(boot.W, per, axes=(1, 0))
    return nmi_from_counts(full), np.array([nmi_from_counts(m) for m in b])


def r4(x):
    return None if x is None or (isinstance(x, float) and math.isnan(x)) else round(float(x), 4)


# ----------------------------------------------------------------------------------- analyses

def rq1b(canon, ref, edges, boot, label):
    R = {e: canon["realization"][e] for e in edges if e in canon["realization"]}
    out = {"sample": label}
    for name, realized in (("primary_realized_partial_or_yes", {"partial", "yes"}),
                           ("sensitivity_realized_yes_only", {"yes"})):
        trip = [(e, R[e] in realized, ref["credits_contribution"][e] == "specific",
                 ref["uptake_mode"][e] not in ("none", "unclear"))
                for e in edges if e in R and e in ref["credits_contribution"] and e in ref["uptake_mode"]]
        (kc, nc), bc = binary_stats([(e, r, c) for e, r, c, _ in trip], boot)
        (ku, nu), bu = binary_stats([(e, r, u) for e, r, _, u in trip], boot)
        diff = bc[:, 0] - bu[:, 0]
        out[name] = {"n": len(trip),
                     "canonical_realized_rate": r4(np.mean([t[1] for t in trip])),
                     "kevin_credits_specific_rate": r4(np.mean([t[2] for t in trip])),
                     "kevin_uptake_rate": r4(np.mean([t[3] for t in trip])),
                     "kappa_realized_vs_crediting": r4(kc), "kappa_crediting_ci95": ci(bc[:, 0]),
                     "kappa_realized_vs_uptake": r4(ku), "kappa_uptake_ci95": ci(bu[:, 0]),
                     "kappa_difference_crediting_minus_uptake": r4(kc - ku), "difference_ci95": ci(diff),
                     "nmi_geometric_realized_vs_crediting": r4(nc), "nmi_crediting_ci95": ci(bc[:, 1]),
                     "nmi_geometric_realized_vs_uptake": r4(nu), "nmi_uptake_ci95": ci(bu[:, 1])}
    p = out["primary_realized_partial_or_yes"]
    lo, hi = p["difference_ci95"]
    out["verdict"] = ("SUPPORTED" if lo > 0 else "NOT SUPPORTED (favours uptake)" if hi < 0
                      else "NOT SUPPORTED (CI includes 0)")
    s = out["sensitivity_realized_yes_only"]
    out["threshold_dependent"] = (s["difference_ci95"][0] > 0) != (lo > 0)
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    targets = {n: OUT / n for n in ("rq1b.json", "rq2.json", "rq3.json", "d1.json")}
    for p in targets.values():
        if p.exists():
            raise SystemExit(f"refusing to overwrite {p}")
    cw = S.crosswalk()
    alias, cluster = S.components()
    canon, v2b_raw, v2a_raw, v1 = references(cw)
    v2b, drop_b = drop_unclear(v2b_raw)
    v2a, drop_a = drop_unclear(v2a_raw)
    B1 = sorted(set(v2b_raw["citation_function"]))          # 300
    B2 = sorted(set(v2b_raw["uptake_mode"]))                # 240
    A = sorted(set(v2a_raw["uptake_mode"]))                 # 250
    bootB, bootA = Boot(B1, cluster), Boot(A, cluster)
    counts = {"B_pass1_edges": len(B1), "B_pass2_edges": len(B2), "A250_edges": len(A),
              "unclear_excluded_B": drop_b, "unclear_excluded_A": drop_a}

    # ---------------- RQ1(b)
    res1 = {"counts": counts,
            "B": rq1b(canon, v2b, B2, bootB, "B (240 Pass 2 edges; deviation 3a; unclear excluded)"),
            "A250_replication": rq1b(canon, v2a, A, bootA, "A250 (deviation 4; unclear excluded)")}
    targets["rq1b.json"].write_text(json.dumps(res1, indent=2) + "\n")

    # ---------------- RQ2
    fnB, upB, crB, clB = v2b_raw["citation_function"], v2b_raw["uptake_mode"], v2b_raw["credits_contribution"], v2b_raw["claim_dependent"]
    cells = {
        "background x uptake in {finding, framing, design_knowledge}":
            sum(fnB.get(e) == "background" and upB[e] in ("finding", "framing", "design_knowledge") for e in B2),
        "background x specific x none":
            sum(fnB.get(e) == "background" and crB.get(e) == "specific" and upB[e] == "none" for e in B2),
        "claim_dependent=no x uptake not in {none, unclear}":
            sum(clB.get(e) == "no" and upB[e] not in ("none", "unclear") for e in B2)}
    rq2 = {"a_critical_cells_B": {"counts": cells, "verdict": "SUPPORTED (all non-empty)" if all(cells.values())
                                  else "NOT SUPPORTED (codebook defect)"}}

    def realized(u):
        return "realized" if u not in ("none",) else "not_realized"

    def nmi_fn(fn, up, edges, boot):
        pairs = [(e, fn[e], realized(up[e])) for e in edges if e in fn and e in up and up[e] != "unclear"]
        v, b = nmi_multi(pairs, boot)
        return len(pairs), v, b

    nA, vA, bA = nmi_fn(v2a_raw["citation_function"], v2a_raw["uptake_mode"], A, bootA)
    nB, vB, bB = nmi_fn(fnB, upB, B2, bootB)
    v1pairs_m = [(e, v1["citation_function"][e], v1["realization"][e]) for e in A
                 if e in v1["citation_function"] and e in v1["realization"]]
    v1pairs_b = [(e, f, "realized" if r != "no" else "not_realized") for e, f, r in v1pairs_m]
    v1m, bv1m = nmi_multi(v1pairs_m, bootA)
    v1b, bv1b = nmi_multi(v1pairs_b, bootA)
    lo, hi = ci(bA - V1_NMI)
    rq2["b_function_realization_nmi"] = {
        "registered": "NMI(function, realized-vs-not) on A vs v1's 0.700; prediction lower, CI of difference excludes 0",
        "A250": {"n": nA, "nmi_geometric_v2": r4(vA), "ci95": ci(bA),
                 "difference_vs_0.700": r4(vA - V1_NMI), "difference_ci95": [lo, hi]},
        "verdict": "SUPPORTED" if hi < 0 else ("NOT SUPPORTED (CI includes 0)" if lo <= 0 <= hi else "NOT SUPPORTED (higher)"),
        "B_descriptive": {"n": nB, "nmi_geometric_v2": r4(vB), "ci95": ci(bB)},
        "transparency_same_250_edges_v1": {
            "n": len(v1pairs_m), "nmi_v1_multiclass": r4(v1m), "ci95": ci(bv1m),
            "nmi_v1_binarized": r4(v1b), "ci95_binarized": ci(bv1b),
            "paired_difference_v2_minus_v1_binarized": r4(vA - v1b), "paired_ci95": ci(bA - bv1b),
            "note": "not registered; shows the comparison on identical edges and an identical binarisation"}}

    def xtab(a, b, edges):
        t = Counter((a[e], b[e]) for e in edges if e in a and e in b)
        return {f"{x} | {y}": n for (x, y), n in sorted(t.items())}

    upbin = {e: ("uptake" if upB[e] not in ("none", "unclear") else upB[e]) for e in B2}
    off_claim = sum(1 for e in B2 if e in clB and clB[e] in ("yes", "no") and upbin[e] in ("uptake", "none")
                    and (clB[e] == "yes") != (upbin[e] == "uptake"))
    n_claim = sum(1 for e in B2 if e in clB and clB[e] in ("yes", "no") and upbin[e] in ("uptake", "none"))
    off_cred = sum(1 for e in B2 if crB.get(e) in ("specific", "generic") and upbin[e] in ("uptake", "none")
                   and (crB[e] == "specific") != (upbin[e] == "uptake"))
    n_cred = sum(1 for e in B2 if crB.get(e) in ("specific", "generic") and upbin[e] in ("uptake", "none"))
    rq2["c_within_v2_contrasts_B"] = {
        "claim_dependent_x_uptake": xtab(clB, upbin, B2),
        "claim_vs_uptake_off_diagonal": f"{off_claim}/{n_claim} = {off_claim / n_claim:.3f}",
        "credits_x_uptake": xtab(crB, upbin, B2),
        "credits_vs_uptake_off_diagonal": f"{off_cred}/{n_cred} = {off_cred / n_cred:.3f}"}
    bp, ap_ = res1["B"]["primary_realized_partial_or_yes"], res1["A250_replication"]["primary_realized_partial_or_yes"]
    overlap = lambda x, y: x[0] <= y[1] and y[0] <= x[1]
    rq2["d_A_vs_B_consistency"] = {
        "rq1b_difference": {"B": [bp["kappa_difference_crediting_minus_uptake"], bp["difference_ci95"]],
                            "A250": [ap_["kappa_difference_crediting_minus_uptake"], ap_["difference_ci95"]],
                            "cis_overlap": overlap(bp["difference_ci95"], ap_["difference_ci95"])},
        "rq2b_nmi": {"B": [r4(vB), ci(bB)], "A250": [r4(vA), ci(bA)], "cis_overlap": overlap(ci(bB), ci(bA))}}
    targets["rq2.json"].write_text(json.dumps(rq2, indent=2) + "\n")

    # ---------------- RQ3
    sysp = load_systems(cw, alias)
    refsets = {"canonical@B": ({f: canon[f] for f in PASS1}, B1, bootB),
               "kevin_v2@B": (v2b, None, bootB),
               "kevin_v2@A250": (v2a, A, bootA)}
    scores, boots = {}, {}
    for refname, (ref, edges, boot) in refsets.items():
        for system, seeds in sysp.items():
            for field in PASS1 + PASS2:
                if field not in ref or (refname == "canonical@B" and field not in PASS1):
                    continue
                if refname == "kevin_v2@A250" and field in PASS1:
                    continue
                elist = edges if edges is not None else (B1 if field in PASS1 else B2)
                for variant in ("primary", "sensitivity_all"):
                    if variant == "sensitivity_all" and system not in ("scicite", "s2_intents", "multicite"):
                        continue
                    per_seed = []
                    for seed, fields in seeds.items():
                        pred = fields.get(field, {})
                        pairs = []
                        for e in elist:
                            if e not in ref[field] or e not in pred:
                                continue
                            if field == "uptake_degree" and pred[e] not in ("full", "partial"):
                                continue
                            ep = eligible_pair(system, field, ref[field][e], pred[e], variant)
                            if ep:
                                pairs.append((e, *ep))
                        m = macro_f1(pairs, boot)
                        if m:
                            per_seed.append(m)
                    if not per_seed:
                        continue
                    f1 = np.mean([m["macro_f1"] for m in per_seed])
                    bf1 = np.nanmean(np.stack([m["boot_f1"] for m in per_seed]), axis=0)
                    key = (refname, field, system, variant)
                    boots[key] = bf1
                    scores["|".join(key)] = {
                        "n": per_seed[0]["n"], "seeds": len(per_seed),
                        "macro_f1": r4(f1), "macro_f1_ci95": ci(bf1),
                        "macro_f1_seed_sd": r4(np.std([m["macro_f1"] for m in per_seed], ddof=1)) if len(per_seed) > 1 else None,
                        "kappa": r4(np.mean([m["kappa"] for m in per_seed])),
                        "accuracy": r4(np.mean([m["accuracy"] for m in per_seed])),
                        "majority_rate": r4(per_seed[0]["majority_rate"]), "withheld": per_seed[0]["withheld"]}
    # rankings and flips (primary variant only)
    prim = {k: v for k, v in boots.items() if k[3] == "primary"}
    flips_ref, flips_field = [], []
    for field in PASS1:
        sys_c = {k[2] for k in prim if k[0] == "canonical@B" and k[1] == field}
        sys_k = {k[2] for k in prim if k[0] == "kevin_v2@B" and k[1] == field}
        for s1, s2 in combinations(sorted(sys_c & sys_k), 2):
            a = np.sign(prim[("canonical@B", field, s1, "primary")] - prim[("canonical@B", field, s2, "primary")])
            b = np.sign(prim[("kevin_v2@B", field, s1, "primary")] - prim[("kevin_v2@B", field, s2, "primary")])
            flips_ref.append({"field": field, "pair": [s1, s2], "p_flip": r4(np.mean(a != b))})
    v2fields = [f for f in PASS1 + PASS2]
    for f1_, f2_ in combinations(v2fields, 2):
        s_a = {k[2] for k in prim if k[0] == "kevin_v2@B" and k[1] == f1_}
        s_b = {k[2] for k in prim if k[0] == "kevin_v2@B" and k[1] == f2_}
        for s1, s2 in combinations(sorted(s_a & s_b), 2):
            a = np.sign(prim[("kevin_v2@B", f1_, s1, "primary")] - prim[("kevin_v2@B", f1_, s2, "primary")])
            b = np.sign(prim[("kevin_v2@B", f2_, s1, "primary")] - prim[("kevin_v2@B", f2_, s2, "primary")])
            flips_field.append({"fields": [f1_, f2_], "pair": [s1, s2], "p_flip": r4(np.mean(a != b))})
    rankings = {}
    for (refname, field, system, variant) in prim:
        rankings.setdefault(f"{refname}|{field}", []).append(
            (system, scores["|".join((refname, field, system, variant))]["macro_f1"]))
    rankings = {k: sorted(v, key=lambda x: -(x[1] or -1)) for k, v in rankings.items()}
    tau = {}
    for field in PASS1:
        c = dict(rankings.get(f"canonical@B|{field}", []))
        k = dict(rankings.get(f"kevin_v2@B|{field}", []))
        common = sorted(set(c) & set(k))
        if len(common) >= 5:
            conc = disc = 0
            for s1, s2 in combinations(common, 2):
                x = np.sign(c[s1] - c[s2]) * np.sign(k[s1] - k[s2])
                conc += x > 0
                disc += x < 0
            tau[field] = {"n_systems": len(common), "kendall_tau_a": r4((conc - disc) / (len(common) * (len(common) - 1) / 2))}
        else:
            tau[field] = {"n_systems": len(common), "note": "fewer than 5 eligible systems; not reported (§7)"}
    max_flip = max([x["p_flip"] for x in flips_ref + flips_field] or [0])
    # construct recovery: each system's uptake-type output vs Kevin's three v2 binaries on B
    kb = {"credits_specific": {e: crB[e] == "specific" for e in B2 if crB.get(e) in ("specific", "generic")},
          "uptake": {e: upB[e] not in ("none",) for e in B2 if upB.get(e) not in (None, "unclear")},
          "claim_dependent_yes": {e: clB[e] == "yes" for e in B2 if clB.get(e) in ("yes", "no")}}
    sysbin = {}
    for system, seeds in sysp.items():
        f = seeds[min(seeds)]
        if "uptake_mode" in f:
            sysbin[system] = {e: v not in ("none",) for e, v in f["uptake_mode"].items() if v != "unclear"}
        elif "claim_dependent" in f:
            sysbin[system] = {e: v == "yes" for e, v in f["claim_dependent"].items() if v in ("yes", "no")}
    sysbin["canonical_realization"] = {e: v in ("partial", "yes") for e, v in canon["realization"].items()}
    recovery = {}
    for system, sb in sysbin.items():
        row = {}
        for name, kv in kb.items():
            pairs = [(e, sb[e], kv[e]) for e in B2 if e in sb and e in kv]
            if len(pairs) < 10:
                continue
            (k_, n_), _ = binary_stats(pairs, bootB)
            row[name] = {"n": len(pairs), "kappa": r4(k_), "nmi_geometric": r4(n_)}
        if row:
            best = max(row, key=lambda x: row[x]["kappa"] if row[x]["kappa"] is not None else -9)
            recovery[system] = {"by_construct": row, "agrees_most_with": best}
    rq3 = {"scores": scores, "rankings_macro_f1": rankings, "kendall_tau_canonical_vs_kevin": tau,
           "flip_probabilities_between_references": flips_ref,
           "flip_probabilities_between_fields": flips_field,
           "max_flip_probability": max_flip,
           "verdict": ("SUPPORTED (some pairwise order flips with p > 0.5)" if max_flip > 0.5
                       else "NOT SUPPORTED (rankings stable: no flip p > 0.5)"),
           "construct_recovery_B": recovery,
           "notes": ["encoder = SciBERT pooled, unconditioned, 5 seeds (frozen predictions); mean macro-F1 over seeds",
                     "d1_distilled = 3 seeds; mean over seeds",
                     "no-model baseline exists only for Pass 1 fields: no training-split human labels exist for v2-only fields",
                     "S2 isInfluential: missing (no S2 context) excluded, not 'no'; self-citation exclusion NOT applied "
                     "(author overlap not available in the frozen snapshot) — registered, not computable",
                     "coarsening (§5): primary excludes reference labels with no system class; sensitivity counts them as errors"]}
    targets["rq3.json"].write_text(json.dumps(rq3, indent=2, default=float) + "\n")

    # ---------------- D1 vs Sol on Kevin v2 (B)
    d1 = {}
    for field in PASS1 + PASS2:
        d = scores.get(f"kevin_v2@B|{field}|d1_distilled|primary")
        s = scores.get(f"kevin_v2@B|{field}|gpt-6.1-sol|primary")
        if d and s:
            d1[field] = {"d1_macro_f1_mean": d["macro_f1"], "d1_seed_sd": d["macro_f1_seed_sd"], "d1_ci95": d["macro_f1_ci95"],
                         "sol_macro_f1": s["macro_f1"], "sol_ci95": s["macro_f1_ci95"],
                         "d1_minus_sol": r4((d["macro_f1"] or 0) - (s["macro_f1"] or 0))}
    sep = recovery.get("d1_distilled", {}), recovery.get("gpt-6.1-sol", {})
    targets["d1.json"].write_text(json.dumps({"per_field_vs_sol_on_B": d1,
                                              "construct_recovery_d1_vs_sol": {"d1": sep[0], "sol": sep[1]}},
                                             indent=2) + "\n")
    print(json.dumps({"rq1b_B": res1["B"]["verdict"], "rq1b_A": res1["A250_replication"]["verdict"],
                      "rq2a": rq2["a_critical_cells_B"]["verdict"], "rq2b": rq2["b_function_realization_nmi"]["verdict"],
                      "rq3": rq3["verdict"], "max_flip": max_flip}, indent=2))


if __name__ == "__main__":
    main()
