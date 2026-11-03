#!/usr/bin/env python3
"""Exploratory, Part B of private/preregistration/AMBIGUITY_AND_RETEST_SPEC_2026-10-25.md (fixed before any retest
data existed; the spec names this script 42, a number already in use): does RQ3's reference dependence survive the
coder's own retest instability?

Fields: citation_function, cited_contribution_type (the fields with registered reference flips). Noise model per field:
switch probability p = 1 - raw agreement of the stratum Pass 1 retest (script 64); a switched label is drawn from the
coder's B marginal for that field, excluding the original label. Conservative variant: p = upper Wilson 95% bound of the
disagreement rate. Each of 10,000 draws takes a fresh noisy copy of the coder's B Pass 1 reference (300 edges) and the
registered cluster-bootstrap resample of the same index (seed 20261003), applied to both references. A pair flips in a
draw if its macro-F1 order differs between canonical and the noisy reference. Lineup, eligibility, coarsening and the
support >= 5 rule are those of 29_confirmatory_2026_10_08.py (primary variant), reused from it. A p = 0 run must
reproduce the registered flip probabilities. Survives = at least one pair flips with p > 0.5 under the main model;
the registered verdict excludes D1, so counts are given with and without D1.

    python3 analysis/experiments/67_rq3_under_retest_noise.py [--out PATH]
Default output: analysis/results/stability/rq3_under_retest_noise_2026-11-02.json (refuses to overwrite).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from collections import Counter
from itertools import combinations
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("conf", ROOT / "analysis/experiments/29_confirmatory_2026_10_08.py")
C = importlib.util.module_from_spec(spec)
spec.loader.exec_module(C)
S = C.S
RETEST = ROOT / "analysis/results/stability/stratum_pass1_retest_2026-11-01.json"
REGISTERED = ROOT / "analysis/results/confirmatory_2026-10-08/rq3.json"
NOISE_SEED = 20261025
D1 = "d1_distilled"


def wilson_upper(k, n, z=1.959964):
    ph = k / n
    return (ph + z * z / (2 * n) + z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n))) / (1 + z * z / n)


def noisy_copies(labels, marg, p, rng, n_draws):
    """labels: int array (edges). Returns (n_draws, edges) with each label switched w.p. p to marg minus itself."""
    T = np.tile(labels, (n_draws, 1))
    if p == 0:
        return T
    switch = rng.random(T.shape) < p
    for o in range(len(marg)):
        mask = switch & (T == o)
        k = int(mask.sum())
        if k:
            q = marg.copy()
            q[o] = 0
            T[mask] = rng.choice(len(q), size=k, p=q / q.sum())
    return T


def macro_f1_draws(T, pred, w_cl, elig, L):
    """T (D, E) true idx per draw, pred (E,), w_cl (D, E) bootstrap weights, elig (D, E) bool. Macro-F1 per draw."""
    D, E = T.shape
    w = w_cl * elig
    flat = (np.arange(D)[:, None] * L * L + T * L + pred[None, :]).ravel()
    M = np.bincount(flat, weights=w.ravel(), minlength=D * L * L).reshape(D, L, L)
    sup_full = np.stack([((T == l) & elig).sum(1) for l in range(L)], axis=1)   # registered rule: support in the sample
    rep = sup_full >= S.MIN_SUPPORT
    tp = np.diagonal(M, axis1=1, axis2=2)
    support, predicted = M.sum(2), M.sum(1)
    with np.errstate(divide="ignore", invalid="ignore"):
        f1 = np.where(support + predicted > 0, 2 * tp / (support + predicted), 0.0)
        return np.where(rep.any(1), (f1 * rep).sum(1) / rep.sum(1), np.nan)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, default=ROOT / "analysis/results/stability/rq3_under_retest_noise_2026-11-02.json")
    a = ap.parse_args()
    if a.out.exists():
        sys.exit(f"refusing to overwrite {a.out}")
    cw = S.crosswalk()
    alias, cluster = S.components()
    canon, v2b_raw, _, _ = C.references(cw)
    v2b, _ = C.drop_unclear(v2b_raw)
    B1 = sorted(set(v2b_raw["citation_function"]))
    boot = C.Boot(B1, cluster)
    sysp = C.load_systems(cw, alias)
    retest = json.load(open(RETEST))["fields"]
    registered = {(x["field"], tuple(x["pair"])): x["p_flip"]
                  for x in json.load(open(REGISTERED))["flip_probabilities_between_references"]}
    rng = np.random.default_rng(NOISE_SEED)
    out = {"design": __doc__.split("\n\n")[1].replace("\n", " "), "noise": {}, "flips": {}, "summary": {}}

    for field in C.PASS1:
        # canonical side: exactly as registered (same Boot, same W)
        canon_boot, systems = {}, []
        for system, seeds in sysp.items():
            per_seed = []
            for seed, fields in seeds.items():
                pred = fields.get(field, {})
                pairs = [(e, *ep) for e in B1 if e in canon[field] and e in pred
                         for ep in [C.eligible_pair(system, field, canon[field][e], pred[e], "primary")] if ep]
                m = C.macro_f1(pairs, boot)
                if m:
                    per_seed.append(m["boot_f1"])
            if per_seed:
                canon_boot[system] = np.nanmean(np.stack(per_seed), axis=0)
        # coder side: noisy copies
        ref = v2b[field]
        kedges = sorted(e for e in B1 if e in ref)
        marg_c = Counter(ref[e] for e in kedges)
        all_pred = {v for seeds in sysp.values() for f in seeds.values() for v in f.get(field, {}).values()}
        labels = sorted(set(marg_c) | all_pred)
        li = {l: i for i, l in enumerate(labels)}
        L = len(labels)
        base = np.array([li[ref[e]] for e in kedges])
        marg = np.array([marg_c.get(l, 0) for l in labels], dtype=float)
        cl_idx = np.array([boot.idx(e) for e in kedges])
        Wk = boot.W[:, cl_idx].astype(float)                     # (D, E): same resample index as canonical side
        coarse = np.array([l in C.COARSE_FN for l in labels])
        n_dis = round((1 - retest[field]["all_items"]["raw_agreement"]) * retest[field]["all_items"]["n"])
        n_rt = retest[field]["all_items"]["n"]
        models = {"p0_validation": 0.0, "main": n_dis / n_rt, "conservative": wilson_upper(n_dis, n_rt)}
        out["noise"][field] = {"retest_disagreements": f"{n_dis}/{n_rt}", **{k: round(v, 4) for k, v in models.items()},
                               "coder_B_marginal": dict(marg_c)}
        for mname, p in models.items():
            T = noisy_copies(base, marg, p, rng, C.N_BOOT)
            kboot = {}
            for system, seeds in sysp.items():
                if system not in canon_boot:
                    continue
                per_seed = []
                for seed, fields in seeds.items():
                    pred = fields.get(field, {})
                    sel = np.array([e in pred for e in kedges])
                    if not sel.any():
                        continue
                    pidx = np.array([li[pred[e]] for e, s in zip(kedges, sel) if s])
                    Ts = T[:, sel]
                    elig = coarse[Ts] if system in ("scicite", "s2_intents") else np.ones_like(Ts, dtype=bool)
                    per_seed.append(macro_f1_draws(Ts, pidx, Wk[:, sel], elig, L))
                if per_seed:
                    kboot[system] = np.nanmean(np.stack(per_seed), axis=0)
            rows = []
            for s1, s2 in combinations(sorted(set(canon_boot) & set(kboot)), 2):
                x = np.sign(canon_boot[s1] - canon_boot[s2])
                y = np.sign(kboot[s1] - kboot[s2])
                rows.append({"pair": [s1, s2], "p_flip": round(float(np.mean(x != y)), 4),
                             "registered_p_flip": registered.get((field, (s1, s2)))})
            out["flips"].setdefault(mname, {})[field] = rows

    for mname, by_field in out["flips"].items():
        rows = [r for rs in by_field.values() for r in rs]
        no_d1 = [r for r in rows if D1 not in r["pair"]]
        out["summary"][mname] = {
            "pairs_flipping_gt_0.5": f"{sum(r['p_flip'] > 0.5 for r in rows)}/{len(rows)}",
            "pairs_flipping_gt_0.5_without_D1": f"{sum(r['p_flip'] > 0.5 for r in no_d1)}/{len(no_d1)}",
            "flipping_pairs": [r["pair"] + [r["p_flip"]] for r in rows if r["p_flip"] > 0.5]}
    val = [r for rs in out["flips"]["p0_validation"].values() for r in rs]
    out["validation_max_abs_diff_vs_registered"] = max(abs(r["p_flip"] - r["registered_p_flip"]) for r in val)
    main_no_d1 = [r for rs in out["flips"]["main"].values() for r in rs if D1 not in r["pair"]]
    out["verdict_registered_lineup_without_D1"] = (
        "SURVIVES (at least one pair flips with p > 0.5 under the main noise model)"
        if any(r["p_flip"] > 0.5 for r in main_no_d1) else "DOES NOT SURVIVE")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({k: out[k] for k in ("noise", "summary", "validation_max_abs_diff_vs_registered",
                                          "verdict_registered_lineup_without_D1")}, indent=1))


if __name__ == "__main__":
    main()
