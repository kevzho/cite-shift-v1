#!/usr/bin/env python3
"""Registered calibration (pre-registration §8, OSF osf.io/nfz9c): systems with probabilities -- Jev 1.13.0
(per-field choice probabilities), the fine-tuned encoder (citation_function, seeds 1-5, unconditioned) and D1
(seeds 1-3) -- ECE (15 equal-width bins) and Brier (multiclass sum of squares) against Kevin v2 (sample B), and on
the D3 edges against the soft label distribution. GPT-6.1 Sol and Qwen expose no probabilities in their harnesses
(not applicable).

Rules fixed before the run (12 Oct 2026):
  * A system's class probabilities are mapped to v2 names with the registered mechanical crosswalk
    (22_score_systems.norm) and summed per v2 label; mass on labels the reference never uses (a system's
    `unclear`, unmappable encoder classes) stays in the vector and counts as probability on a wrong answer.
  * References and exclusions as 29_confirmatory_2026_10_08.py: B Pass 1 fields on 300 edges, Pass 2 on the 240
    B-rest edges; reference `unclear` excluded. uptake_degree only where the reference has a degree and the
    system's most probable degree is full/partial; its probabilities are renormalised over {full, partial}.
  * ECE: confidence = max probability, correct = predicted label equals the reference (soft target: soft mass on
    the predicted label, as D3). Seeds: per-seed mean ± sd; CIs from seed-averaged probabilities, cluster
    bootstrap over components (10,000, seed 20261003).
Output (refuses to overwrite): analysis/results/calibration_2026-10-12/calibration.json + docs/results note.
"""
from __future__ import annotations

import csv
import importlib.util
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def mod(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


C = mod("conf", "analysis/experiments/29_confirmatory_2026_10_08.py")
D3 = mod("d3", "analysis/ml/11_d3_soft_labels.py")
S = C.S
RES = ROOT / "analysis/results"
OUT = RES / "calibration_2026-10-12"
FIELDS = C.PASS1 + C.PASS2
BINS = 15


def load_probs(cw, alias):
    P = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))   # system -> seed -> field -> {edge: {label: p}}

    def put(system, seed, field, e, probs):
        m = defaultdict(float)
        for k, v in probs.items():
            m[S.norm(field, k, cw) or f"unmapped:{k}"] += float(v)
        P[system][seed][field][alias.get(e, e)] = dict(m)

    for r in C.read(RES / "external_systems/jev_codebook_v2/edges_canonical.csv"):
        for f in FIELDS:
            if r.get(f"{f}_probs"):
                put("jev-1.13.0", 0, f, r["edge_id"], json.loads(r[f"{f}_probs"]))
    for s in (1, 2, 3):
        for r in C.read(RES / f"ml/d1_2026-10-07/d1_frozen_predictions_seed{s}.csv"):
            for f in FIELDS:
                put("d1_distilled", s, f, r["edge_id"],
                    {k.split("__p_", 1)[1]: v for k, v in r.items() if k.startswith(f + "__p_")})
    for s in (1, 2, 3, 4, 5):
        for c in ("hci", "nature_oa"):
            for r in C.read(RES / f"ml/frozen_predictions/function/scibert_pooled_uncond_{c}_seed{s}.csv"):
                put("encoder_scibert", s, "citation_function", r["edge_id_v1"],
                    {k[2:]: v for k, v in r.items() if k.startswith("p_")})
    return P


def degree_view(p):
    """uptake_degree: None unless the most probable degree is full/partial; else renormalised over the two."""
    if max(p, key=p.get) not in ("full", "partial"):
        return None
    z = p.get("full", 0.0) + p.get("partial", 0.0)
    return {"full": p.get("full", 0.0) / z, "partial": p.get("partial", 0.0) / z}


def matrices(probs, target, edges, field):
    """P (n x L), T (n x L) over the union label space; target is {edge: label} or {edge: {label: mass}}."""
    rows = []
    for e in edges:
        if e not in probs or e not in target:
            continue
        p = probs[e]
        if field == "uptake_degree":
            p = degree_view(p)
            if p is None:
                continue
        t = target[e] if isinstance(target[e], dict) else {target[e]: 1.0}
        rows.append((e, p, t))
    L = sorted({k for _, p, t in rows for k in list(p) + list(t)})
    li = {l: i for i, l in enumerate(L)}
    P, T = np.zeros((len(rows), len(L))), np.zeros((len(rows), len(L)))
    for i, (_, p, t) in enumerate(rows):
        for k, v in p.items():
            P[i, li[k]] = v
        for k, v in t.items():
            T[i, li[k]] = v
    return [e for e, _, _ in rows], P, T


def per_edge(P, T):
    conf = P.max(1)
    acc = T[np.arange(len(T)), P.argmax(1)]
    b = np.clip(np.ceil(conf * BINS).astype(int) - 1, 0, BINS - 1)
    return conf, acc, b, ((P - T) ** 2).sum(1)


def ece_w(conf, acc, b, w):
    tot = 0.0
    for k in range(BINS):
        sel = b == k
        ws = (w * sel).sum(-1)
        nz = ws > 0
        gap = np.where(nz, np.abs(((w * sel * acc).sum(-1) - (w * sel * conf).sum(-1))) / np.where(nz, ws, 1), 0)
        tot = tot + ws * gap
    return tot / w.sum(-1)


def evaluate(seeds, target, edges, field, boot):
    per, avg = [], defaultdict(list)
    for s, fields in sorted(seeds.items()):
        es, P, T = matrices(fields.get(field, {}), target, edges, field)
        if not es:
            continue
        conf, acc, b, br = per_edge(P, T)
        per.append({"seed": s, "n": len(es), "ece": float(ece_w(conf, acc, b, np.ones(len(es)))),
                    "brier": float(br.mean())})
    if not per:
        return None
    # seed-averaged probabilities on edges every seed covers
    common = set(edges)
    for s, fields in seeds.items():
        common &= set(fields.get(field, {}))
    avg_probs = {}
    for e in common:
        labs = {k for s in seeds for k in seeds[s][field][e]}
        avg_probs[e] = {k: float(np.mean([seeds[s][field][e].get(k, 0.0) for s in seeds])) for k in labs}
    es, P, T = matrices(avg_probs, target, sorted(common), field)
    conf, acc, b, br = per_edge(P, T)
    W = boot.W[:, [boot.idx(e) for e in es]].astype(float)
    bece, bbr = ece_w(conf, acc, b, W), (W * br).sum(1) / W.sum(1)
    out = {"n": per[0]["n"], "seeds": len(per),
           "ece_seed_mean": C.r4(np.mean([p["ece"] for p in per])), "brier_seed_mean": C.r4(np.mean([p["brier"] for p in per])),
           "ece_seed_sd": C.r4(np.std([p["ece"] for p in per], ddof=1)) if len(per) > 1 else None,
           "brier_seed_sd": C.r4(np.std([p["brier"] for p in per], ddof=1)) if len(per) > 1 else None,
           "seed_avg": {"n": len(es), "ece": C.r4(ece_w(conf, acc, b, np.ones(len(es)))), "ece_ci95": C.ci(bece),
                        "brier": C.r4(br.mean()), "brier_ci95": C.ci(bbr),
                        "mean_confidence": C.r4(conf.mean()), "accuracy": C.r4(acc.mean())}}
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    out_json = OUT / "calibration.json"
    if out_json.exists():
        raise SystemExit(f"refusing to overwrite {out_json}")
    cw = S.crosswalk()
    alias, cluster = S.components()
    _, v2b_raw, _, _ = C.references(cw)
    v2b, dropped = C.drop_unclear(v2b_raw)
    B1, B2 = sorted(v2b_raw["citation_function"]), sorted(v2b_raw["uptake_mode"])
    bootB = C.Boot(B1, cluster)
    probs = load_probs(cw, alias)
    res = {}
    for system, seeds in probs.items():
        for field in FIELDS:
            if not any(field in f for f in seeds.values()):
                continue
            r = evaluate(seeds, v2b[field], B1 if field in C.PASS1 else B2, field, bootB)
            if r:
                res[f"kevin_v2@B|{field}|{system}"] = r
    d3_edges, d3_excl = D3.load_edges()
    soft = {e["edge_id"]: {c: float(v) for c, v in zip(D3.CLASSES, D3.target(e, "soft")) if v > 0} for e in d3_edges}
    bootD = C.Boot(sorted(soft), cluster)
    for system, seeds in probs.items():
        r = evaluate(seeds, soft, sorted(soft), "citation_function", bootD)
        if r:
            res[f"d3_soft@B|citation_function|{system}"] = r
    meta = {"spec": "pre-registration §8", "bins": BINS, "n_boot": C.N_BOOT, "seed": C.SEED,
            "unclear_excluded_B": dropped, "d3_excluded": dict(d3_excl), "results": res}
    out_json.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    def fmt(r):
        sa = r["seed_avg"]
        sd = lambda x: f" ± {x}" if x is not None else ""
        return (f"{r['ece_seed_mean']}{sd(r['ece_seed_sd'])} | {r['brier_seed_mean']}{sd(r['brier_seed_sd'])} | "
                f"{sa['ece']} [{sa['ece_ci95'][0]}, {sa['ece_ci95'][1]}] | {sa['brier']} [{sa['brier_ci95'][0]}, "
                f"{sa['brier_ci95'][1]}] | {sa['mean_confidence']} | {sa['accuracy']}")
    lines = [f"| {k.split('|')[0]} | {k.split('|')[1]} | {k.split('|')[2]} | {v['n']} | {v['seeds']} | {fmt(v)} |"
             for k, v in res.items()]
    day = time.strftime("%Y-%m-%d")
    note = f"""# Calibration of systems with probabilities ({day})

**Registered analysis** (pre-registration §8, OSF osf.io/nfz9c). Script `analysis/experiments/32_calibration.py`
(rules in its docstring, fixed before the run); output `{out_json.relative_to(ROOT)}`. ECE: 15 equal-width bins,
confidence = max probability; Brier: multiclass sum of squares. Seed-averaged CIs: cluster bootstrap over
components, {C.N_BOOT:,} resamples, seed {C.SEED}. Lower is better for both. Sol and Qwen: no probabilities (n/a).
Reference `unclear` excluded: {dropped}. D3 soft-label edges: {len(soft)} (excluded {dict(d3_excl)}).

| Reference | Field | System | n | Seeds | ECE (seed mean ± sd) | Brier (seed mean ± sd) | ECE seed-avg [95% CI] | Brier seed-avg [95% CI] | Mean confidence | Accuracy |
|---|---|---|---|---|---|---|---|---|---|---|
""" + "\n".join(lines) + "\n"
    (ROOT / f"docs/results/{day}_calibration.md").write_text(note, encoding="utf-8")
    print(note)


if __name__ == "__main__":
    main()
