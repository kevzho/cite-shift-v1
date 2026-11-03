#!/usr/bin/env python3
"""D3 (registered, pre-registration §7 D3 and §8): does training on human disagreement improve calibration?

Data: the 300 sample-B edges with two human records on `citation_function`:
  * Kevin v2 Pass 1 (B_stratum_pass1 + B_rest_pass1; stratum Pass 1 is valid, only stratum Pass 2 was excluded),
  * the canonical record, mapped to v2 names by the mechanical crosswalk (annotations/label_crosswalk.csv).
`unclear` (either record) and Kevin `other` (no canonical target: crosswalk exclude_nonmappable) are excluded (§10).

Arms (identical SciBERT encoder + one head; only the training target differs):
  * soft          : per-edge distribution from the two human labels (agree -> one-hot; disagree -> 0.5 / 0.5)
  * hard_kevin    : Kevin v2 label (the primary human reference, §3) -- the registered "hard" arm
  * hard_canonical: canonical label (reported as a sensitivity arm; not registered as primary)
Cross-validated by citation-graph component (analysis/ml/splits component_key), 5 folds, 3 seeds.

Primary (§7/§8): ECE (15 equal-width bins) and Brier score against the SOFT label distribution
  (ECE: confidence = max predicted probability; accuracy = soft-label mass on the predicted class).
Secondary: macro-F1 (classes with reference support >= 5) against each human record.
Uncertainty: per-edge probabilities averaged over seeds; soft-minus-hard differences with a cluster bootstrap over
components (10,000 resamples, seed 20261003). Verdict: soft improves calibration if the 95% CI of BOTH the ECE and
the Brier difference (soft - hard_kevin) lies below 0.

CPU deterministic mode, as 06/10. Outputs refuse to overwrite.
Usage: python3 analysis/ml/11_d3_soft_labels.py [--smoke]
"""
from __future__ import annotations

import os

os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "4")

import argparse
import csv
import json
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, logging as hf_logging

hf_logging.set_verbosity_error()

ROOT = Path(__file__).resolve().parents[2]
D = ROOT / "annotations/generated/v2_coding_2026-10-02"
H = ROOT / "analysis/results/external_systems/hosted_llm_codebook_v2"
OUT = ROOT / "analysis/results/ml/d3_2026-10"
BACKBONE = "allenai/scibert_scivocab_uncased"
CLASSES = ["background", "uses_method", "applies_framework", "builds_on_finding", "reuses_artifact",
           "compares_against"]
ARMS = ["soft", "hard_kevin", "hard_canonical"]


def read(p, enc="utf-8"):
    with open(p, newline="", encoding=enc) as h:
        return list(csv.DictReader(h))


def crosswalk():
    m = {}
    for r in read(ROOT / "annotations/label_crosswalk.csv"):
        if r["dimension"] == "citance_function" and r["status"] == "mechanical" and r["canonical_value"]:
            m[r["canonical_value"].strip()] = r["codebook_label"].strip()
    m["motivates_background"] = "motivates_background"
    return m


def load_edges():
    kevin = {}
    for f in ("B_stratum_pass1", "B_rest_pass1"):
        k = {r["item"]: r["edge_id"] for r in read(D / f"{f}_key.csv")}
        for r in read(D / f"{f}.csv", "utf-8-sig"):
            kevin[k[r["item"]]] = r["citation_function"].strip()
    cw = crosswalk()
    v1name = {"motivates_background": "background"}           # v1 -> v2 rename (codebook v2 §10)
    canon_raw = {}
    import hashlib
    for c in ("hci", "nature_oa"):
        for r in read(ROOT / f"analysis/derived/{c}_contribution_coding_human_reviewed.csv", "utf-8-sig"):
            e = hashlib.sha256("\x1f".join((c, r["cited_id"].strip(), r["citing_id"].strip())).encode()).hexdigest()[:20]
            canon_raw[e] = r["citance_function"].strip()
    comp = {}
    for c in ("hci", "nature_oa"):
        for s in read(ROOT / f"analysis/ml/splits/{c}_splits.csv"):
            comp[s["edge_id_v1"]] = f"{c}:{s['component_key']}"
    text = {r["edge_id"]: r for r in read(H / "input_canonical.csv")}
    edges, excluded = [], Counter()
    for e, kv in sorted(kevin.items()):
        cv = v1name.get(cw.get(canon_raw.get(e, ""), ""), cw.get(canon_raw.get(e, ""), ""))
        if kv not in CLASSES:
            excluded[f"kevin_{kv or 'blank'}"] += 1
            continue
        if cv not in CLASSES:
            excluded[f"canonical_{canon_raw.get(e) or 'missing'}"] += 1
            continue
        r = text[e]
        cits = [x.strip() for x in r["citances"].split("|||") if x.strip()]
        edges.append({"edge_id": e, "corpus": r["corpus"], "comp": comp[e], "kevin": kv, "canon": cv,
                      "text": f"{r.get('cited_title') or ''} [SEP] " + " ".join(cits)})
    return edges, excluded


def folds_by_component(edges, k, seed):
    by = defaultdict(list)
    for i, e in enumerate(edges):
        by[e["comp"]].append(i)
    keys = sorted(by)
    random.Random(seed).shuffle(keys)
    keys.sort(key=lambda c: -len(by[c]))                      # largest first, greedy balance
    load, fold = [0] * k, {}
    for c in keys:
        j = min(range(k), key=lambda t: load[t])
        fold[c] = j
        load[j] += len(by[c])
    return [fold[e["comp"]] for e in edges]


def target(e, arm):
    t = np.zeros(len(CLASSES), dtype=np.float32)
    if arm == "soft":
        t[CLASSES.index(e["kevin"])] += 0.5
        t[CLASSES.index(e["canon"])] += 0.5
    else:
        t[CLASSES.index(e["kevin"] if arm == "hard_kevin" else e["canon"])] = 1.0
    return t


class Head(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(BACKBONE)
        self.drop = nn.Dropout(0.1)
        self.out = nn.Linear(self.backbone.config.hidden_size, len(CLASSES))

    def forward(self, ids, mask):
        return self.out(self.drop(self.backbone(input_ids=ids, attention_mask=mask).last_hidden_state[:, 0]))


def batches(X, idx, size, pad):
    for i in range(0, len(idx), size):
        chunk = [X[j] for j in idx[i:i + size]]
        w = max(len(s) for s in chunk)
        ids = torch.full((len(chunk), w), pad, dtype=torch.long)
        m = torch.zeros((len(chunk), w), dtype=torch.long)
        for k, s in enumerate(chunk):
            ids[k, :len(s)] = torch.tensor(s)
            m[k, :len(s)] = 1
        yield idx[i:i + size], ids, m


def fit_predict(X, T, tr, te, pad, a, seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    model = Head()
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr)
    rng = random.Random(seed)
    Tt = torch.tensor(T)
    for _ in range(a.epochs):
        model.train()
        order = list(tr)
        rng.shuffle(order)
        for idx, ids, m in batches(X, order, a.batch_size, pad):
            logp = torch.log_softmax(model(ids, m), -1)
            loss = -(Tt[idx] * logp).sum(-1).mean()           # cross-entropy with a (possibly soft) target
            opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    P = np.zeros((len(te), len(CLASSES)), dtype=np.float64)
    with torch.no_grad():
        for idx, ids, m in batches(X, list(te), 32, pad):
            pos = [te.index(i) for i in idx]
            P[pos] = torch.softmax(model(ids, m), -1).numpy()
    return P


def ece(P, S, bins=15):
    conf = P.max(1)
    pred = P.argmax(1)
    acc = S[np.arange(len(S)), pred]                           # soft-label mass on the predicted class
    edges = np.linspace(0, 1, bins + 1)
    tot = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
        if sel.any():
            tot += sel.mean() * abs(acc[sel].mean() - conf[sel].mean())
    return float(tot)


def brier(P, S):
    return float(((P - S) ** 2).sum(1).mean())


def macro_f1(pred, ref, min_support=5):
    sup = Counter(ref)
    cls = [c for c in CLASSES if sup[c] >= min_support]
    f1s = []
    for c in cls:
        tp = sum(p == c and r == c for p, r in zip(pred, ref))
        fp = sum(p == c and r != c for p, r in zip(pred, ref))
        fn = sum(p != c and r == c for p, r in zip(pred, ref))
        f1s.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f1s)) if f1s else float("nan"), [c for c in CLASSES if c not in cls]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="tiny CPU check: 40 edges, 2 folds, 1 seed, 1 epoch")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seeds", default="1,2,3")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--n-boot", type=int, default=10000)
    a = ap.parse_args()
    seeds = [int(s) for s in a.seeds.split(",")]
    out = OUT / (f"smoke_{time.strftime('%Y%m%d_%H%M%S')}" if a.smoke else "main")
    if (out / "d3_results.json").exists():
        raise SystemExit(f"refusing to overwrite {out}")
    out.mkdir(parents=True, exist_ok=True)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(4)

    edges, excluded = load_edges()
    if a.smoke:
        edges, a.folds, seeds, a.epochs, a.max_len, a.n_boot = edges[:40], 2, [1], 1, 64, 200
    tok = AutoTokenizer.from_pretrained(BACKBONE)
    pad = tok.pad_token_id
    X = tok([e["text"] for e in edges], truncation=True, max_length=a.max_len)["input_ids"]
    S = np.stack([target(e, "soft") for e in edges]).astype(np.float64)
    fold = folds_by_component(edges, a.folds, 20261003)
    probs = {arm: [] for arm in ARMS}                            # per seed, full out-of-fold matrix
    t0 = time.time()
    for arm in ARMS:
        T = np.stack([target(e, arm) for e in edges])
        for seed in seeds:
            P = np.zeros((len(edges), len(CLASSES)))
            for k in range(a.folds):
                tr = [i for i in range(len(edges)) if fold[i] != k]
                te = [i for i in range(len(edges)) if fold[i] == k]
                P[te] = fit_predict(X, T, tr, te, pad, a, seed)
                print(f"{arm} seed {seed} fold {k + 1}/{a.folds} done ({time.time() - t0:.0f}s)", flush=True)
            probs[arm].append(P)
            np.save(out / f"oof_{arm}_seed{seed}.npy", P)

    res = {"n_edges": len(edges), "excluded": dict(excluded), "folds": a.folds, "seeds": seeds,
           "epochs": a.epochs, "agreement_rate_kevin_canonical": round(float(np.mean([e["kevin"] == e["canon"] for e in edges])), 4),
           "class_counts_kevin": dict(Counter(e["kevin"] for e in edges)),
           "class_counts_canonical": dict(Counter(e["canon"] for e in edges)), "arms": {}}
    mean_P = {arm: np.mean(probs[arm], axis=0) for arm in ARMS}
    for arm in ARMS:
        per_seed = [{"ece": ece(P, S), "brier": brier(P, S)} for P in probs[arm]]
        pred = [CLASSES[i] for i in mean_P[arm].argmax(1)]
        f1k, wk = macro_f1(pred, [e["kevin"] for e in edges])
        f1c, wc = macro_f1(pred, [e["canon"] for e in edges])
        res["arms"][arm] = {
            "ece_seed_mean": round(float(np.mean([p["ece"] for p in per_seed])), 4),
            "ece_seed_sd": round(float(np.std([p["ece"] for p in per_seed], ddof=1)) if len(per_seed) > 1 else 0.0, 4),
            "brier_seed_mean": round(float(np.mean([p["brier"] for p in per_seed])), 4),
            "brier_seed_sd": round(float(np.std([p["brier"] for p in per_seed], ddof=1)) if len(per_seed) > 1 else 0.0, 4),
            "ece_seed_avg_probs": round(ece(mean_P[arm], S), 4), "brier_seed_avg_probs": round(brier(mean_P[arm], S), 4),
            "macro_f1_vs_kevin": round(f1k, 4), "macro_f1_vs_canonical": round(f1c, 4),
            "classes_withheld_kevin": wk, "classes_withheld_canonical": wc}
    # cluster bootstrap over components for soft - hard differences (seed-averaged probabilities)
    by = defaultdict(list)
    for i, e in enumerate(edges):
        by[e["comp"]].append(i)
    keys = sorted(by)
    rng = random.Random(20261003)
    diffs = {h: {"ece": [], "brier": []} for h in ("hard_kevin", "hard_canonical")}
    for _ in range(a.n_boot):
        idx = [i for c in rng.choices(keys, k=len(keys)) for i in by[c]]
        Sb = S[idx]
        es, bs = ece(mean_P["soft"][idx], Sb), brier(mean_P["soft"][idx], Sb)
        for h in diffs:
            diffs[h]["ece"].append(es - ece(mean_P[h][idx], Sb))
            diffs[h]["brier"].append(bs - brier(mean_P[h][idx], Sb))
    res["soft_minus_hard"] = {}
    for h in diffs:
        d = {}
        for m in ("ece", "brier"):
            v = sorted(diffs[h][m])
            point = (ece if m == "ece" else brier)(mean_P["soft"], S) - (ece if m == "ece" else brier)(mean_P[h], S)
            d[m] = {"diff": round(point, 4), "ci95": [round(v[int(0.025 * len(v))], 4), round(v[int(0.975 * len(v)) - 1], 4)]}
        res["soft_minus_hard"][h] = d
    r = res["soft_minus_hard"]["hard_kevin"]
    res["verdict_vs_registered_hard_arm"] = (
        "SOFT IMPROVES CALIBRATION" if r["ece"]["ci95"][1] < 0 and r["brier"]["ci95"][1] < 0 else
        "NO CLEAR IMPROVEMENT" if r["ece"]["ci95"][0] <= 0 <= r["ece"]["ci95"][1] or r["brier"]["ci95"][0] <= 0 <= r["brier"]["ci95"][1] else
        "MIXED / SOFT WORSE")
    res["wall_s"] = round(time.time() - t0, 1)
    res["smoke"] = a.smoke
    (out / "d3_results.json").write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: res[k] for k in ("n_edges", "excluded", "agreement_rate_kevin_canonical", "soft_minus_hard",
                                          "verdict_vs_registered_hard_arm", "wall_s")}, indent=1))


if __name__ == "__main__":
    main()
