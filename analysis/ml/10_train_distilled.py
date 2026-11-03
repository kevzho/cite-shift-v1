#!/usr/bin/env python3
"""D1. Distilled multi-head encoder trained on hosted-LLM (GPT-6.1 Sol) silver labels.

One SciBERT backbone, one linear head per codebook-v2 field (Passes 1-2):
cited_contribution_type, citation_function, credits_contribution, uptake_mode, uptake_degree,
claim_dependent. Loss = sum of per-head cross-entropies; a head is skipped for an example whose
silver label is missing (Sol call invalid / content-filtered). Empty uptake_degree (mode none or
unclear) is the class `not_applicable`, as in the prompt schema.

Training data (silver only; never a human label):
  * Sol labels on canonical edges whose split is train or dev (analysis/ml/splits). Every frozen-split
    edge is excluded, which removes all v2 B/A/calibration edges, the v1 500 and the anchors; those
    are additionally excluded by id as a guard.
  * Sol labels on the 2,000-edge CHI sample (edges_chi_d1.csv), drawn disjoint from canonical papers.
Held-out silver: canonical test-split edges, scored against Sol only (distillation fidelity).
Never scored against any human label here. Predictions + probabilities are written for every frozen
edge, to be scored against Kevin's v2 labels after B/A coding (pre-registration §7 D1).

Text = "<cited title> [SEP] <citance 1> <citance 2> ..." (titles + citances, the evidence Sol saw,
minus the citing title to save length). Deterministic CPU mode, as 06_train_encoder.py.

Usage:
    python3 analysis/ml/10_train_distilled.py --seed 1 --out analysis/results/ml/d1_distilled
"""
from __future__ import annotations

import os

os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "4")

import argparse
import csv
import json
import random
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, logging as hf_logging

hf_logging.set_verbosity_error()

ROOT = Path(__file__).resolve().parents[2]
H = ROOT / "analysis/results/external_systems/hosted_llm_codebook_v2"
FIELDS = ["cited_contribution_type", "citation_function", "credits_contribution", "uptake_mode",
          "uptake_degree", "claim_dependent"]
LABELS = {
    "cited_contribution_type": ["empirical", "artifact", "methodological", "theoretical", "dataset",
                                "survey", "opinion", "unclear"],
    "citation_function": ["background", "uses_method", "applies_framework", "builds_on_finding",
                          "reuses_artifact", "compares_against", "other", "unclear"],
    "credits_contribution": ["specific", "generic", "unclear"],
    "uptake_mode": ["method", "artifact", "finding", "extension", "framing", "design_knowledge",
                    "contrast", "none", "unclear"],
    "uptake_degree": ["full", "partial", "not_applicable"],
    "claim_dependent": ["yes", "no", "unclear"],
}
BACKBONE = "allenai/scibert_scivocab_uncased"
CPU_THREADS = 4


def read(path):
    with open(path, newline="", encoding="utf-8") as h:
        return list(csv.DictReader(h))


def text_of(row):
    cits = [c.strip() for c in row["citances"].split("|||") if c.strip()]
    return f"{row.get('cited_title') or ''} [SEP] " + " ".join(cits)


def silver(row):
    out = {}
    for f in FIELDS:
        v = (row.get(f) or "").strip()
        if f == "uptake_degree" and not v and row.get("uptake_mode"):
            v = "not_applicable"
        out[f] = v if v in LABELS[f] else None
    return out


def guard_ids():
    ids = set()
    for p in (ROOT / "annotations/generated").glob("v2_*/*_key.csv"):
        ids |= {r["edge_id"] for r in read(p)}
    p = ROOT / "annotations/generated/expansion_2026-09-16/expansion_batch_kz.csv"
    ids |= {r["edge_id"] for r in csv.DictReader(open(p, encoding="utf-8-sig"))}
    for g in json.load(open(ROOT / "annotations/generated/codebook_v2_anchors.json")).values():
        ids |= {a["edge_id"] for a in g}
    return ids


class MultiHead(nn.Module):
    def __init__(self, init=None):
        super().__init__()
        # --init: start from a continued-pretrained backbone (D2: mlm / jepa); default = SciBERT as released
        self.backbone = AutoModel.from_pretrained(str(init) if init else BACKBONE)
        d = self.backbone.config.hidden_size
        self.drop = nn.Dropout(0.1)
        self.heads = nn.ModuleDict({f: nn.Linear(d, len(LABELS[f])) for f in FIELDS})

    def forward(self, ids, mask):
        dev = self.backbone.device
        ids, mask = ids.to(dev), mask.to(dev)
        h = self.drop(self.backbone(input_ids=ids, attention_mask=mask).last_hidden_state[:, 0])
        return {f: head(h) for f, head in self.heads.items()}


def batches(X, idx, size, pad_id):
    for i in range(0, len(idx), size):
        chunk = [X[j] for j in idx[i:i + size]]
        w = max(len(s) for s in chunk)
        ids = torch.full((len(chunk), w), pad_id, dtype=torch.long)
        m = torch.zeros((len(chunk), w), dtype=torch.long)
        for k, s in enumerate(chunk):
            ids[k, :len(s)] = torch.tensor(s)
            m[k, :len(s)] = 1
        yield idx[i:i + size], ids, m


@torch.no_grad()
def predict(model, X, pad_id, size=32):
    model.eval()
    out = {f: [] for f in FIELDS}
    for _, ids, m in batches(X, list(range(len(X))), size, pad_id):
        logits = model(ids, m)
        for f in FIELDS:
            out[f].append(torch.softmax(logits[f], -1).cpu().numpy())
    return {f: np.concatenate(v) for f, v in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--out", type=Path, default=ROOT / "analysis/results/ml/d1_distilled")
    ap.add_argument("--init", type=Path, default=None,
                    help="backbone checkpoint dir (save_pretrained) from analysis/ml/d2_jepa/pretrain.py; "
                         "default: SciBERT as released. Nothing else changes between arms.")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"],
                    help="cpu (default) keeps the deterministic CPU path used for D1")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    pred_path = a.out / f"d1_frozen_predictions_seed{a.seed}.csv"
    if pred_path.exists():
        raise SystemExit(f"refusing to overwrite {pred_path}")

    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    if a.device == "cuda":
        import os
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.cuda.manual_seed_all(a.seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(CPU_THREADS)

    split = {}
    for c in ("hci", "nature_oa"):
        for s in read(ROOT / f"analysis/ml/splits/{c}_splits.csv"):
            split[s["edge_id_v1"]] = s["split"]
    guard = guard_ids()
    canon_in = {r["edge_id"]: r for r in read(H / "input_canonical.csv")}
    canon_sol = {r["edge_id"]: r for r in read(H / "edges_canonical.csv")}
    chi_in = {r["edge_id"]: r for r in read(H / "input_chi_d1_2000.csv")}
    chi_sol = {r["edge_id"]: r for r in read(H / "edges_chi_d1.csv")}

    train, held, frozen = [], [], []
    for e, r in canon_in.items():
        sp = split.get(e)
        if sp == "frozen":
            frozen.append((e, r))
        elif e in guard:
            continue
        elif sp in ("train", "dev") and e in canon_sol:
            train.append((text_of(r), silver(canon_sol[e])))
        elif sp == "test" and e in canon_sol:
            held.append((text_of(r), silver(canon_sol[e])))
    for e, r in chi_in.items():
        if e in chi_sol:
            train.append((text_of(r), silver(chi_sol[e])))
    n_canon_train = len(train) - sum(1 for e in chi_in if e in chi_sol)

    tok = AutoTokenizer.from_pretrained(BACKBONE)
    pad = tok.pad_token_id
    enc = lambda texts: tok(texts, truncation=True, max_length=a.max_len)["input_ids"]
    Xtr = enc([t for t, _ in train])
    Ytr = {f: torch.tensor([LABELS[f].index(y[f]) if y[f] else -100 for _, y in train]) for f in FIELDS}

    model = MultiHead(a.init).to(a.device)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr)
    lossf = nn.CrossEntropyLoss(ignore_index=-100)
    rng = random.Random(a.seed)
    t0 = time.process_time(); w0 = time.time()
    for ep in range(a.epochs):
        model.train()
        order = list(range(len(Xtr)))
        rng.shuffle(order)
        tot = 0.0
        for idx, ids, m in batches(Xtr, order, a.batch_size, pad):
            out = model(ids, m)
            loss = sum(lossf(out[f], Ytr[f][idx].to(out[f].device)) for f in FIELDS)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss)
        print(f"epoch {ep + 1}/{a.epochs} loss {tot:.2f}  wall {time.time() - w0:.0f}s", flush=True)
    train_cpu = time.process_time() - t0

    # distillation fidelity on held-out silver (agreement with Sol only)
    Ph = predict(model, enc([t for t, _ in held]), pad)
    fid = {}
    for f in FIELDS:
        pairs = [(LABELS[f][int(np.argmax(Ph[f][i]))], y[f]) for i, (_, y) in enumerate(held) if y[f]]
        fid[f] = {"n": len(pairs), "agreement_with_sol": round(sum(p == s for p, s in pairs) / len(pairs), 4)}

    # frozen predictions (for later scoring vs Kevin v2), with inference cost per edge
    t1 = time.process_time()
    Pf = predict(model, enc([text_of(r) for _, r in frozen]), pad)
    infer_cpu = time.process_time() - t1
    with pred_path.open("w", newline="", encoding="utf-8") as h:
        cols = ["edge_id", "corpus", "seed"] + FIELDS + [f"{f}__p_{l}" for f in FIELDS for l in LABELS[f]]
        w = csv.DictWriter(h, fieldnames=cols)
        w.writeheader()
        for i, (e, r) in enumerate(frozen):
            row = {"edge_id": e, "corpus": r["corpus"], "seed": a.seed}
            for f in FIELDS:
                row[f] = LABELS[f][int(np.argmax(Pf[f][i]))]
                for k, l in enumerate(LABELS[f]):
                    row[f"{f}__p_{l}"] = round(float(Pf[f][i][k]), 6)
            w.writerow(row)
    meta = {"seed": a.seed, "backbone": BACKBONE, "init": str(a.init) if a.init else None, "device": a.device, "epochs": a.epochs, "lr": a.lr, "batch_size": a.batch_size,
            "max_len": a.max_len, "n_train": len(train), "n_train_canonical": n_canon_train,
            "n_train_chi": len(train) - n_canon_train, "n_heldout_silver": len(held), "n_frozen": len(frozen),
            "train_label_counts": {f: dict(Counter(y[f] for _, y in train if y[f])) for f in FIELDS},
            "heldout_fidelity_vs_sol": fid, "train_cpu_s": round(train_cpu, 1),
            "train_wall_s": round(time.time() - w0, 1),
            "inference_cpu_s_per_edge": round(infer_cpu / max(1, len(frozen)), 4),
            "deterministic": torch.are_deterministic_algorithms_enabled(), "threads": CPU_THREADS}
    (a.out / f"d1_meta_seed{a.seed}.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: meta[k] for k in ("n_train", "n_heldout_silver", "n_frozen", "heldout_fidelity_vs_sol",
                                           "train_wall_s", "inference_cpu_s_per_edge")}, indent=1))


if __name__ == "__main__":
    main()
