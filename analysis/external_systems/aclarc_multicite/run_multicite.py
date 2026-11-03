"""Run the released MultiCite multi-label intent classifiers (Lauscher et al., NAACL 2022) locally.

Models (official AllenAI releases on the Hugging Face Hub, MIT licence, pinned revisions):
  scibert        allenai/multicite-multilabel-scibert        @ 7696929cf8ba26c21cc1da0daeb76010575dff80
  roberta-large  allenai/multicite-multilabel-roberta-large  @ 4ed0c81f083f288375da6b8c86a737fb2aaf941a

Decision rule = the paper's / repo's: sigmoid(logit) > 0.5 per class (multi-label; empty set possible).
Input = MultiCite format: plain text with the target marker wrapped in <cite>...</cite> (see markers.py).

Two edge-level input modes are written side by side (both deterministic):
  per_citance  each citance is one instance (MultiCite 1-sentence setting); edge labels = union over
               citances, edge prob per class = max over citances.
  joined       all citances of the edge concatenated (in CSV order, space-joined) as ONE multi-sentence
               instance. NB: our citances are not contiguous, unlike MultiCite's gold contexts.

Runs on CPU, eval mode, no dropout, no sampling. Nothing leaves the machine
(HF_HUB_OFFLINE=1 is set after the one-time pinned download).

Usage:
  python run_multicite.py --model scibert --tag run1
  python run_multicite.py --model scibert --throughput 200     # timing only, outputs discarded
"""
import os as _os
from pathlib import Path as _Path
_CITATION_ROOT = _os.environ.get("CITATION_ROOT", str(_Path(__file__).resolve().parents[3]))
_CITATION_VENVS = _os.environ.get("CITATION_VENVS", _os.path.expanduser("~/.cache/citation_corpus/venvs"))
import argparse, csv, hashlib, json, os, random, sys, time

os.environ.setdefault("HF_HOME", _CITATION_VENVS + "/aclarc_multicite/hf_home")
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import markers

REPO = _CITATION_ROOT
TEST = f"{REPO}/analysis/results/external_systems/test_edges.csv"
OUT = f"{REPO}/analysis/results/external_systems/aclarc_multicite"
CHI = f"{REPO}/external_data/chi_chiea_grobid_2026-09-15/citances/chi_chiea_citances.jsonl"
MODELS = {
    "scibert": ("allenai/multicite-multilabel-scibert", "7696929cf8ba26c21cc1da0daeb76010575dff80"),
    "roberta-large": ("allenai/multicite-multilabel-roberta-large", "4ed0c81f083f288375da6b8c86a737fb2aaf941a"),
}
# Proposed crosswalk to codebook v2 (NOT validated; for interpretation only)
TO_V2 = {
    "uses": {"citation_function": "uses_method|reuses_artifact", "uptake_mode": "method|artifact"},
    "extends": {"citation_function": "uses_method|reuses_artifact", "uptake_mode": "extension"},
    "motivation": {"citation_function": "background", "uptake_mode": "framing"},
    "differences": {"citation_function": "compares_against", "uptake_mode": "contrast"},
    "similarities": {"citation_function": "compares_against", "uptake_mode": "none|finding (unresolved)"},
    "background": {"citation_function": "background", "uptake_mode": "none (framing/finding/design_knowledge not separable)"},
    "future_work": {"citation_function": "other", "uptake_mode": "none"},
}


def set_determinism(threads):
    random.seed(0); np.random.seed(0); torch.manual_seed(0)
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)


def load(model_key):
    name, rev = MODELS[model_key]
    tok = AutoTokenizer.from_pretrained(name, revision=rev)
    model = AutoModelForSequenceClassification.from_pretrained(name, revision=rev)
    model.eval()
    labels = [model.config.id2label[i] for i in range(model.config.num_labels)]
    return tok, model, labels


@torch.no_grad()
def predict(tok, model, texts, batch):
    probs = []
    for i in range(0, len(texts), batch):
        enc = tok(texts[i:i + batch], truncation=True, max_length=512, padding=True, return_tensors="pt")
        logits = model(**enc).logits
        probs.append(torch.sigmoid(logits).cpu().numpy().astype(np.float64))
    return np.concatenate(probs, 0) if probs else np.zeros((0, model.config.num_labels))


def build_instances(rows):
    inst = []  # (edge_idx, kind, text)
    meta = []
    for k, r in enumerate(rows):
        cs = [c.strip() for c in r["citances"].split("|||") if c.strip()]
        marked, res, tnums = markers.mark_edge(cs, r["corpus"])
        meta.append({"marked": marked, "resolution": res, "target_numbers": tnums})
        for j, m in enumerate(marked):
            inst.append((k, f"c{j}", m))
        inst.append((k, "joined", " ".join(marked)))
    return inst, meta


def run_test(args):
    rows = list(csv.DictReader(open(args.edges, encoding="utf-8")))
    out_path = f"{OUT}/outputs/multicite_{args.model}_{args.tag}.jsonl"
    if os.path.exists(out_path):
        sys.exit(f"refusing to overwrite {out_path}")
    tok, model, labels = load(args.model)
    inst, meta = build_instances(rows)
    t0 = time.perf_counter()
    P = predict(tok, model, [t for _, _, t in inst], args.batch)
    dt = time.perf_counter() - t0
    by_edge = {}
    for (k, kind, _), p in zip(inst, P):
        by_edge.setdefault(k, {})[kind] = p
    with open(out_path, "w") as f:
        for k, r in enumerate(rows):
            d = by_edge[k]
            cit = [d[f"c{j}"] for j in range(len(meta[k]["marked"]))]
            per = [{"probs": {l: round(float(p[i]), 6) for i, l in enumerate(labels)},
                    "labels": [l for i, l in enumerate(labels) if p[i] > 0.5]} for p in cit]
            mx = np.max(np.stack(cit), 0)
            union = [l for i, l in enumerate(labels) if mx[i] > 0.5]
            jp = d["joined"]
            rec = {
                "edge_id": r["edge_id"], "corpus": r["corpus"], "model": MODELS[args.model][0],
                "revision": MODELS[args.model][1], "n_citances": len(cit),
                "marker_resolution": meta[k]["resolution"], "target_numbers": meta[k]["target_numbers"],
                "per_citance": per,
                "edge_per_citance_union_labels": union,
                "edge_per_citance_max_probs": {l: round(float(mx[i]), 6) for i, l in enumerate(labels)},
                "edge_joined_labels": [l for i, l in enumerate(labels) if jp[i] > 0.5],
                "edge_joined_probs": {l: round(float(jp[i]), 6) for i, l in enumerate(labels)},
                "raw_probs_sha1": hashlib.sha1(np.stack(cit + [jp]).tobytes()).hexdigest(),
            }
            f.write(json.dumps(rec) + "\n")
    with open(f"{OUT}/inputs/marked_inputs_{args.tag}.jsonl", "w") as f:
        for k, r in enumerate(rows):
            f.write(json.dumps({"edge_id": r["edge_id"], **meta[k]}) + "\n")
    print(f"wrote {out_path}; {len(inst)} instances in {dt:.2f}s ({len(inst)/dt:.2f} inst/s)")


def run_throughput(args):
    lines = open(CHI, encoding="utf-8").read().splitlines()
    n_total = len(lines)
    random.Random(12345).shuffle(lines)
    sample = [json.loads(l) for l in lines[:args.throughput]]
    del lines
    tok, model, labels = load(args.model)
    t0 = time.perf_counter()
    texts = []
    for s in sample:
        cs = [c for c in s["citances"] if c.strip()]
        if not cs:
            continue
        marked, _, _ = markers.mark_edge(cs, "hci")
        texts.extend(marked + [" ".join(marked)])
    P = predict(tok, model, texts, args.batch)
    dt = time.perf_counter() - t0
    del P  # outputs are NOT kept
    per_edge = dt / len(sample)
    print(json.dumps({"model": args.model, "edges": len(sample), "instances": len(texts), "seconds": round(dt, 2),
                      "edges_per_s": round(len(sample) / dt, 3), "instances_per_s": round(len(texts) / dt, 3),
                      "extrap_2937_edges_min": round(per_edge * 2937 / 60, 1),
                      "extrap_all_edges_h": round(per_edge * n_total / 3600, 2), "jsonl_lines": n_total,
                      "threads": args.threads, "batch": args.batch}))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(MODELS), required=True)
    ap.add_argument("--tag", default="run1")
    ap.add_argument("--edges", default=TEST, help="CSV with edge_id, corpus, citances ('|||'-joined)")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--throughput", type=int, default=0)
    a = ap.parse_args()
    set_determinism(a.threads)
    run_throughput(a) if a.throughput else run_test(a)
