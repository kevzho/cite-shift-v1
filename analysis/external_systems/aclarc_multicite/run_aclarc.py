"""Run the released ACL-ARC scaffold model (Cohan et al., NAACL 2019; allenai/scicite aclarc.tar.gz)
through the pure-PyTorch port in aclarc_scaffold.py, on the 20 test edges.

The model is single-sentence, single-label (softmax over Background, Uses, CompareOrContrast,
Motivation, Extends, Future). It has no target-citation marker: input is the raw citance text,
exactly as the scicite predictor passes `text`. Each citance is scored separately; edge-level
fields are aggregations of the per-citance outputs (no thresholds tuned, no labels used):
  edge_label_set     set of per-citance argmax labels
  edge_mean_probs    mean of per-citance probability vectors; edge_mean_argmax = its argmax

Usage:
  python run_aclarc.py --tag run1
  python run_aclarc.py --throughput 200     # timing only, outputs discarded
"""
import os as _os
from pathlib import Path as _Path
_CITATION_ROOT = _os.environ.get("CITATION_ROOT", str(_Path(__file__).resolve().parents[3]))
_CITATION_VENVS = _os.environ.get("CITATION_VENVS", _os.path.expanduser("~/.cache/citation_corpus/venvs"))
import argparse, csv, hashlib, json, os, random, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from aclarc_scaffold import AclArcClassifier

REPO = _CITATION_ROOT
TEST = f"{REPO}/analysis/results/external_systems/test_edges.csv"
OUT = f"{REPO}/analysis/results/external_systems/aclarc_multicite"
CHI = f"{REPO}/external_data/chi_chiea_grobid_2026-09-15/citances/chi_chiea_citances.jsonl"
ARCHIVE = _CITATION_VENVS + "/aclarc_multicite/models/scicite_aclarc"
# Proposed crosswalk to codebook v2 (NOT validated; for interpretation only)
TO_V2 = {
    "Uses": {"citation_function": "uses_method|reuses_artifact", "uptake_mode": "method|artifact"},
    "Extends": {"citation_function": "uses_method|reuses_artifact", "uptake_mode": "extension"},
    "Motivation": {"citation_function": "background", "uptake_mode": "framing"},
    "CompareOrContrast": {"citation_function": "compares_against", "uptake_mode": "contrast (if differences) | none"},
    "Background": {"citation_function": "background", "uptake_mode": "none (framing/finding/design_knowledge not separable)"},
    "Future": {"citation_function": "other", "uptake_mode": "none"},
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="run1")
    ap.add_argument("--edges", default=TEST, help="CSV with edge_id, corpus, citances ('|||'-joined)")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--throughput", type=int, default=0)
    a = ap.parse_args()
    random.seed(0); np.random.seed(0); torch.manual_seed(0)
    torch.set_num_threads(a.threads); torch.use_deterministic_algorithms(True)
    clf = AclArcClassifier(ARCHIVE)

    if a.throughput:
        lines = open(CHI, encoding="utf-8").read().splitlines()
        n_total = len(lines)
        random.Random(12345).shuffle(lines)
        sample = [json.loads(l) for l in lines[:a.throughput]]
        del lines
        t0 = time.perf_counter(); n = 0
        for s in sample:
            for c in s["citances"]:
                if c.strip():
                    clf.predict(c); n += 1   # outputs NOT kept
        dt = time.perf_counter() - t0
        per_edge = dt / len(sample)
        print(json.dumps({"model": "scicite-aclarc", "edges": len(sample), "instances": n, "seconds": round(dt, 2),
                          "edges_per_s": round(len(sample) / dt, 3), "instances_per_s": round(n / dt, 3),
                          "extrap_2937_edges_min": round(per_edge * 2937 / 60, 1),
                          "extrap_all_edges_h": round(per_edge * n_total / 3600, 2), "jsonl_lines": n_total,
                          "threads": a.threads}))
        return

    rows = list(csv.DictReader(open(a.edges, encoding="utf-8")))
    out_path = f"{OUT}/outputs/aclarc_scaffold_{a.tag}.jsonl"
    if os.path.exists(out_path):
        sys.exit(f"refusing to overwrite {out_path}")
    t0 = time.perf_counter(); n = 0
    with open(out_path, "w") as f:
        for r in rows:
            cs = [c.strip() for c in r["citances"].split("|||") if c.strip()]
            per = [clf.predict(c) for c in cs]
            n += len(cs)
            P = np.array([[p["probs"][l] for l in clf.labels] for p in per])
            mean = P.mean(0)
            rec = {
                "edge_id": r["edge_id"], "corpus": r["corpus"], "model": "allenai/scicite ACL-ARC scaffold (aclarc.tar.gz)",
                "n_citances": len(cs),
                "per_citance": [{"label": p["label"], "probs": {k: round(v, 6) for k, v in p["probs"].items()},
                                 "n_tokens": len(p["tokens"])} for p in per],
                "edge_label_set": sorted({p["label"] for p in per}, key=clf.labels.index),
                "edge_mean_probs": {l: round(float(mean[i]), 6) for i, l in enumerate(clf.labels)},
                "edge_mean_argmax": clf.labels[int(np.argmax(mean))],
                "raw_logits_sha1": hashlib.sha1(np.array([p["logits"] for p in per]).tobytes()).hexdigest(),
            }
            f.write(json.dumps(rec) + "\n")
    dt = time.perf_counter() - t0
    print(f"wrote {out_path}; {n} citances in {dt:.1f}s ({n/dt:.2f}/s)")


if __name__ == "__main__":
    main()
