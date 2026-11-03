"""Merge raw per-edge outputs of both systems into one CSV (no human labels are read or written),
and verify run1 == run2 byte-for-byte. Usage: python make_table.py"""
import os as _os
from pathlib import Path as _Path
_CITATION_ROOT = _os.environ.get("CITATION_ROOT", str(_Path(__file__).resolve().parents[3]))
import csv, filecmp, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

OUT = _CITATION_ROOT + "/analysis/results/external_systems/aclarc_multicite/outputs"
MC_MAP = {"uses": "method|artifact", "extends": "extension", "motivation": "framing", "differences": "contrast"}
AA_MAP = {"Uses": "method|artifact", "Extends": "extension", "Motivation": "framing"}


def load(p):
    return {d["edge_id"]: d for d in map(json.loads, open(p))}


det = {}
for stem in ("multicite_scibert", "aclarc_scaffold"):
    det[stem] = filecmp.cmp(f"{OUT}/{stem}_run1.jsonl", f"{OUT}/{stem}_run2.jsonl", shallow=False)
mc, aa = load(f"{OUT}/multicite_scibert_run1.jsonl"), load(f"{OUT}/aclarc_scaffold_run1.jsonl")
cols = ["edge_id", "corpus", "n_citances", "mc_marker_resolution", "mc_per_citance_labels", "mc_union_labels",
        "mc_joined_labels", "mc_max_probs", "mc_uptake_signal", "aa_per_citance_labels", "aa_mean_argmax",
        "aa_mean_probs", "aa_uptake_signal"]
with open(f"{OUT}/per_edge_outputs.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=cols)
    w.writeheader()
    for eid, m in mc.items():
        a = aa[eid]
        w.writerow({
            "edge_id": eid, "corpus": m["corpus"], "n_citances": m["n_citances"],
            "mc_marker_resolution": "|".join(m["marker_resolution"]),
            "mc_per_citance_labels": " || ".join(",".join(c["labels"]) or "(none>0.5)" for c in m["per_citance"]),
            "mc_union_labels": ",".join(m["edge_per_citance_union_labels"]) or "(none>0.5)",
            "mc_joined_labels": ",".join(m["edge_joined_labels"]) or "(none>0.5)",
            "mc_max_probs": json.dumps({k: round(v, 3) for k, v in m["edge_per_citance_max_probs"].items()}),
            "mc_uptake_signal": ",".join(sorted({MC_MAP[l] for l in m["edge_per_citance_union_labels"] if l in MC_MAP})) or "none",
            "aa_per_citance_labels": " || ".join(c["label"] for c in a["per_citance"]),
            "aa_mean_argmax": a["edge_mean_argmax"],
            "aa_mean_probs": json.dumps({k: round(v, 3) for k, v in a["edge_mean_probs"].items()}),
            "aa_uptake_signal": ",".join(sorted({AA_MAP[l] for l in a["edge_label_set"] if l in AA_MAP})) or "none",
        })
print(json.dumps({"identical_run1_run2": det, "edges": len(mc)}))
