#!/usr/bin/env python3
"""No-model baseline (pre-registration §4): majority class per field from the TRAINING split.

Majority is taken over canonical labels of train-split edges (analysis/ml/splits, both corpora pooled),
in v2 names via the same normalisation as the scorer, and predicted for every canonical edge.
Output: the scorer's long format (edge_id, field, label, system), one row per edge x field.
Fields: citation_function, cited_contribution_type, realization (canonical-eligible fields only).
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("scorer", ROOT / "analysis/experiments/22_score_systems.py")
S = importlib.util.module_from_spec(spec)
spec.loader.exec_module(S)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    if a.out.exists():
        raise SystemExit(f"refusing to overwrite {a.out}")
    cw = S.crosswalk()
    ref = S.ref_canonical(cw)
    train = set()
    for path in S.SPLITS.values():
        train |= {r["edge_id_v1"] for r in S.read_csv(path) if r["split"] == "train"}
    majority = {}
    for field, labels in ref.items():
        c = Counter(v for e, v in labels.items() if e in train)
        majority[field] = c.most_common(1)[0][0]
    rows = [{"edge_id": e, "field": f, "label": majority[f], "system": "no_model_baseline"}
            for f, labels in ref.items() for e in labels]
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open("w", newline="", encoding="utf-8") as h:
        w = csv.DictWriter(h, fieldnames=["edge_id", "field", "label", "system"])
        w.writeheader()
        w.writerows(rows)
    print(json.dumps({"train_edges": len(train), "majority": majority, "rows": len(rows)}, indent=1))


if __name__ == "__main__":
    main()
