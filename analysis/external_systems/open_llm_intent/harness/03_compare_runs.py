"""Determinism check: are run1 and run2 outputs byte-identical (citance- and edge-level CSVs)?
usage: python 03_compare_runs.py --dataset scicite"""
import argparse
import hashlib
import json

import pandas as pd

OUT = "/Volumes/kz_extended/SALT-UIUC/citation_iConf/citation_corpus/analysis/results/external_systems/open_llm_intent/outputs"
ap = argparse.ArgumentParser()
ap.add_argument("--dataset", default="scicite")
a = ap.parse_args()
tag = "sc" if a.dataset == "scicite" else "aa"
res = {}
for kind in ("citance", "edge"):
    f1, f2 = (f"{OUT}/{tag}_{kind}_preds_run{r}.csv" for r in (1, 2))
    h1, h2 = (hashlib.sha256(open(f, "rb").read()).hexdigest() for f in (f1, f2))
    d1, d2 = pd.read_csv(f1), pd.read_csv(f2)
    lab = "predicted_label" if kind == "citance" else "citance_labels"
    res[kind] = {"sha256_run1": h1, "sha256_run2": h2, "byte_identical": h1 == h2,
                 "label_identical": bool((d1[lab].astype(str) == d2[lab].astype(str)).all()),
                 "n_rows": len(d1)}
print(json.dumps(res, indent=1))
json.dump(res, open(f"{OUT}/{tag}_determinism.json", "w"), indent=1)
