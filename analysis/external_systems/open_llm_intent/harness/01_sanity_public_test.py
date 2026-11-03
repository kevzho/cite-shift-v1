"""Install sanity check on the PUBLIC benchmark test split (not project data): run the local Q4_K_M model with the
paper's fine-tuned zero-shot SP3/QT2/T0.0 configuration on a seeded random subset of the public SciCite / ACL-ARC
test set that ships in the official repo, and compare with (a) the public gold labels and (b) the authors' own
published predictions for the same items (results/<ds>/2 - Finetuned Q8/zero-shot/SP3/QT2/T0.0/*.csv).
No project edge and no project human label is touched.

usage: python 01_sanity_public_test.py --dataset scicite --n 150
"""
import argparse
import glob
import json
import os
import sys
import time

import pandas as pd
from sklearn.metrics import f1_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cic_llm  # noqa: E402

OUT = "/Volumes/kz_extended/SALT-UIUC/citation_iConf/citation_corpus/analysis/results/external_systems/open_llm_intent"
ap = argparse.ArgumentParser()
ap.add_argument("--dataset", default="scicite", choices=["scicite", "acl-arc"])
ap.add_argument("--n", type=int, default=150)
a = ap.parse_args()

pub = glob.glob(f"{cic_llm.REPO}/results/{a.dataset}/2 - Finetuned Q8/zero-shot/SP3/QT2/T0.0/*.csv")[0]
dfp = pd.read_csv(pub)  # has citation_context, citation_class_label (gold), predicted_classes (authors, cleaned)
full_n = len(dfp)
full_f1 = f1_score(dfp.citation_class_label, dfp.predicted_classes.fillna("NA"), average="macro",
                   labels=cic_llm.CLASS_LABELS[a.dataset])
sub = dfp.sample(n=min(a.n, len(dfp)), random_state=cic_llm.SEED).reset_index(drop=True)

m = cic_llm.CICModel(a.dataset)
t = time.time()
ours = []
for s in sub.citation_context:
    ours.append(m.classify(s)["predicted_label"])
el = time.time() - t
m.close()
sub["local_q4km_pred"] = ours
tag = "sc" if a.dataset == "scicite" else "aa"
sub[["unique_id", "citation_class_label", "predicted_classes", "local_q4km_pred"]].to_csv(
    f"{OUT}/outputs/{tag}_sanity_public_test_subset.csv", index=False)
labs = cic_llm.CLASS_LABELS[a.dataset]
res = {
    "dataset": a.dataset, "published_predictions_file": os.path.relpath(pub, cic_llm.REPO),
    "public_test_n": full_n, "authors_q8_macroF1_full_test_recomputed": round(full_f1, 4),
    "subset_n": len(sub), "subset_seed": cic_llm.SEED,
    "authors_q8_macroF1_on_subset": round(f1_score(sub.citation_class_label, sub.predicted_classes.fillna("NA"), average="macro", labels=labs), 4),
    "local_q4km_macroF1_on_subset": round(f1_score(sub.citation_class_label, sub.local_q4km_pred.fillna("NA"), average="macro", labels=labs), 4),
    "local_vs_authors_identical_frac": round(float((sub.local_q4km_pred.fillna("NA") == sub.predicted_classes.fillna("NA")).mean()), 4),
    "local_unparsed": int(sub.local_q4km_pred.isna().sum()),
    "seconds": round(el, 1), "items_per_s": round(len(sub) / el, 3),
}
json.dump(res, open(f"{OUT}/outputs/{tag}_sanity_public_test_summary.json", "w"), indent=1)
print(json.dumps(res))
