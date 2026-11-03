"""Score the 20 test edges with a released fine-tuned CIC model. Raw outputs only (pre-registration: no comparison
with canonical_* columns or any human label is computed here or anywhere in this harness).

usage: python 02_score_edges.py --dataset scicite --run 1
"""
import argparse
import json
import os
import sys
import time

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cic_llm  # noqa: E402

ROOT = "/Volumes/kz_extended/SALT-UIUC/citation_iConf/citation_corpus"
EDGES = f"{ROOT}/analysis/results/external_systems/test_edges.csv"
OUT = f"{ROOT}/analysis/results/external_systems/open_llm_intent"

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", default="scicite", choices=["scicite", "acl-arc"])
ap.add_argument("--run", type=int, required=True)
a = ap.parse_args()

# only the columns needed as model input are read; canonical_* are never loaded
edges = pd.read_csv(EDGES, usecols=["edge_id", "corpus", "cited_id", "citing_id", "citances"], dtype=str)
t0 = time.time()
m = cic_llm.CICModel(a.dataset)
load_s = time.time() - t0
tag = "sc" if a.dataset == "scicite" else "aa"

cit_rows, edge_rows, inputs = [], [], []
t1 = time.time()
for _, r in edges.iterrows():
    cits = [c.strip() for c in r.citances.split("|||") if c.strip()]
    tagged, res = cic_llm.tag_edge(cits, r.corpus)
    labs, probs = [], []
    for i, (c, t, rs) in enumerate(zip(cits, tagged, res)):
        o = m.classify(t)
        labs.append(o["predicted_label"])
        probs.append(o["first_token_probs"])
        cit_rows.append({"edge_id": r.edge_id, "corpus": r.corpus, "citance_idx": i, "marker_resolution": rs,
                         "tagged_input": t, "raw_completion": o["raw_completion"],
                         "predicted_label": o["predicted_label"],
                         **{f"p1_{k.replace(' ', '_')}": round(v, 6) for k, v in o["first_token_probs"].items()},
                         "n_prompt_tokens": o["n_prompt_tokens"]})
        inputs.append({"edge_id": r.edge_id, "citance_idx": i, "prompt_text": o["prompt_text"]})
    # edge aggregation (declared before any scoring): label with highest mean first-token probability over citances;
    # the per-citance labels are kept as the primary output.
    mean_p = {l: sum(p[l] for p in probs) / len(probs) for l in m.labels}
    edge_rows.append({"edge_id": r.edge_id, "corpus": r.corpus, "n_citances": len(cits),
                      "citance_labels": "|".join(str(x) for x in labs),
                      "edge_label_meanp": max(mean_p, key=mean_p.get),
                      **{f"any_{l.replace(' ', '_')}": int(l in labs) for l in m.labels},
                      **{f"meanp1_{l.replace(' ', '_')}": round(v, 6) for l, v in mean_p.items()},
                      "n_unparsed": sum(x is None for x in labs)})
run_s = time.time() - t1
m.close()

os.makedirs(f"{OUT}/outputs", exist_ok=True)
pd.DataFrame(cit_rows).to_csv(f"{OUT}/outputs/{tag}_citance_preds_run{a.run}.csv", index=False)
pd.DataFrame(edge_rows).to_csv(f"{OUT}/outputs/{tag}_edge_preds_run{a.run}.csv", index=False)
with open(f"{OUT}/inputs/{tag}_prompts.jsonl", "w") as f:
    for x in inputs:
        f.write(json.dumps(x, ensure_ascii=False) + "\n")
print(json.dumps({"dataset": a.dataset, "run": a.run, "model": m.path, "n_edges": len(edge_rows),
                  "n_citances": len(cit_rows), "load_s": round(load_s, 1), "score_s": round(run_s, 1),
                  "citances_per_s": round(len(cit_rows) / run_s, 3)}))
