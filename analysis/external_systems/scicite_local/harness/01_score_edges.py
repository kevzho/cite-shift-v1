"""Score the 20 test edges with the OFFICIAL SciCite model archive (allenai/scicite).

Each citance ('|||'-separated) is classified separately with the official `predictor_scicite` (batch size 1).
Only the citance string goes in: no titles, no section name, no IDs other than opaque placeholders.
Reads ONLY edge_id, corpus and citances from test_edges.csv; canonical_* / bulk_or_core are never read.

Edge aggregation (declared before any scoring; no agreement with human labels is computed):
  edge_label = argmax of the MEAN class-probability vector over the edge's citances that pass the official
               is_sentence() filter; if none pass, over all citances (edge_agg_basis says which).
  Also reported: per-citance labels, label counts, any_method, any_result.
"""
import os as _os
from pathlib import Path as _Path
_CITATION_ROOT = _os.environ.get("CITATION_ROOT", str(_Path(__file__).resolve().parents[4]))
import argparse, csv, json, os, sys, time
sys.path.insert(0, os.path.dirname(__file__))
from compat import load_predictor, classify, LABELS

ap = argparse.ArgumentParser()
ap.add_argument("--edges", default=_CITATION_ROOT + "/analysis/results/external_systems/test_edges.csv")
ap.add_argument("--tag", required=True)
ap.add_argument("--outdir", default=_CITATION_ROOT + "/analysis/results/external_systems/scicite_local/outputs")
a = ap.parse_args()

for _n in (f"scicite_citance_preds_{a.tag}.csv", f"scicite_edge_preds_{a.tag}.csv"):
    if os.path.exists(os.path.join(a.outdir, _n)):
        sys.exit(f"refusing to overwrite {_n}")
pred = load_predictor()
edges = [{k: r[k] for k in ("edge_id", "corpus", "citances")} for r in csv.DictReader(open(a.edges))]
cit_rows, edge_rows = [], []
t0 = time.time()
for e in edges:
    cits = [c.strip() for c in e["citances"].split("|||") if c.strip()]
    per = []
    for i, c in enumerate(cits):
        off, mdl, probs, ok = classify(pred, c, idx=i)
        per.append((off, mdl, probs, ok))
        cit_rows.append({"edge_id": e["edge_id"], "citance_index": i, "passes_sentence_filter": int(ok),
                         "official_prediction": off, "model_prediction": mdl,
                         **{f"p_{l}": repr(p) for l, p in zip(LABELS, probs)}})
    basis = [p for p in per if p[3]] or per
    mean = [sum(p[2][j] for p in basis) / len(basis) for j in range(3)]
    counts = {l: sum(1 for p in basis if p[1] == l) for l in LABELS}
    edge_rows.append({"edge_id": e["edge_id"], "corpus": e["corpus"], "n_citances": len(cits),
                      "n_pass_filter": sum(p[3] for p in per),
                      "edge_agg_basis": "filtered" if any(p[3] for p in per) else "all_citances_fallback",
                      "edge_label": LABELS[max(range(3), key=lambda j: mean[j])],
                      **{f"mean_p_{l}": repr(m) for l, m in zip(LABELS, mean)},
                      **{f"n_{l}": counts[l] for l in LABELS},
                      "any_method": int(counts["method"] > 0), "any_result": int(counts["result"] > 0),
                      "citance_labels": "|".join(p[1] for p in per)})
dt = time.time() - t0
os.makedirs(a.outdir, exist_ok=True)
for name, rows in ((f"scicite_citance_preds_{a.tag}.csv", cit_rows), (f"scicite_edge_preds_{a.tag}.csv", edge_rows)):
    with open(os.path.join(a.outdir, name), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
print(json.dumps({"tag": a.tag, "edges": len(edge_rows), "citances": len(cit_rows), "seconds": round(dt, 2)}))
