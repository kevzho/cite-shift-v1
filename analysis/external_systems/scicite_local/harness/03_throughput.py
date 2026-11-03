"""Throughput estimate ONLY: time the official model on a seeded random sample of lines from the CHI/CHI EA
citance file. Predictions are discarded (not written anywhere). Prints timing and extrapolation."""
import os as _os
from pathlib import Path as _Path
_CITATION_ROOT = _os.environ.get("CITATION_ROOT", str(_Path(__file__).resolve().parents[4]))
import json, os, random, sys, time
sys.path.insert(0, os.path.dirname(__file__))
from compat import load_predictor, classify
SRC = _CITATION_ROOT + "/external_data/chi_chiea_grobid_2026-09-15/citances/chi_chiea_citances.jsonl"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 200
n_lines, n_cits = 0, 0
with open(SRC) as f:
    for l in f:
        n_lines += 1; n_cits += json.loads(l)["n_citances"]
rng = random.Random(20260926)
pick = set(rng.sample(range(n_lines), N))
sample = []
with open(SRC) as f:
    for i, l in enumerate(f):
        if i in pick:
            sample.append([c.strip() for c in json.loads(l)["citances"] if c.strip()])
t_load = time.time(); pred = load_predictor(); t_load = time.time() - t_load
t0 = time.time(); k = 0
for cits in sample:
    for c in cits:
        classify(pred, c); k += 1
dt = time.time() - t0
cps = k / dt; mean_cits = n_cits / n_lines
print(json.dumps({"sample_edges": N, "sample_citances": k, "seconds": round(dt, 1), "model_load_s": round(t_load, 1),
                  "citances_per_s": round(cps, 2), "edges_per_s_sample": round(N / dt, 2),
                  "file_edges": n_lines, "file_citances": n_cits, "mean_citances_per_edge_file": round(mean_cits, 3),
                  "est_hours_226793_edges": round(n_cits / cps / 3600, 2),
                  "est_minutes_2937_edges_at_file_mean": round(2937 * mean_cits / cps / 60, 1)}, indent=1))
