"""Throughput estimate on ~200 lines of the CHI/CHI EA citance file. Outputs are discarded (only timings kept).
Every citance of each sampled edge is classified (as a full run would), after @@CITATION@@ tagging.
usage: python 04_throughput.py --dataset scicite --n 200"""
import argparse
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cic_llm  # noqa: E402

SRC = ("/Volumes/kz_extended/SALT-UIUC/citation_iConf/citation_corpus/external_data/chi_chiea_grobid_2026-09-15/"
       "citances/chi_chiea_citances.jsonl")
OUT = "/Volumes/kz_extended/SALT-UIUC/citation_iConf/citation_corpus/analysis/results/external_systems/open_llm_intent"
ap = argparse.ArgumentParser()
ap.add_argument("--dataset", default="scicite")
ap.add_argument("--n", type=int, default=200)
a = ap.parse_args()

lines = open(SRC).readlines()
N_TOTAL = len(lines)
rng = random.Random(cic_llm.SEED)
sample = [json.loads(lines[i]) for i in sorted(rng.sample(range(N_TOTAL), a.n))]
n_cit_total = sum(len(json.loads(l)["citances"]) for l in lines)
del lines

m = cic_llm.CICModel(a.dataset)
t = time.time()
n_cit = 0
for rec in sample:
    cits = [c for c in rec["citances"] if c.strip()]
    tagged, _ = cic_llm.tag_edge(cits, "hci")
    for c in tagged:
        m.classify(c)  # result intentionally discarded
        n_cit += 1
el = time.time() - t
m.close()
eps = a.n / el
res = {"dataset": a.dataset, "sample_edges": a.n, "sample_citances": n_cit, "seconds": round(el, 1),
       "edges_per_s": round(eps, 4), "citances_per_s": round(n_cit / el, 4),
       "file_edges": N_TOTAL, "file_citances": n_cit_total,
       "est_hours_2937_edges": round(2937 / eps / 3600, 2),
       "est_hours_226793_edges": round(N_TOTAL / eps / 3600, 1)}
json.dump(res, open(f"{OUT}/outputs/throughput_{a.dataset}.json", "w"), indent=1)
print(json.dumps(res))
