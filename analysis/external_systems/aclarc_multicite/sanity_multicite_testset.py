"""Install check: score the released MultiCite SciBERT checkpoint on MultiCite's OWN public test split
(allenai/multicite data/classification_<ctx>_context/test.json) with the repo's strict/weak accuracy.
This touches no project data and no human labels from this project.

Usage: python sanity_multicite_testset.py <path-to-multicite-clone> [gold|1|3]
"""
import json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_multicite import load, predict, set_determinism

repo, ctx = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 else "gold")
set_determinism(4)
tok, model, labels = load("scibert")
data = json.load(open(f"{repo}/data/classification_{ctx}_context/test.json"))
texts = [" ".join(d["x"]) if isinstance(d["x"], list) else d["x"] for d in data]
golds = [set(d["y"].split(" ")) for d in data]
t0 = time.perf_counter()
P = predict(tok, model, texts, 8)
dt = time.perf_counter() - t0
preds = [{labels[i] for i in range(len(labels)) if p[i] > 0.5} for p in P]
strict = sum(p == g for p, g in zip(preds, golds)) / len(golds)
weak = sum(bool(p & g) for p, g in zip(preds, golds)) / len(golds)
print(json.dumps({"context": ctx, "n": len(data), "strict": round(strict, 4), "weak": round(weak, 4),
                  "seconds": round(dt, 1), "inst_per_s": round(len(data) / dt, 2)}))
