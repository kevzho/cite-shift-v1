"""Install check for the ported ACL-ARC scaffold model: score it on the public ACL-ARC test split
(https://s3-us-west-2.amazonaws.com/ai2-s2-research/scicite/acl-arc.tar.gz, test.jsonl, 139 citations)
with the scicite metric (macro-F1 over the 6 classes, as ScaffoldBilstmAttentionClassifier.get_metrics).
Touches no project data. Usage: python sanity_aclarc_testset.py <archive_dir> <acl-arc/test.jsonl>
"""
import json, os, sys, time
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from aclarc_scaffold import AclArcClassifier

torch.manual_seed(0); torch.set_num_threads(4); torch.use_deterministic_algorithms(True)
clf = AclArcClassifier(sys.argv[1])
rows = [json.loads(l) for l in open(sys.argv[2])]
t0 = time.perf_counter()
preds = [clf.predict(r["text"])["label"] for r in rows]
dt = time.perf_counter() - t0
gold = [r["intent"] for r in rows]
f1s = {}
for c in clf.labels:
    tp = sum(p == c and g == c for p, g in zip(preds, gold))
    fp = sum(p == c and g != c for p, g in zip(preds, gold))
    fn = sum(p != c and g == c for p, g in zip(preds, gold))
    pr = tp / (tp + fp) if tp + fp else 0.0
    rc = tp / (tp + fn) if tp + fn else 0.0
    f1s[c] = round(2 * pr * rc / (pr + rc), 4) if pr + rc else 0.0
print(json.dumps({"n": len(rows), "accuracy": round(sum(p == g for p, g in zip(preds, gold)) / len(rows), 4),
                  "macro_f1": round(sum(f1s.values()) / len(f1s), 4), "per_class_f1": f1s,
                  "seconds": round(dt, 1), "inst_per_s": round(len(rows) / dt, 2)}))
