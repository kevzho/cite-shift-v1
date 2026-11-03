"""Sanity check: does the official archive, loaded on this modern stack, reproduce its published SciCite test score?

Scores the model against the SciCite TEST split's own gold labels (1,861 citances; public dataset
s3://ai2-s2-research/scicite/scicite.tar.gz). This is the system's own benchmark, NOT project data: no project
human label is touched. Paper (Cohan et al. 2019, Table 3, BiLSTM-Attn w/ ELMo + both scaffolds): macro-F1 84.0.
Every test citance is scored (no is_sentence filter), batch size 1, CPU.
"""
import json, os, sys, time
sys.path.insert(0, os.path.dirname(__file__))
from compat import ART, load_predictor, LABELS
from sklearn.metrics import f1_score, accuracy_score

pred = load_predictor()
rows = [json.loads(l) for l in open(os.path.join(ART, "scicite_data", "test.jsonl"))]
gold, sys_ = [], []
t0 = time.time()
for i, r in enumerate(rows):
    inst = pred._dataset_reader.text_to_instance(citation_text=r["string"], intent=None, citing_paper_id=r["citingPaperId"],
                                                 cited_paper_id=r["citedPaperId"], citation_excerpt_index=r.get("excerpt_index", i))
    o = pred._model.forward_on_instance(inst)
    gold.append(r["label"]); sys_.append(o["prediction"])
dt = time.time() - t0
print(json.dumps({"n": len(rows), "macro_f1": round(100 * f1_score(gold, sys_, average="macro"), 2),
                  "per_class_f1": dict(zip(LABELS, [round(100 * x, 2) for x in f1_score(gold, sys_, average=None, labels=LABELS)])),
                  "accuracy": round(100 * accuracy_score(gold, sys_), 2), "seconds": round(dt, 1),
                  "citances_per_s": round(len(rows) / dt, 2), "paper_macro_f1": 84.0}, indent=1))
