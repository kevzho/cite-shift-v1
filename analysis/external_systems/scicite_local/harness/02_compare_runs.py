"""Determinism check: are two runs byte-identical (labels and full-precision probabilities)?"""
import os as _os
from pathlib import Path as _Path
_CITATION_ROOT = _os.environ.get("CITATION_ROOT", str(_Path(__file__).resolve().parents[4]))
import filecmp, json, sys
d = _CITATION_ROOT + "/analysis/results/external_systems/scicite_local/outputs/"
res = {k: filecmp.cmp(d + f"scicite_{k}_preds_run1.csv", d + f"scicite_{k}_preds_run2.csv", shallow=False) for k in ("edge", "citance")}
print(json.dumps({"identical": res, "deterministic": all(res.values())}))
