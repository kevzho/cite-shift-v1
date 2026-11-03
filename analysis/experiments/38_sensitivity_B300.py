#!/usr/bin/env python3
"""Post hoc sensitivity (DEVIATIONS.md, 23 Oct 2026): 29_confirmatory_2026_10_08.py re-run unchanged except that sample
B Pass 2 also includes the third stratum Pass 2 file (excluded from the registered analysis, deviation 3a; its labels
are identical to an LLM-generated file on 52/60 rows). Writes to analysis/results/sensitivity_2026-10-23_B300/.
The registered 240-edge results are not modified.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("conf", ROOT / "analysis/experiments/29_confirmatory_2026_10_08.py")
C = importlib.util.module_from_spec(spec)
spec.loader.exec_module(C)
THIRD = C.V2 / "received/B_stratum_pass2_as_submitted_2026-10-05_2223.csv"

_orig = C.references


def references_b300(cw):
    canon, v2b, v2a, v1 = _orig(cw)
    edge = {k["item"]: k["edge_id"] for k in C.read(C.V2 / "B_stratum_pass2_key.csv")}
    for row in C.read(THIRD):
        for f in C.PASS2:
            v = (row.get(f) or "").strip()
            if v and v != "not_applicable":
                v2b[f][edge[row["item"]]] = v
    return canon, v2b, v2a, v1


C.references = references_b300
C.OUT = ROOT / "analysis/results/sensitivity_2026-10-23_B300"
C.main()
