#!/usr/bin/env python3
"""Determinism check: compares two runs of THIS system with each other (never with human labels).

Reports byte-identity of raw model output per (edge, pass), identity of parsed labels,
JSON parse / schema validity, and the degree rule. Writes determinism_<a>_vs_<b>.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

OUT = Path(__file__).resolve().parents[3] / "analysis/results/external_systems/local_llm_codebook_v2"
LABELS = ["cited_contribution_type", "citation_function", "credits_contribution", "uptake_mode",
          "uptake_degree", "claim_dependent", "centrality", "rhetorical_polarity"]


def load(tag):
    return {d["edge_id"]: d for d in map(json.loads, (OUT / f"raw_{tag}.jsonl").open())}


def main(a="run1", b="run2"):
    A, B = load(a), load(b)
    assert set(A) == set(B)
    calls = raw_same = parse_ok = valid = 0
    label_diffs, raw_diffs = [], []
    for e in sorted(A):
        for p in A[e]["passes"]:
            calls += 1
            ra, rb = A[e]["passes"][p], B[e]["passes"][p]
            raw_same += ra["raw"] == rb["raw"]
            if ra["raw"] != rb["raw"]:
                raw_diffs.append(f"{e}:{p}")
            parse_ok += ra["json_parse_ok"] + rb["json_parse_ok"]
            valid += (not ra["schema_problems"]) + (not rb["schema_problems"])
            pa, pb = ra["parsed"] or {}, rb["parsed"] or {}
            for f in pa.keys() | pb.keys():
                if f in LABELS and pa.get(f) != pb.get(f):
                    label_diffs.append({"edge_id": e, "field": f, a: pa.get(f), b: pb.get(f)})
    rep = {"runs": [a, b], "edges": len(A), "calls_per_run": calls,
           "raw_output_identical_calls": raw_same, "raw_output_differing_calls": raw_diffs,
           "label_differences": label_diffs,
           "json_parse_ok": f"{parse_ok}/{2*calls}", "schema_valid": f"{valid}/{2*calls}",
           "deterministic_labels": not label_diffs, "deterministic_raw": raw_same == calls}
    (OUT / f"determinism_{a}_vs_{b}.json").write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main(*sys.argv[1:3])
