#!/usr/bin/env python3
"""Rescore frozen three-judge disagreement with the full-B reference.

No inference or training. Preserve original results and the dated inclusion
history; choosing this presentation does not make the inclusion decision prior
to results. Field-specific unclear exclusions still apply.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "analysis/results/exploratory/2026-10-25_model_disagreement_B297_primary.json"


def module(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / "analysis/experiments" / filename)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def main():
    if OUT.exists():
        raise SystemExit(f"refusing to overwrite {OUT}")
    full = module("full_b_reference", "39_B300_manuscript_companions.py")
    analysis = module("full_b_disagreement", "41_model_disagreement_ambiguity.py")
    analysis.C, analysis.S, analysis.OUT = full.C, full.C.S, OUT
    analysis.main()
    result = json.loads(OUT.read_text())
    result["reference"] = "Full B: 300 selected, field-specific eligibility; 297 eligible binary cases"
    result["inclusion_chronology"] = (
        "Third stratum submission included on author's report of hand coding after the "
        "October 23 result review; no independent authorship determination. "
        "Two acknowledged LLM-generated submissions remain excluded."
    )
    result["status"] = "Exploratory rescore of frozen predictions, primary full-B presentation"
    OUT.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
