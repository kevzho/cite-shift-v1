#!/usr/bin/env python3
"""Post hoc precision of the registered RQ1(b) contrast (kappa(realized, crediting) - kappa(realized, uptake)) on B300:
the standard error implied by its 95% cluster-bootstrap interval, and the smallest true difference the design would
detect with 80% power at two-sided alpha = 0.05 (normal approximation: (1.96 + 0.8416) * SE). Reads the existing
output; computes nothing new from labels.

    python3 analysis/experiments/71_rq1_detectable_difference.py [--out PATH]
Default output: analysis/results/rq1_detectable_difference_2026-11-03.json (refuses to overwrite).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "analysis/results/sensitivity_2026-10-23_B300/rq1b.json"
Z_A, Z_B = 1.959964, 0.841621


def precision(block):
    lo, hi = block["difference_ci95"]
    se = (hi - lo) / (2 * Z_A)
    return {"difference": block["kappa_difference_crediting_minus_uptake"], "ci95": [lo, hi],
            "ci_half_width": round((hi - lo) / 2, 4), "se": round(se, 4),
            "mdd_80_power": round((Z_A + Z_B) * se, 4)}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, default=ROOT / "analysis/results/rq1_detectable_difference_2026-11-03.json")
    a = ap.parse_args()
    if a.out.exists():
        sys.exit(f"refusing to overwrite {a.out}")
    d = json.load(open(SRC))
    out = {"source": str(SRC.relative_to(ROOT)), "method": __doc__.split("\n\n")[0].replace("\n", " "),
           "B300_primary": precision(d["B"]["primary_realized_partial_or_yes"]),
           "B300_yes_only": precision(d["B"]["sensitivity_realized_yes_only"]),
           "A250_primary": precision(d["A250_replication"]["primary_realized_partial_or_yes"])}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
