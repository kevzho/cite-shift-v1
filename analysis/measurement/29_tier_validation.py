#!/usr/bin/env python3
"""Restore the full validation sample and split it into two ordered tiers.

D41 cut validation from 150 per corpus to 75 on coder-hour grounds. Those hours
are no longer binding, so the full sample is restored -- but restoring it raises
a stopping problem: if the coders run out of time partway through, the point at
which they stop is decided by how the coding went, and a stopping rule that
depends on the data is not a sampling rule.

The fix is to fix the boundary in advance:

    core       the 75-per-corpus deterministic subsample already drawn under
               D41 (seed 20260820). Coded first.
    extension  the remaining 75 per corpus. Coded only after core is complete.

The analysis pre-commits: report on `core`, and report on `core + extension`
only if `extension` completes in full. A partially coded extension tier is
reported as coverage and never merged into a reliability figure.

Within each tier, each pass keeps its own original row order, so the compact
and multidimensional passes remain non-alignable.

Usage:
    python3 analysis/measurement/29_tier_validation.py --apply
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import random
from pathlib import Path

CORPORA = {"hci": "hci_contribution_coding_human_reviewed.csv",
           "nature_oa": "nature_oa_contribution_coding_human_reviewed.csv"}
PASSES = ("validation_compact_blinded", "validation_multidimensional_blinded")
CORE_PER_CORPUS = 75
CORE_SEED = 20260820          # must match 28_cut_validation.py, or core changes


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def edge_id_v1(corpus: str, row: dict) -> str:
    key = "\x1f".join((corpus, (row.get("cited_id") or "").strip(),
                       (row.get("citing_id") or "").strip()))
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:20]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--derived-dir", type=Path, default=Path("analysis/derived"))
    ap.add_argument("--generated-dir", type=Path, default=Path("annotations/generated"))
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    superseded = args.generated_dir / "superseded"
    corpus_of = {edge_id_v1(c, r): c
                 for c, f in CORPORA.items()
                 for r in read_csv(args.derived_dir / f)}

    for name in PASSES:
        candidates = sorted(superseded.glob(f"{name}_*.csv"))
        if not candidates:
            raise SystemExit(f"no superseded copy of {name}; nothing to restore")
        full = read_csv(max(candidates, key=lambda p: len(read_csv(p))))

        by_corpus: dict[str, list[str]] = {}
        for row in full:
            by_corpus.setdefault(corpus_of[row["edge_id"]], []).append(row["edge_id"])

        core: set[str] = set()
        for corpus, edges in sorted(by_corpus.items()):
            pool = sorted(edges)
            random.Random(CORE_SEED).shuffle(pool)
            core |= set(pool[:CORE_PER_CORPUS])

        tiered = ([{**r, "tier": "core"} for r in full if r["edge_id"] in core]
                  + [{**r, "tier": "extension"} for r in full if r["edge_id"] not in core])

        counts = {t: sum(1 for r in tiered if r["tier"] == t)
                  for t in ("core", "extension")}
        print(f"{name}: {len(full)} rows -> core {counts['core']}, "
              f"extension {counts['extension']}")

        if not args.apply:
            continue

        out = args.generated_dir / f"{name}.csv"
        fields = [f for f in full[0].keys()]
        if name == "validation_multidimensional_blinded":
            # Round two introduced two diagnostic fields after the original
            # 300-edge draw, and the codebook now requires `partial_reason`
            # whenever a coder marks `partial`. The superseded copies predate
            # both, so they are reinstated here. Evidence, edge selection and
            # row order are untouched.
            for field in ("partial_reason", "multi_reference"):
                if field not in fields:
                    fields.insert(fields.index("annotator_note"), field)
        fields = fields + ["tier"]
        with out.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows({f: r.get(f, "") for f in fields} for r in tiered)
        print(f"  wrote {out}")

    if not args.apply:
        print("\ndry run. Re-run with --apply to write.")
        return 0

    print("\nRe-run 02_make_splits.py, 03_build_datasets.py, 04_train.py: the "
          "frozen reservation grows again and training data shrinks.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
