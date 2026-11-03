#!/usr/bin/env python3
"""Reduce the frozen validation sample to a smaller per-corpus size.

The validation pass was approved for reduction. The existing sample is 150 per
corpus (300 rows total); this script takes a deterministic subsample of it
rather than drawing a fresh one, because the original draw was already blinded
and disjoint from calibration, and redrawing would forfeit that guarantee for
no gain.

Two properties are preserved deliberately:

  * The compact and multidimensional passes keep their own independent row
    orders. They were shuffled separately so a coder cannot align them, and
    taking "the first N rows" of each would silently select two different edge
    sets. The subsample is chosen over edge ids, then each pass is filtered in
    its own existing order.
  * Per-corpus balance is preserved. The passes are blinded and carry no corpus
    column, so corpus is recovered by recomputing the round-one blinded id from
    the derived CSVs.

Superseded files are moved to annotations/generated/superseded/ rather than
deleted, and that folder is not picked up by 02_make_splits.py, whose glob is
non-recursive.

Usage:
    python3 analysis/measurement/28_cut_validation.py --per-corpus 75
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import random
import shutil
from pathlib import Path

CORPORA = {"hci": "hci_contribution_coding_human_reviewed.csv",
           "nature_oa": "nature_oa_contribution_coding_human_reviewed.csv"}
PASSES = ("validation_compact_blinded", "validation_multidimensional_blinded")


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
    ap.add_argument("--per-corpus", type=int, required=True)
    ap.add_argument("--seed", type=int, default=20260820)
    ap.add_argument("--apply", action="store_true",
                    help="without this the script reports what it would do and exits")
    args = ap.parse_args()

    corpus_of: dict[str, str] = {}
    for corpus, filename in CORPORA.items():
        for row in read_csv(args.derived_dir / filename):
            corpus_of[edge_id_v1(corpus, row)] = corpus

    passes = {name: read_csv(args.generated_dir / f"{name}.csv") for name in PASSES}

    ids = {name: [r["edge_id"] for r in rows] for name, rows in passes.items()}
    if set(ids[PASSES[0]]) != set(ids[PASSES[1]]):
        raise SystemExit("the two passes cover different edges; refusing to cut")

    by_corpus: dict[str, list[str]] = {}
    for edge in sorted(set(ids[PASSES[0]])):
        by_corpus.setdefault(corpus_of.get(edge, "UNMATCHED"), []).append(edge)
    if "UNMATCHED" in by_corpus:
        raise SystemExit(f"{len(by_corpus['UNMATCHED'])} edge id(s) match no corpus; "
                         "the blinded id scheme has changed. Refusing to cut.")

    keep: set[str] = set()
    for corpus, edges in sorted(by_corpus.items()):
        if args.per_corpus > len(edges):
            raise SystemExit(f"{corpus}: asked for {args.per_corpus}, have {len(edges)}")
        pool = sorted(edges)
        random.Random(args.seed).shuffle(pool)
        keep |= set(pool[:args.per_corpus])
        print(f"{corpus}: {len(edges)} -> {args.per_corpus}")

    print(f"\ntotal: {len(ids[PASSES[0]])} -> {len(keep)} edges, "
          f"{len(keep) * 2} codings per coder across both passes")

    if not args.apply:
        print("\ndry run. Re-run with --apply to write.")
        return 0

    superseded = args.generated_dir / "superseded"
    superseded.mkdir(parents=True, exist_ok=True)

    for name, rows in passes.items():
        src = args.generated_dir / f"{name}.csv"
        shutil.move(str(src), str(superseded / f"{name}_{len(rows)}.csv"))
        kept = [r for r in rows if r["edge_id"] in keep]   # original order preserved
        fieldnames = list(rows[0].keys())
        if name == "validation_multidimensional_blinded":
            # Round two introduced two diagnostic fields after the original
            # 300-edge validation draw. Add them to the cut template without
            # changing any evidence, edge selection, or row order.
            for field in ("partial_reason", "multi_reference"):
                if field not in fieldnames:
                    fieldnames.insert(fieldnames.index("annotator_note"), field)
        with src.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows({field: row.get(field, "") for field in fieldnames}
                             for row in kept)
        print(f"wrote {src} ({len(kept)} rows); "
              f"superseded copy in {superseded.name}/")

    print("\nRe-run 02_make_splits.py and 03_build_datasets.py: the frozen "
          "reservation shrinks and training data comes back.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
