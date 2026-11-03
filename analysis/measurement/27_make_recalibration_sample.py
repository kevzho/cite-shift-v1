#!/usr/bin/env python3
"""Draw the second calibration sample: 40 fresh edges, 20 per corpus.

Round two. The first 40 measured a draft codebook and returned kappa 0.14-0.44,
which is what a calibration round is for. These 40 test the revised codebook and
must be disjoint from the first set -- those edges have now been discussed and
cannot measure anything.

Stratification, seed, and exclusions are fixed here rather than passed on the
command line, so the sample is reproducible from the file alone.

    python3 analysis/measurement/27_make_recalibration_sample.py --out-dir annotations/generated
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCES = {
    "hci": ROOT / "analysis/derived/hci_contribution_coding_human_reviewed.csv",
    "nature_oa": ROOT / "analysis/derived/nature_oa_contribution_coding_human_reviewed.csv",
}
PRIOR = ROOT / "annotations/generated/calibration_multidimensional_blinded.csv"

SEED = 20260817
PER_CORPUS = 20

COMPACT = ["edge_id", "cited_title", "citing_title", "citation_context",
           "compact_intent", "annotator_note"]
MULTI = ["edge_id", "cited_title", "citing_title", "citation_context",
         "cited_contribution_type", "citation_function", "contribution_realization",
         "centrality", "rhetorical_polarity", "partial_reason", "multi_reference",
         "annotator_note"]

# `multi_reference` is new in round two. A citance citing several papers at once
# was roughly half of round-one rows; the coder is asked to tick it so the rate
# is measured rather than recalled.
#
# `partial_reason` is new in round two. The first round could not separate two
# things a coder might mean by `partial`: that only part of the contribution was
# used, or that the paper was used but not as the type it was labelled. One
# coder's notes pointed at the second. Asking for it inline measures it on live
# cases instead of relying on recall.


def edge_id(corpus: str, cited: str, citing: str) -> str:
    """Stable blinded id. Scheme recorded here because the round-one scheme
    could not be recovered by inspection; these are fresh edges, so a fresh
    scheme costs nothing as long as it is documented."""
    return hashlib.sha256(f"{corpus}|{cited}|{citing}".encode()).hexdigest()[:20]


def read(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--tag", default="recalibration")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    used = {r["cited_title"].strip() for r in read(PRIOR)}
    print(f"excluding {len(used)} cited papers from round one")

    picked = []
    rng = random.Random(SEED)
    for corpus, path in SOURCES.items():
        pool = [r for r in read(path) if r["cited_title"].strip() not in used]
        if len(pool) < PER_CORPUS:
            raise SystemExit(f"{corpus}: pool of {len(pool)} is smaller than {PER_CORPUS}")
        chosen = rng.sample(pool, PER_CORPUS)
        print(f"  {corpus}: pool {len(pool)}, drew {len(chosen)}")
        for r in chosen:
            picked.append({
                "edge_id": edge_id(corpus, r["cited_id"], r["citing_id"]),
                "cited_title": r["cited_title"].strip(),
                "citing_title": r["citing_title"].strip(),
                # Verbatim. Parity with the canonical pipeline is what makes the
                # human-canonical comparison legitimate; any reformatting breaks it.
                "citation_context": r["citances"],
            })

    # Row order is shuffled so corpus is not inferable from position, and the
    # two passes get different orders so a coder cannot align them by memory.
    for name, fields in (("compact", COMPACT), ("multidimensional", MULTI)):
        rows = picked[:]
        random.Random(SEED + len(name)).shuffle(rows)
        out = args.out_dir / f"{args.tag}_{name}_blinded.csv"
        with out.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k, "") for k in fields})
        print(f"wrote {out} ({len(rows)} rows)")

    overlap = {r["cited_title"] for r in picked} & used
    print(f"\noverlap with round one: {len(overlap)} (must be 0)")
    return 0 if not overlap else 1


if __name__ == "__main__":
    raise SystemExit(main())
