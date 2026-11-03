#!/usr/bin/env python3
"""Inter-coder agreement on the 40-edge calibration set.

Reports per dimension and never pooled (plan A5). Pooling across dimensions
would average a near-deterministic judgement (polarity, read off the sentence)
against one the evidence may not determine at all (what the cited paper
contributes), and the mean of those is not a property of anything.

Every dimension is reported twice: once over all rows, and once excluding rows
where either coder said `unclear`. The two answer different questions. Including
`unclear` measures agreement on the task as posed. Excluding it measures
agreement on the cases both coders were willing to decide, which is what a
reliability figure is normally taken to mean -- and it is computed on a
self-selected subset, so it is an upper bound, not a correction.

Cohen's kappa is unstable when one class dominates: with a marginal at 90% the
chance-correction term swamps the observed agreement and kappa collapses even
though the coders rarely differ. Where that happens the script says so rather
than printing a small number without comment.

Usage:
    python3 analysis/measurement/24_calibration_agreement.py \
        --repo annotations/coders/round_one
"""
from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

DIMENSIONS = {
    "calibration_compact_blinded": ["compact_intent"],
    "calibration_multidimensional_blinded": [
        "cited_contribution_type",
        "citation_function",
        "contribution_realization",
        "centrality",
        "rhetorical_polarity",
    ],
}
KEY = "edge_id"
SKEW_WARN = 0.80  # a marginal above this makes kappa hard to read


def read(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def pick(repo: Path, coder: str, stem: str) -> Path:
    """Prefer the token-normalised copy; fall back to the original.

    Normalisation only rewrites spelling of a token ('results' -> 'result').
    It never changes a judgement, so agreement is identical either way -- but
    an un-normalised token would be counted as its own class and silently
    depress agreement.
    """
    folder = repo / "coders" / coder
    norm = folder / f"{stem}_normalized.csv"
    return norm if norm.exists() else folder / f"{stem}.csv"


def cohen_kappa(pairs: list[tuple[str, str]]) -> tuple[float, float, float]:
    """Returns (observed, expected, kappa). kappa is nan when expected == 1."""
    n = len(pairs)
    observed = sum(1 for a, b in pairs if a == b) / n
    ca, cb = Counter(a for a, _ in pairs), Counter(b for _, b in pairs)
    expected = sum((ca[k] / n) * (cb[k] / n) for k in set(ca) | set(cb))
    kappa = float("nan") if expected == 1 else (observed - expected) / (1 - expected)
    return observed, expected, kappa


def per_class(pairs: list[tuple[str, str]]) -> list[tuple[str, int, int, int, float]]:
    """Per class: support from each coder, joint count, and Jaccard.

    Jaccard (both / either) is the honest per-class figure here. Recall against
    one coder would treat that coder as ground truth, and neither is.
    """
    out = []
    for label in sorted(set(a for a, _ in pairs) | set(b for _, b in pairs)):
        na = sum(1 for a, _ in pairs if a == label)
        nb = sum(1 for _, b in pairs if b == label)
        both = sum(1 for a, b in pairs if a == b == label)
        either = sum(1 for a, b in pairs if label in (a, b))
        out.append((label, na, nb, both, both / either if either else float("nan")))
    return out


def confusion(pairs: list[tuple[str, str]]) -> None:
    labels = sorted(set(a for a, _ in pairs) | set(b for _, b in pairs))
    w = max(len(x) for x in labels) + 2
    axis_label = "A \\ B"
    print(f"      {axis_label:<{w}}" + "".join(f"{x[:10]:>12}" for x in labels))
    for a in labels:
        row = [sum(1 for x, y in pairs if x == a and y == b) for b in labels]
        print(f"      {a:<{w}}" + "".join(f"{v:>12}" for v in row))


def report(dimension: str, pairs: list[tuple[str, str]]) -> None:
    print(f"\n  {dimension}")
    observed, expected, kappa = cohen_kappa(pairs)
    print(f"    all rows        n={len(pairs):3d}  agreement {observed:6.1%}  "
          f"kappa {kappa:6.3f}   (chance {expected:.1%})")

    kept = [(a, b) for a, b in pairs if "unclear" not in (a, b)]
    dropped = len(pairs) - len(kept)
    if kept and dropped:
        o2, _, k2 = cohen_kappa(kept)
        print(f"    unclear removed n={len(kept):3d}  agreement {o2:6.1%}  "
              f"kappa {k2:6.3f}   ({dropped} row(s) dropped -- upper bound)")
    elif not dropped:
        print(f"    unclear removed -- no rows contained `unclear`")

    counts = Counter(a for a, _ in pairs) + Counter(b for _, b in pairs)
    top, top_n = counts.most_common(1)[0]
    share = top_n / (2 * len(pairs))
    if share >= SKEW_WARN:
        print(f"    NOTE: `{top}` is {share:.0%} of all judgements. Kappa is not "
              f"readable at this skew; use raw agreement and per-class support.")

    print(f"    {'class':<24}{'A':>5}{'B':>5}{'both':>7}{'jaccard':>10}")
    for label, na, nb, both, jac in per_class(pairs):
        flag = "   <- 1 coder only" if (na == 0) != (nb == 0) else ""
        print(f"    {label:<24}{na:>5}{nb:>5}{both:>7}{jac:>10.2f}{flag}")

    if len(set(a for a, _ in pairs) | set(b for _, b in pairs)) <= 8:
        print()
        confusion(pairs)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, required=True)
    ap.add_argument("--coder-a", default="annotator_a")
    ap.add_argument("--coder-b", default="annotator_b")
    args = ap.parse_args()

    print("Calibration agreement -- 40 edges, draft codebook, two coders")
    print("Coder identities are folder names only; neither is ground truth.")

    for stem, dimensions in DIMENSIONS.items():
        pa, pb = pick(args.repo, args.coder_a, stem), pick(args.repo, args.coder_b, stem)
        rows_a, rows_b = read(pa), read(pb)
        print(f"\n{'=' * 72}\n{stem}\n  A: {pa.name} ({len(rows_a)} rows)"
              f"\n  B: {pb.name} ({len(rows_b)} rows)\n{'=' * 72}")

        by_a = {r[KEY]: r for r in rows_a}
        by_b = {r[KEY]: r for r in rows_b}
        shared = [k for k in by_a if k in by_b]
        if len(shared) != len(rows_a) or len(shared) != len(rows_b):
            print(f"  WARNING: aligned on {len(shared)} shared {KEY}s "
                  f"(A has {len(rows_a) - len(shared)} unmatched, "
                  f"B has {len(rows_b) - len(shared)}). Agreement is over the "
                  f"intersection only.")

        identical = sum(
            1 for k in shared
            if all(by_a[k][d] == by_b[k][d] for d in dimensions)
        )
        print(f"  rows identical across every dimension: {identical}/{len(shared)}"
              f" ({identical / len(shared):.0%})")
        if identical == len(shared):
            print("  STOP: the two files agree on every cell. Independent coding of "
                  "40 items does not produce this. Verify blinding before using "
                  "any number below.")

        for dimension in dimensions:
            report(dimension, [(by_a[k][dimension], by_b[k][dimension]) for k in shared])

    print("\n" + "=" * 72)
    print("Reminder: this measures the DRAFT codebook, which is the point -- it is\n"
          "a defect report, not the paper's reliability figure. The reported\n"
          "reliability comes from the 300-edge validation pass under the FROZEN\n"
          "codebook. Do not quote these numbers as the study's agreement.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
