#!/usr/bin/env python3
"""Round-two (recalibration) agreement, 40 edges, revised codebook, two coders.

Two things changed between round one and round two, and both change how
agreement has to be computed.

1. Coders were told they may assign more than one contribution type, marking the
   primary with a trailing `*` (`methodological*;empirical`). A multi-label cell
   has no single correct scoring. This script reports BOTH, because they answer
   different questions and neither subsumes the other:

     primary  -- collapse each cell to its starred label and run Cohen's kappa.
                 This asks "do the coders lead with the same type?" and is
                 comparable to round one, where every cell was single-label.
     set      -- treat each cell as a label set and report mean pairwise Jaccard
                 plus the exact-set agreement rate. This asks "do the coders see
                 the same types at all?", and is the only scoring that can show
                 whether the multi-label instruction is doing work or adding
                 noise.

   Reporting only `primary` would hide the secondary labels entirely; reporting
   only `set` would not be comparable to round one. So: both, always, never a
   single headline number for this dimension.

2. `partial_reason` and `multi_reference` were added. `multi_reference` is
   scored as a binary. `partial_reason` is scored only on rows where at least
   one coder said `partial`, and with n that small it is reported as a raw
   count, not a kappa.

Normalisation is token spelling only -- `neutral use` -> `neutral_use`,
`results` -> `result`, `builds on findings` -> `builds_on_finding`. It never
changes a judgement. Every repair is logged so the rewrite can be audited
rather than trusted. Note the deliberate absence of a `startswith` fallback:
tools/validate_submission.py has one, and on a multi-label cell it silently
truncates `methodological*;empirical` to `methodological`. A repair that
discards a coder's second label is worse than no repair, so unrecognised
tokens are reported and left alone.

Files are read from git refs by default, because the coder branches are not
merged into main and agreement should not wait on a merge.

Usage:
    python3 analysis/measurement/30_round_two_agreement.py \
        --repo private/history/citation-annotation-2027
"""
from __future__ import annotations

import argparse
import csv
import io
import subprocess
from collections import Counter
from pathlib import Path

KEY = "edge_id"
MISSING = "<blank>"
SKEW_WARN = 0.80

# (ref, path) per coder per pass. Defaults match what the coders actually
# committed, including annotator A's ad-hoc filenames.
SOURCES = {
    "compact": {
        "A": ("origin/annotator-a", "coders/annotator_a/compact.csv"),
        "B": ("origin/annotator-b",
              "coders/annotator_b/recalibration_compact_blinded_completed.csv"),
    },
    "multidimensional": {
        "A": ("origin/annotator-a", "coders/annotator_a/mult.csv"),
        "B": ("origin/annotator-b",
              "coders/annotator_b/recalibration_multidimensional_blinded_completed.csv"),
    },
}

LEGAL = {
    "compact_intent": ["background", "method", "result", "unclear"],
    "cited_contribution_type": [
        "artifact", "dataset", "empirical", "methodological",
        "opinion", "survey", "theoretical", "unclear",
    ],
    "citation_function": [
        "motivates_background", "builds_on_finding", "compares_against",
        "applies_framework", "reuses_artifact", "uses_method", "other", "unclear",
    ],
    "contribution_realization": ["yes", "partial", "no_background", "unclear"],
    "centrality": ["core", "bulk", "unclear"],
    "rhetorical_polarity": [
        "supports", "neutral_use", "qualifies", "contrasts", "mixed", "unclear",
    ],
}
MULTI_LABEL = {"cited_contribution_type"}
DIMENSIONS = {
    "compact": ["compact_intent"],
    "multidimensional": [
        "cited_contribution_type", "citation_function",
        "contribution_realization", "centrality", "rhetorical_polarity",
    ],
}


# ---------------------------------------------------------------- reading

def git_show(repo: Path, ref: str, path: str) -> str:
    out = subprocess.run(
        ["git", "-C", str(repo), "show", f"{ref}:{path}"],
        capture_output=True, text=True,
    )
    if out.returncode != 0:
        raise SystemExit(f"cannot read {ref}:{path} -- {out.stderr.strip()}")
    return out.stdout


def load(repo: Path, ref: str, path: str) -> list[dict]:
    return list(csv.DictReader(io.StringIO(git_show(repo, ref, path))))


# ---------------------------------------------------------- normalisation

def norm(value: str) -> str:
    return value.strip().lower().replace(" ", "_").replace("-", "_")


def repair(token: str, legal: list[str]) -> str | None:
    """Spelling repair, or None if the token is not recognisable.

    Deliberately no prefix matching: see the module docstring.
    """
    t = norm(token)
    if t in legal:
        return t
    if t.endswith("s") and t[:-1] in legal:
        return t[:-1]
    return None


def parse_cell(raw: str, dimension: str, log: list) -> tuple[str, frozenset[str]]:
    """Return (primary, label_set). Unrecognised tokens are kept verbatim."""
    legal = LEGAL[dimension]
    if not raw.strip():
        return MISSING, frozenset()

    parts = [p.strip() for p in raw.split(";") if p.strip()]
    labels: list[str] = []
    starred: list[str] = []
    for part in parts:
        is_primary = part.endswith("*")
        token = part[:-1] if is_primary else part
        fixed = repair(token, legal)
        if fixed is None:
            fixed = norm(token)
            log.append((dimension, raw, fixed, "UNRECOGNISED"))
        elif fixed != token.strip():
            log.append((dimension, raw, fixed, "repaired"))
        labels.append(fixed)
        if is_primary:
            starred.append(fixed)

    if starred:
        primary = starred[0]
        if len(starred) > 1:
            log.append((dimension, raw, primary, "MULTIPLE STARS -- took first"))
    elif len(labels) == 1:
        primary = labels[0]
    else:
        primary = labels[0]
        log.append((dimension, raw, primary,
                    "MULTI-LABEL WITH NO STAR -- took first, primary is a guess"))
    return primary, frozenset(labels)


# -------------------------------------------------------------- statistics

def cohen_kappa(pairs: list[tuple[str, str]]) -> tuple[float, float, float]:
    n = len(pairs)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    observed = sum(1 for a, b in pairs if a == b) / n
    ca, cb = Counter(a for a, _ in pairs), Counter(b for _, b in pairs)
    expected = sum((ca[k] / n) * (cb[k] / n) for k in set(ca) | set(cb))
    kappa = float("nan") if expected == 1 else (observed - expected) / (1 - expected)
    return observed, expected, kappa


def per_class(pairs: list[tuple[str, str]]) -> list[tuple]:
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
    print(f"      {axis_label:<{w}}" + "".join(f"{x[:11]:>13}" for x in labels))
    for a in labels:
        row = [sum(1 for x, y in pairs if x == a and y == b) for b in labels]
        print(f"      {a:<{w}}" + "".join(f"{v:>13}" for v in row))


def report_single(dimension: str, pairs: list[tuple[str, str]], label: str = "") -> None:
    head = f"  {dimension}{label}"
    print(f"\n{head}")
    blanks = sum(1 for a, b in pairs if MISSING in (a, b))
    scored = [(a, b) for a, b in pairs if MISSING not in (a, b)]
    if blanks:
        print(f"    {blanks} row(s) blank for at least one coder -- excluded "
              f"from every figure below")
    if not scored:
        print("    nothing left to score")
        return

    o, e, k = cohen_kappa(scored)
    print(f"    all rows        n={len(scored):3d}  agreement {o:6.1%}  "
          f"kappa {k:6.3f}   (chance {e:.1%})")

    kept = [(a, b) for a, b in scored if "unclear" not in (a, b)]
    dropped = len(scored) - len(kept)
    if kept and dropped:
        o2, _, k2 = cohen_kappa(kept)
        print(f"    unclear removed n={len(kept):3d}  agreement {o2:6.1%}  "
              f"kappa {k2:6.3f}   ({dropped} row(s) dropped -- upper bound)")
    elif not dropped:
        print("    unclear removed -- no rows contained `unclear`")

    counts = Counter(a for a, _ in scored) + Counter(b for _, b in scored)
    top, top_n = counts.most_common(1)[0]
    share = top_n / (2 * len(scored))
    if share >= SKEW_WARN:
        print(f"    NOTE: `{top}` is {share:.0%} of all judgements. Kappa is not "
              f"readable at this skew; use raw agreement and per-class support.")

    print(f"    {'class':<24}{'A':>5}{'B':>5}{'both':>7}{'jaccard':>10}")
    for lbl, na, nb, both, jac in per_class(scored):
        flag = "   <- 1 coder only" if (na == 0) != (nb == 0) else ""
        print(f"    {lbl:<24}{na:>5}{nb:>5}{both:>7}{jac:>10.2f}{flag}")
    if len(set(a for a, _ in scored) | set(b for _, b in scored)) <= 9:
        print()
        confusion(scored)


def report_sets(dimension: str, sets: list[tuple[frozenset, frozenset]]) -> None:
    scored = [(a, b) for a, b in sets if a and b]
    blanks = len(sets) - len(scored)
    print(f"\n  {dimension}  [set scoring]")
    if blanks:
        print(f"    {blanks} row(s) blank for at least one coder -- excluded")
    if not scored:
        print("    nothing left to score")
        return
    jac = [len(a & b) / len(a | b) for a, b in scored]
    exact = sum(1 for a, b in scored if a == b)
    multi_a = sum(1 for a, _ in scored if len(a) > 1)
    multi_b = sum(1 for _, b in scored if len(b) > 1)
    either_multi = sum(1 for a, b in scored if len(a) > 1 or len(b) > 1)
    print(f"    n={len(scored)}  mean Jaccard {sum(jac) / len(jac):.3f}  "
          f"exact-set agreement {exact}/{len(scored)} ({exact / len(scored):.0%})")
    print(f"    multi-label cells: A {multi_a}, B {multi_b}, "
          f"either {either_multi}/{len(scored)} ({either_multi / len(scored):.0%})")
    if either_multi:
        sub = [len(a & b) / len(a | b) for a, b in scored if len(a) > 1 or len(b) > 1]
        rest = [len(a & b) / len(a | b) for a, b in scored
                if len(a) == 1 and len(b) == 1]
        print(f"      mean Jaccard on multi-label rows  {sum(sub) / len(sub):.3f}"
              f"  (n={len(sub)})")
        if rest:
            print(f"      mean Jaccard on single-label rows {sum(rest) / len(rest):.3f}"
                  f"  (n={len(rest)})")
        print("      If the multi-label rows score no better than the single-label\n"
              "      rows, the instruction is adding disagreement, not resolving it.")


def report_binary(name: str, pairs: list[tuple[bool, bool]]) -> None:
    print(f"\n  {name}  [binary]")
    strs = [("yes" if a else "no", "yes" if b else "no") for a, b in pairs]
    o, e, k = cohen_kappa(strs)
    na = sum(1 for a, _ in pairs if a)
    nb = sum(1 for _, b in pairs if b)
    both = sum(1 for a, b in pairs if a and b)
    either = sum(1 for a, b in pairs if a or b)
    print(f"    n={len(pairs)}  agreement {o:6.1%}  kappa {k:6.3f}   (chance {e:.1%})")
    print(f"    ticked: A {na}, B {nb}, both {both}, either {either}, "
          f"jaccard {both / either if either else float('nan'):.3f}")
    rate = either / len(pairs)
    print(f"    at least one coder ticked it on {either}/{len(pairs)} rows ({rate:.0%})")


# -------------------------------------------------------------------- main

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, required=True)
    args = ap.parse_args()
    repo = args.repo.resolve()

    print("Round-two (recalibration) agreement -- 40 edges, revised codebook")
    print("Coder identities are folder names only; neither is ground truth.")
    print("This measures the codebook AFTER revision. It is still a defect")
    print("report, not the study's reliability figure -- that comes from the")
    print("validation pass under the frozen codebook.")

    repair_log: list = []

    for pass_name, dimensions in DIMENSIONS.items():
        ref_a, path_a = SOURCES[pass_name]["A"]
        ref_b, path_b = SOURCES[pass_name]["B"]
        rows_a, rows_b = load(repo, ref_a, path_a), load(repo, ref_b, path_b)
        print(f"\n{'=' * 74}\nrecalibration -- {pass_name}")
        print(f"  A: {ref_a}:{path_a} ({len(rows_a)} rows)")
        print(f"  B: {ref_b}:{path_b} ({len(rows_b)} rows)\n{'=' * 74}")

        by_a = {r[KEY]: r for r in rows_a}
        by_b = {r[KEY]: r for r in rows_b}
        shared = [k for k in by_a if k in by_b]
        if len(shared) != len(rows_a) or len(shared) != len(rows_b):
            print(f"  WARNING: aligned on {len(shared)} shared {KEY}s. Agreement "
                  f"is over the intersection only.")

        for dimension in dimensions:
            parsed_a = [parse_cell(by_a[k][dimension], dimension, repair_log)
                        for k in shared]
            parsed_b = [parse_cell(by_b[k][dimension], dimension, repair_log)
                        for k in shared]
            if dimension in MULTI_LABEL:
                report_single(dimension,
                              [(a[0], b[0]) for a, b in zip(parsed_a, parsed_b)],
                              "  [primary scoring]")
                report_sets(dimension,
                            [(a[1], b[1]) for a, b in zip(parsed_a, parsed_b)])
            else:
                report_single(dimension,
                              [(a[0], b[0]) for a, b in zip(parsed_a, parsed_b)])

        if pass_name == "multidimensional":
            report_binary("multi_reference", [
                (bool(by_a[k].get("multi_reference", "").strip()),
                 bool(by_b[k].get("multi_reference", "").strip()))
                for k in shared
            ])
            partial = [k for k in shared
                       if "partial" in (norm(by_a[k]["contribution_realization"]),
                                        norm(by_b[k]["contribution_realization"]))]
            print(f"\n  partial_reason  [raw counts; n={len(partial)} is too small "
                  f"for kappa]")
            for k in partial:
                print(f"    {k}  A: {by_a[k]['contribution_realization']:<14}"
                      f"{by_a[k].get('partial_reason','') or '-':<14}"
                      f"B: {by_b[k]['contribution_realization']:<14}"
                      f"{by_b[k].get('partial_reason','') or '-'}")

    print(f"\n{'=' * 74}\nToken repairs applied ({len(repair_log)})")
    print("=" * 74)
    if not repair_log:
        print("  none")
    else:
        seen = Counter((d, raw, fixed, kind) for d, raw, fixed, kind in repair_log)
        for (d, raw, fixed, kind), n in sorted(seen.items()):
            mark = "  <-- REVIEW" if kind != "repaired" else ""
            print(f"  {n:>3}x  {d:<26}{raw!r:<40} -> {fixed!r}  [{kind}]{mark}")
    print("\n  Spelling only. No judgement was changed. Neither coder ran")
    print("  tools/validate_submission.py -- every space-separated token above")
    print("  would have been caught before commit.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
