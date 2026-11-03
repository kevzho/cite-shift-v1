#!/usr/bin/env python3
"""Validation-pass agreement, from Kevin Zhou's row-32 specification.

Source specification:
    Row 32 reconstruction specification — Kevin Zhou, 7 September 2026
    (analysis/verification/ROW32_SPEC_KZ_2026-09-07.md), carried over unmodified.

Relationship to row32_kz.py:
    Identical logic. Only the input paths, their pinned digests, and the output
    path differ. The specification was written, and the implementation tested
    against a known reference, before these files were opened.

Annotator A input:
    The 7 September 14:44 resubmission (annotation repo origin/annotator-a,
    commit 6ba8cd0), which passes tools/validate_submission.py. The earlier
    A submission is superseded and retained under its own lock entry.

Not covered by that specification:
    The validation files carry a `tier` column the round-two files did not.
    This program pools all rows. Any tier-stratified statistic needs a
    specification amendment first.

Run:
    python3 analysis/verification/validation_kz.py
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import Counter
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Paths and frozen input facts
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]
VALIDATION = ROOT / "annotations" / "coders" / "validation"

RESULTS_PATH = (
    ROOT
    / "analysis"
    / "verification"
    / "validation_kz_results_2026-09-07.json"
)

EDGE_ID = "edge_id"

INPUTS = {
    "compact": {
        "a": {
            "path": VALIDATION / "annotator_a_validation_compact_resubmit.csv",
            "sha256": "3b9a2ec87a0dfa833e2dda0fb3235d90644e7e373bebed7faeed8e830287a187",
        },
        "b": {
            "path": VALIDATION / "annotator_b_validation_compact.csv",
            "sha256": "f49e86fb75ef5129c4d2e2d57ea3e8712297db4aa59c51700107ed0789203b4d",
        },
        "dimensions": {
            "compact_intent": "scalar",
        },
    },
    "multidimensional": {
        "a": {
            "path": VALIDATION / "annotator_a_validation_multidimensional_resubmit.csv",
            "sha256": "4473daaa9502ef53f80978970392d5a36fa6a759202b72de9426329def50b245",
        },
        "b": {
            "path": VALIDATION / "annotator_b_validation_multidimensional.csv",
            "sha256": "ac8067805368affde540cecb3d88ad2362374b06242c4f82b9b308cbc7149262",
        },
        "dimensions": {
            "cited_contribution_type": "contribution",
            "citation_function": "scalar",
            "contribution_realization": "scalar",
            "centrality": "scalar",
            "rhetorical_polarity": "scalar",
        },
    },
}


# The specification defines the behavior of an explicitly encoded empty set,
# but not its literal spelling. These are kept isolated from ordinary labels.
EMPTY_SET_ENCODINGS = frozenset({"∅", "{}"})


# ---------------------------------------------------------------------------
# Basic utilities
# ---------------------------------------------------------------------------

def digest_file(path: Path) -> str:
    """Return the SHA-256 digest of path."""
    hasher = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)

    return hasher.hexdigest()


def missing(value: str | None) -> bool:
    """A trimmed-empty CSV cell is a missing judgement."""
    return value is None or value.strip() == ""

def normalize(value: str) -> str:
    """Apply only the normalization permitted by the specification."""
    return re.sub(r"[\s-]+", "_", value.strip().casefold())

# ---------------------------------------------------------------------------
# CSV loading and pairing
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AnnotationFile:
    fields: tuple[str, ...]
    rows: dict[str, dict[str, str]]

def read_annotations(path: Path) -> AnnotationFile:
    """Load one coder file, enforcing valid and unique edge_id values."""
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)

        if reader.fieldnames is None:
            raise ValueError(f"{path}: CSV has no header")

        if EDGE_ID not in reader.fieldnames:
            raise ValueError(
                f"{path}: required ID column {EDGE_ID!r} is absent"
            )

        rows: dict[str, dict[str, str]] = {}

        for line_number, source_row in enumerate(reader, start=2):
            edge_id = (source_row.get(EDGE_ID) or "").strip()

            if not edge_id:
                raise ValueError(
                    f"{path}:{line_number}: blank {EDGE_ID}"
                )

            if edge_id in rows:
                raise ValueError(
                    f"{path}:{line_number}: duplicate {EDGE_ID} "
                    f"{edge_id!r}"
                )

            rows[edge_id] = {
                column: value or ""
                for column, value in source_row.items()
                if column is not None
            }

    return AnnotationFile(
        fields=tuple(reader.fieldnames),
        rows=rows,
    )


def verify_hash(path: Path, expected: str) -> str:
    """Assert that an input file is the hash-pinned file in the specification."""
    actual = digest_file(path)

    if actual != expected:
        raise ValueError(
            f"{path}: SHA-256 mismatch\n"
            f"expected: {expected}\n"
            f"actual:   {actual}"
        )

    return actual


def form_pair(
    path_a: Path,
    path_b: Path,
    required_columns: set[str],
) -> tuple[
    dict[str, dict[str, str]],
    dict[str, dict[str, str]],
    list[str],
]:
    """Load and pair one form by edge_id only."""
    coder_a = read_annotations(path_a)
    coder_b = read_annotations(path_b)

    for path, annotation_file in (
        (path_a, coder_a),
        (path_b, coder_b),
    ):
        absent = required_columns - set(annotation_file.fields)

        if absent:
            raise ValueError(
                f"{path}: required columns absent: {sorted(absent)}"
            )

    ids_a = set(coder_a.rows)
    ids_b = set(coder_b.rows)

    if ids_a != ids_b:
        only_a = ids_a - ids_b
        only_b = ids_b - ids_a

        raise ValueError(
            "coder ID sets differ; pairing aborted: "
            f"only A={len(only_a)}, only B={len(only_b)}"
        )

    return coder_a.rows, coder_b.rows, sorted(ids_a)


# ---------------------------------------------------------------------------
# Cohen's kappa
# ---------------------------------------------------------------------------

def nominal_kappa(
    left: list[str],
    right: list[str],
) -> float | None:
    """Compute unweighted nominal Cohen's kappa."""
    if len(left) != len(right):
        raise ValueError("kappa inputs must be equally sized")

    if not left:
        return None

    n = len(left)

    observed = sum(
        x == y
        for x, y in zip(left, right)
    ) / n

    left_counts = Counter(left)
    right_counts = Counter(right)

    # Category universe is exactly the union of labels observed in the
    # complete paired observations supplied to this statistic.
    categories = set(left_counts) | set(right_counts)

    expected = sum(
        (left_counts[label] / n) * (right_counts[label] / n)
        for label in categories
    )

    denominator = 1.0 - expected

    if denominator == 0:
        return None

    return (observed - expected) / denominator


# ---------------------------------------------------------------------------
# Label distributions and disagreement structure
# ---------------------------------------------------------------------------

def label_diagnostics(
    labels_a: list[str],
    labels_b: list[str],
) -> dict[str, Any]:
    """Return per-coder counts and an A-row/B-column confusion matrix.

    These diagnostics use exactly the complete pairs supplied to the agreement
    statistic.  Categories are the union of labels observed from either coder;
    zero-count cells are retained so that the JSON is a rectangular matrix.
    """
    if len(labels_a) != len(labels_b):
        raise ValueError("diagnostic inputs must be equally sized")

    categories = sorted(set(labels_a) | set(labels_b))
    counts_a = Counter(labels_a)
    counts_b = Counter(labels_b)
    matrix = {
        label_a: {
            label_b: 0
            for label_b in categories
        }
        for label_a in categories
    }

    for label_a, label_b in zip(labels_a, labels_b):
        matrix[label_a][label_b] += 1

    return {
        "label_distributions": {
            "annotator_a": {
                label: counts_a[label]
                for label in categories
            },
            "annotator_b": {
                label: counts_b[label]
                for label in categories
            },
        },
        "confusion_matrix": {
            "row_coder": "annotator_a",
            "column_coder": "annotator_b",
            "categories": categories,
            "counts": matrix,
        },
    }


def contribution_set_label_distributions(
    parsed: list[tuple[Contribution, Contribution]],
) -> dict[str, dict[str, int]]:
    """Count all contribution labels, including non-primary set members."""
    counts_a: Counter[str] = Counter()
    counts_b: Counter[str] = Counter()

    for left, right in parsed:
        counts_a.update(left.labels)
        counts_b.update(right.labels)

    categories = sorted(set(counts_a) | set(counts_b))

    return {
        "annotator_a": {
            label: counts_a[label]
            for label in categories
        },
        "annotator_b": {
            label: counts_b[label]
            for label in categories
        },
    }


# ---------------------------------------------------------------------------
# Complete-pair handling
# ---------------------------------------------------------------------------

def completeness(
    rows_a: dict[str, dict[str, str]],
    rows_b: dict[str, dict[str, str]],
    ids: list[str],
    column: str,
) -> tuple[int, int, int, list[str]]:
    """Return blank counts and IDs with two nonblank judgements."""
    blank_a = 0
    blank_b = 0
    blank_either = 0
    complete: list[str] = []

    for edge_id in ids:
        a_missing = missing(rows_a[edge_id][column])
        b_missing = missing(rows_b[edge_id][column])

        blank_a += int(a_missing)
        blank_b += int(b_missing)
        blank_either += int(a_missing or b_missing)

        if not a_missing and not b_missing:
            complete.append(edge_id)

    return blank_a, blank_b, blank_either, complete


# ---------------------------------------------------------------------------
# Scalar dimensions
# ---------------------------------------------------------------------------

def scalar_statistics(
    rows_a: dict[str, dict[str, str]],
    rows_b: dict[str, dict[str, str]],
    ids: list[str],
    column: str,
) -> dict[str, Any]:
    """Calculate complete-pair agreement statistics for a scalar dimension."""
    blank_a, blank_b, blank_either, complete = completeness(
        rows_a,
        rows_b,
        ids,
        column,
    )

    labels_a = [
        normalize(rows_a[edge_id][column])
        for edge_id in complete
    ]
    labels_b = [
        normalize(rows_b[edge_id][column])
        for edge_id in complete
    ]

    n = len(complete)

    agreement_count = sum(
        a == b
        for a, b in zip(labels_a, labels_b)
    )

    agreement = (
        None
        if n == 0
        else agreement_count / n
    )
    diagnostics = label_diagnostics(labels_a, labels_b)

    return {
        "n_paired": len(ids),
        "blank_a": blank_a,
        "blank_b": blank_b,
        "blank_either": blank_either,
        "n_complete": n,
        "agreement_count": agreement_count,
        "agreement": agreement,
        "agreement_pct": (
            None if agreement is None else 100.0 * agreement
        ),
        "cohen_kappa": nominal_kappa(labels_a, labels_b),
        "observed_categories": sorted(
            set(labels_a) | set(labels_b)
        ),
        **diagnostics,
    }


# ---------------------------------------------------------------------------
# Contribution-type parsing
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Contribution:
    primary: str | None
    labels: frozenset[str]


def parse_contribution(raw: str) -> Contribution:
    """Parse one nonblank cited_contribution_type judgement."""
    stripped = raw.strip()

    if stripped in EMPTY_SET_ENCODINGS:
        return Contribution(
            primary=None,
            labels=frozenset(),
        )

    pieces = [
        piece.strip()
        for piece in stripped.split(";")
    ]

    if not pieces or any(piece == "" for piece in pieces):
        raise ValueError(
            f"invalid contribution syntax: {raw!r}"
        )

    starred_indices = [
        index
        for index, piece in enumerate(pieces)
        if piece.endswith("*")
    ]

    labels: list[str] = []

    for piece in pieces:
        source = (
            piece[:-1]
            if piece.endswith("*")
            else piece
        )

        label = normalize(source)

        if not label:
            raise ValueError(
                f"invalid contribution label: {raw!r}"
            )

        labels.append(label)

    if len(set(labels)) != len(labels):
        raise ValueError(
            f"duplicate contribution label after normalization: {raw!r}"
        )

    if len(pieces) == 1 and not starred_indices:
        primary = labels[0]

    elif len(pieces) > 1 and len(starred_indices) == 1:
        primary = labels[starred_indices[0]]

    elif len(pieces) == 1 and len(starred_indices) == 1:
        primary = labels[0]

    else:
        raise ValueError(
            "multi-label contribution requires exactly one trailing '*': "
            f"{raw!r}"
        )

    return Contribution(
        primary=primary,
        labels=frozenset(labels),
    )


def jaccard(
    left: frozenset[str],
    right: frozenset[str],
) -> float:
    """Per-item set Jaccard, including the explicit empty/empty case."""
    union = left | right

    if not union:
        # Both judgements explicitly encode the empty set.
        return 1.0

    return len(left & right) / len(union)


# ---------------------------------------------------------------------------
# Contribution dimension
# ---------------------------------------------------------------------------

def contribution_statistics(
    rows_a: dict[str, dict[str, str]],
    rows_b: dict[str, dict[str, str]],
    ids: list[str],
) -> dict[str, Any]:
    """Score cited_contribution_type as primary labels and unordered sets."""
    column = "cited_contribution_type"

    blank_a, blank_b, blank_either, complete = completeness(
        rows_a,
        rows_b,
        ids,
        column,
    )

    parsed: list[tuple[Contribution, Contribution]] = []

    for edge_id in complete:
        parsed.append(
            (
                parse_contribution(rows_a[edge_id][column]),
                parse_contribution(rows_b[edge_id][column]),
            )
        )

    # The present round-two schema is expected to provide a primary for every
    # non-empty contribution judgement. An explicit empty set has no primary,
    # so it cannot silently become a nominal primary category.
    primary_pairs = [
        (left.primary, right.primary)
        for left, right in parsed
        if left.primary is not None and right.primary is not None
    ]

    primary_incomplete = [
        (left, right)
        for left, right in parsed
        if left.primary is None or right.primary is None
    ]

    if primary_incomplete:
        raise ValueError(
            "explicit empty contribution set encountered in a primary-label "
            "comparison; the specification defines empty-set Jaccard behavior "
            "but does not define an empty-set primary category"
        )

    primary_a = [
        left
        for left, _ in primary_pairs
    ]
    primary_b = [
        right
        for _, right in primary_pairs
    ]

    primary_n = len(primary_pairs)

    primary_agreement_count = sum(
        a == b
        for a, b in primary_pairs
    )

    primary_agreement = (
        None
        if primary_n == 0
        else primary_agreement_count / primary_n
    )

    jaccard_scores = [
        jaccard(left.labels, right.labels)
        for left, right in parsed
    ]

    exact_set_agreements = sum(
        left.labels == right.labels
        for left, right in parsed
    )
    diagnostics = label_diagnostics(primary_a, primary_b)
    set_distributions = contribution_set_label_distributions(parsed)

    return {
        "n_paired": len(ids),
        "blank_a": blank_a,
        "blank_b": blank_b,
        "blank_either": blank_either,
        "n_complete": len(complete),

        "primary_agreement_count": primary_agreement_count,
        "primary_agreement": primary_agreement,
        "primary_agreement_pct": (
            None
            if primary_agreement is None
            else 100.0 * primary_agreement
        ),
        "primary_cohen_kappa": nominal_kappa(
            primary_a,
            primary_b,
        ),
        "primary_categories": sorted(
            set(primary_a) | set(primary_b)
        ),
        # Label-level diagnostics use the primary label, which is the unit of
        # the contribution kappa. Set-level agreement remains reported below.
        "label_diagnostic_unit": "primary_contribution_label",
        **diagnostics,
        "set_label_distributions": set_distributions,

        "mean_set_jaccard": (
            None
            if not jaccard_scores
            else sum(jaccard_scores) / len(jaccard_scores)
        ),
        "set_exact_agreement_count": exact_set_agreements,
        "set_exact_agreement": (
            None
            if not parsed
            else exact_set_agreements / len(parsed)
        ),
    }


# ---------------------------------------------------------------------------
# Reconstruction
# ---------------------------------------------------------------------------

def reconstruct_form(
    form_name: str,
    config: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Reconstruct one form independently of the other form."""
    path_a: Path = config["a"]["path"]
    path_b: Path = config["b"]["path"]

    expected_a: str = config["a"]["sha256"]
    expected_b: str = config["b"]["sha256"]

    actual_a = verify_hash(path_a, expected_a)
    actual_b = verify_hash(path_b, expected_b)

    dimensions: dict[str, str] = config["dimensions"]

    required = {
        EDGE_ID,
        *dimensions.keys(),
    }

    rows_a, rows_b, ids = form_pair(
        path_a,
        path_b,
        required,
    )

    dimension_results: dict[str, Any] = {}

    for column, kind in dimensions.items():
        if kind == "scalar":
            dimension_results[column] = scalar_statistics(
                rows_a,
                rows_b,
                ids,
                column,
            )

        elif kind == "contribution":
            dimension_results[column] = contribution_statistics(
                rows_a,
                rows_b,
                ids,
            )

        else:
            raise ValueError(
                f"{form_name}: unknown dimension type {kind!r}"
            )

    input_record = {
        "annotator_a": {
            "path": str(path_a.relative_to(ROOT)),
            "sha256": actual_a,
        },
        "annotator_b": {
            "path": str(path_b.relative_to(ROOT)),
            "sha256": actual_b,
        },
    }

    form_record = {
        "n_paired": len(ids),
        "pairing": (
            "paired within form by unique nonblank edge_id; "
            "identical coder ID sets required"
        ),
        "dimensions": dimension_results,
    }

    return input_record, form_record


def print_result_summary(result: dict[str, Any]) -> None:
    print(f"Wrote validation agreement: {RESULTS_PATH}")

    for form_name, form in result["forms"].items():
        print(
            f"{form_name}: {form['n_paired']} paired IDs"
        )

        for dimension, statistics in form["dimensions"].items():
            if dimension == "cited_contribution_type":
                print(
                    f"  {dimension}: "
                    f"n={statistics['n_complete']}, "
                    f"primary_agreement="
                    f"{statistics['primary_agreement']}, "
                    f"primary_kappa="
                    f"{statistics['primary_cohen_kappa']}, "
                    f"mean_jaccard="
                    f"{statistics['mean_set_jaccard']}"
                )
            else:
                print(
                    f"  {dimension}: "
                    f"n={statistics['n_complete']}, "
                    f"agreement={statistics['agreement']}, "
                    f"kappa={statistics['cohen_kappa']}"
                )


def main() -> None:
    result: dict[str, Any] = {
        "purpose": "Validation-pass agreement",
        "specification": (
            "Row 32 reconstruction specification — "
            "Kevin Zhou, 7 September 2026 (carried over unmodified)"
        ),
        "date": str(date.today()),
        "comparison_status": (
            "independent reconstruction only; "
            "independent derivation; no prior reference value exists"
        ),
        "inputs": {},
        "forms": {},
    }

    for form_name, config in INPUTS.items():
        input_record, form_record = reconstruct_form(
            form_name,
            config,
        )

        result["inputs"][form_name] = input_record
        result["forms"][form_name] = form_record

    serialized = (
        json.dumps(
            result,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    RESULTS_PATH.write_text(
        serialized,
        encoding="utf-8",
    )

    print_result_summary(result)


if __name__ == "__main__":
    main()
