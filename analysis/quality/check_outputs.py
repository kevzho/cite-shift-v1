#!/usr/bin/env python3
"""Check active iConference inputs, outputs, and scope boundaries."""

from __future__ import annotations

import argparse
import csv
import hashlib
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
HCI = ROOT / "yiren_artifacts/HCI contribution coding - hci coding (n=1040).csv"
NATURE = ROOT / "yiren_artifacts/HCI contribution coding - nature_oa coding (n=1897).csv"
MEASUREMENT = ROOT / "analysis/results/measurement"

EXPECTED = {
    HCI: (1040, "352f770e1de35dff0ff7f57066893d3725608bd1062c94a396e070eac3490a32"),
    NATURE: (1897, "e8d5dd06e88a3dfb494334bbd8fbf06e51b0b3df3e79a698485f2a38d1ddabe2"),
}

REQUIRED_OUTPUTS = [
    MEASUREMENT / "cross_domain_label_distributions.csv",
    MEASUREMENT / "cross_domain_realization_by_contribution.csv",
    MEASUREMENT / "cross_domain_function_by_realization.csv",
    MEASUREMENT / "compression_loss_overall.csv",
    MEASUREMENT / "compression_loss_by_contribution.csv",
    MEASUREMENT / "cross_domain_comparison_report.md",
    MEASUREMENT / "compression_loss_report.md",
]

# The canonical edge counts every corpus-level denominator must reproduce.
CORPUS_EDGE_COUNTS = {"hci": 1040, "nature_oa": 1897}

EXCLUDED_ACTIVE_PATHS = [
    ROOT / "analysis/measurement/04_weighted_impact_vectors.py",
    ROOT / "analysis/provenance/storyline_adapter.py",
    ROOT / "analysis/experiments/03_make_ideation_prompts.py",
    ROOT / "analysis/experiments/11_analyze_ideation_pilot.py",
    ROOT / "yiren_artifacts/HCI contribution coding - storylines_hci_coding.csv",
]


def csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strict", action="store_true", help="Treat warnings as failures.")
    args = parser.parse_args()
    errors: list[str] = []
    warnings: list[str] = []

    for path, (expected_rows, expected_hash) in EXPECTED.items():
        if not path.exists():
            errors.append(f"missing canonical input: {path.relative_to(ROOT)}")
            continue
        rows = csv_rows(path)
        if len(rows) != expected_rows:
            errors.append(f"{path.name}: expected {expected_rows} rows, found {len(rows)}")
        if sha256(path) != expected_hash:
            errors.append(f"{path.name}: SHA-256 differs from migration manifest")
        if path in (HCI, NATURE):
            pairs = {((r.get("cited_id") or "").strip(), (r.get("citing_id") or "").strip()) for r in rows}
            if len(pairs) != expected_rows:
                errors.append(f"{path.name}: citation pairs are not unique")

    for path in REQUIRED_OUTPUTS:
        if not path.exists() or path.stat().st_size == 0:
            errors.append(f"missing or empty retained output: {path.relative_to(ROOT)}")

    dist_path = MEASUREMENT / "cross_domain_label_distributions.csv"
    if dist_path.exists():
        totals: dict[tuple[str, str], int] = {}
        for row in csv_rows(dist_path):
            key = (row["corpus"], row["dimension"])
            totals[key] = totals.get(key, 0) + int(row["n"])
        expected_totals = {"hci": 1040, "nature_oa": 1897}
        for (corpus, dimension), total in totals.items():
            if total != expected_totals.get(corpus):
                errors.append(f"{corpus}/{dimension}: distribution totals {total}")

    # The headline compression-loss measure: every count must nest inside its
    # declared denominator, and the by-contribution breakdown must partition
    # exactly the background edges the overall table reports.
    overall_compression = MEASUREMENT / "compression_loss_overall.csv"
    by_contribution = MEASUREMENT / "compression_loss_by_contribution.csv"
    if overall_compression.exists() and by_contribution.exists():
        overall_rows = {row["corpus"]: row for row in csv_rows(overall_compression)}
        if set(overall_rows) != set(CORPUS_EDGE_COUNTS):
            errors.append(
                "compression_loss_overall.csv must report exactly "
                + ", ".join(sorted(CORPUS_EDGE_COUNTS))
            )
        breakdown: dict[str, dict[str, int]] = {}
        for row in csv_rows(by_contribution):
            totals = breakdown.setdefault(row["corpus"], {"background": 0, "hidden": 0})
            totals["background"] += int(row["n_background"])
            totals["hidden"] += int(row["n_hidden_realization"])

        for corpus, expected_edges in CORPUS_EDGE_COUNTS.items():
            row = overall_rows.get(corpus)
            if row is None:
                continue
            edges = int(row["n_edges"])
            background = int(row["n_background"])
            hidden = int(row["n_hidden_realization"])
            hidden_core = int(row["n_hidden_and_core"])
            if edges != expected_edges:
                errors.append(
                    f"compression loss/{corpus}: expected {expected_edges} edges, found {edges}"
                )
            if not hidden_core <= hidden <= background <= edges:
                errors.append(
                    f"compression loss/{corpus}: counts do not nest "
                    f"(core {hidden_core} <= hidden {hidden} <= background {background} <= edges {edges})"
                )
            totals = breakdown.get(corpus)
            if totals is None:
                errors.append(f"compression loss/{corpus}: no by-contribution rows")
                continue
            if totals["background"] != background:
                errors.append(
                    f"compression loss/{corpus}: by-contribution background sums to "
                    f"{totals['background']}, overall reports {background}"
                )
            if totals["hidden"] != hidden:
                errors.append(
                    f"compression loss/{corpus}: by-contribution hidden realisation sums to "
                    f"{totals['hidden']}, overall reports {hidden}"
                )

    for path in EXCLUDED_ACTIVE_PATHS:
        if path.exists():
            errors.append(f"experimental-only artifact is active: {path.relative_to(ROOT)}")

    stale_weighted = list(MEASUREMENT.glob("*weighted*"))
    if stale_weighted:
        errors.append("weighted outputs remain active: " + ", ".join(p.name for p in stale_weighted))
    stale_ideation = list((ROOT / "analysis/results/experiments").glob("ideation*"))
    if stale_ideation:
        errors.append("ideation outputs remain active: " + ", ".join(p.name for p in stale_ideation))

    for warning in warnings:
        print(f"WARNING: {warning}")
    for error in errors:
        print(f"ERROR: {error}")

    if errors or (args.strict and warnings):
        return 1
    print("PASS: active iConference inputs, denominators, outputs, and exclusions are consistent")
    return 0


if __name__ == "__main__":
    sys.exit(main())
