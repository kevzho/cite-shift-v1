#!/usr/bin/env python3
"""Independent I06 row-33 reconstruction from ROW33_SPEC_KZ_2026-09-07.md.

This script must write its JSON before I06 row 33 or the protected measurement
implementations/outputs are opened.

Usage: python3 analysis/verification/row33_kz.py
"""
from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
SPECIFICATION = "analysis/verification/ROW33_SPEC_KZ_2026-09-07.md"
OUTPUT = ROOT / "analysis/verification/row33_kz_results_2026-09-07.json"
INPUTS = {
    "hci": {
        "path": ROOT / "analysis/derived/hci_contribution_coding_human_reviewed.csv",
        "sha256": "b71c2c5c905a2f9aa91eb2a9da52a2340a483aba3ab5bf6fe3cf4545ff404e5e",
    },
    "nature_oa": {
        "path": ROOT / "analysis/derived/nature_oa_contribution_coding_human_reviewed.csv",
        "sha256": "acd497e933934bba6777867ee14649f7395d4f4ba18b134bb9f9273cf8a8780a",
    },
}
REQUIRED_COLUMNS = {
    "domain",
    "citance_function",
    "realizes_contribution",
    "research_contribution",
    "cited_contribution",
    "bulk_or_core",
}
BACKGROUND = "background_bulk"
UPTAKE = {"partial", "yes"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_hash(path: Path, expected: str) -> str:
    observed = sha256(path)
    if observed != expected:
        raise RuntimeError(
            f"hash mismatch for {path}: expected {expected}, observed {observed}"
        )
    return observed


def load(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise RuntimeError(f"{path}: missing header")
        missing = REQUIRED_COLUMNS - set(reader.fieldnames)
        if missing:
            raise RuntimeError(f"{path}: missing required columns {sorted(missing)}")
        return [{key: (value or "") for key, value in row.items()} for row in reader]


def pct(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else 100 * numerator / denominator


def blank_counts(rows: list[dict[str, str]]) -> dict[str, int]:
    return {
        column: sum(not row[column].strip() for row in rows)
        for column in sorted(REQUIRED_COLUMNS)
    }


def contribution_cross_tab(
    rows: list[dict[str, str]], column: str
) -> dict[str, dict[str, int | float | None]]:
    """Audit table; neither contribution column changes the row-33 definition."""
    labels = sorted({row[column].strip() or "<blank>" for row in rows})
    output: dict[str, dict[str, int | float | None]] = {}
    for label in labels:
        group = [row for row in rows if (row[column].strip() or "<blank>") == label]
        background = [row for row in group if row["citance_function"].strip() == BACKGROUND]
        uptake = [
            row
            for row in background
            if row["realizes_contribution"].strip() in UPTAKE
        ]
        output[label] = {
            "rows": len(group),
            "background_rows": len(background),
            "background_with_uptake": len(uptake),
            "within_background_uptake_pct": pct(len(uptake), len(background)),
        }
    return output


def derive(rows: list[dict[str, str]]) -> dict[str, Any]:
    total = len(rows)
    background_rows = [row for row in rows if row["citance_function"].strip() == BACKGROUND]
    partial_rows = [
        row for row in background_rows if row["realizes_contribution"].strip() == "partial"
    ]
    yes_rows = [
        row for row in background_rows if row["realizes_contribution"].strip() == "yes"
    ]
    uptake_rows = [
        row for row in background_rows if row["realizes_contribution"].strip() in UPTAKE
    ]
    hidden_core_rows = [
        row for row in uptake_rows if row["bulk_or_core"].strip() == "core"
    ]
    background_count = len(background_rows)
    uptake_count = len(uptake_rows)
    hidden_core_count = len(hidden_core_rows)
    return {
        "row_count": total,
        "domain_values": dict(sorted(Counter(row["domain"].strip() or "<blank>" for row in rows).items())),
        "blank_counts": blank_counts(rows),
        "definitions": {
            "background_function": BACKGROUND,
            "uptake_realization_values": sorted(UPTAKE),
            "core_value": "core",
        },
        "row33": {
            "background_count": {
                "numerator": background_count,
                "denominator": total,
            },
            "background_pct": {
                "numerator": background_count,
                "denominator": total,
                "value": pct(background_count, total),
            },
            "background_with_uptake_count": {
                "numerator": uptake_count,
                "denominator": total,
            },
            "within_background_uptake_pct": {
                "numerator": uptake_count,
                "denominator": background_count,
                "value": pct(uptake_count, background_count),
            },
            "corpus_wide_uptake_pct": {
                "numerator": uptake_count,
                "denominator": total,
                "value": pct(uptake_count, total),
            },
            "hidden_and_core_count": {
                "numerator": hidden_core_count,
                "denominator": total,
            },
            "hidden_and_core_pct": {
                "numerator": hidden_core_count,
                "denominator": total,
                "value": pct(hidden_core_count, total),
            },
        },
        "uptake_components_within_background": {
            "partial": len(partial_rows),
            "yes": len(yes_rows),
            "partial_or_yes": uptake_count,
        },
        "contribution_cross_tabs": {
            "research_contribution": contribution_cross_tab(rows, "research_contribution"),
            "cited_contribution": contribution_cross_tab(rows, "cited_contribution"),
        },
    }


def main() -> None:
    result: dict[str, Any] = {
        "purpose": "Independent I06 row-33 compression reconstruction",
        "specification": SPECIFICATION,
        "comparison_status": "not compared to I06 row 33 or protected measurement files",
        "inputs": {},
        "corpora": {},
    }
    for corpus, source in INPUTS.items():
        path = source["path"]
        observed = verify_hash(path, source["sha256"])
        rows = load(path)
        result["inputs"][corpus] = {
            "path": str(path.relative_to(ROOT)),
            "sha256_expected": source["sha256"],
            "sha256_observed": observed,
        }
        result["corpora"][corpus] = derive(rows)
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote independent row-33 reconstruction: {OUTPUT}")
    for corpus, details in result["corpora"].items():
        row = details["row33"]
        print(
            f"{corpus}: background={row['background_count']['numerator']}/"
            f"{row['background_count']['denominator']}; "
            f"uptake={row['background_with_uptake_count']['numerator']}/"
            f"{row['within_background_uptake_pct']['denominator']}; "
            f"hidden_core={row['hidden_and_core_count']['numerator']}/"
            f"{row['hidden_and_core_count']['denominator']}"
        )


if __name__ == "__main__":
    main()
