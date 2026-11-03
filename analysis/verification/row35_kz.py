#!/usr/bin/env python3
"""Independent I06 row-35 component-bootstrap reconstruction.

This script carries forward the row-33 derivation unchanged and applies a
component-level bootstrap exactly to that derivation.

Usage:
    python3 analysis/verification/row35_kz.py

Important:
    BY_TYPE_CELLS is fixed by corpus from the seven admitted corpus-type cells
    in analysis/verification/ROW35_SPEC_KZ_2026-09-12.md. Eligibility is not
    re-decided inside bootstrap draws.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from numpy.random import SeedSequence, default_rng


ROOT = Path(__file__).resolve().parents[2]
SPECIFICATION = "analysis/verification/ROW35_SPEC_KZ_2026-09-12.md"
OUTPUT = ROOT / "analysis/verification/row35_kz_results_2026-09-12.json"

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
    "cited_id",
    "citing_id",
}

BACKGROUND = "background_bulk"
UPTAKE = {"partial", "yes"}

N_BOOTSTRAP = 2000
MIN_FINITE_DRAWS = 1900
BOOTSTRAP_ENTROPY = 20260913

EXPECTED_POINT_ESTIMATES = {
    "hci": {"background_count": 676, "uptake_count": 289},
    "nature_oa": {"background_count": 964, "uptake_count": 421},
}

EXPECTED_COMPONENT_COUNTS = {
    "hci": 425,
    "nature_oa": 955,
}

# Seven admitted corpus-type cells, fixed from the observed full sample.
BY_TYPE_CELLS = {
    "hci": (
        "empirical_finding",
        "artifact_system",
        "theory_framework",
    ),
    "nature_oa": (
        "empirical_finding",
        "artifact_system",
        "theory_framework",
        "method_tool",
    ),
}


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
        "domain_values": dict(
            sorted(Counter(row["domain"].strip() or "<blank>" for row in rows).items())
        ),
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


def _find(parent: dict[str, str], node: str) -> str:
    root = node
    while parent[root] != root:
        root = parent[root]
    while parent[node] != node:
        nxt = parent[node]
        parent[node] = root
        node = nxt
    return root


def _union(parent: dict[str, str], rank: dict[str, int], a: str, b: str) -> None:
    for node in (a, b):
        if node not in parent:
            parent[node] = node
            rank[node] = 0

    ra = _find(parent, a)
    rb = _find(parent, b)
    if ra == rb:
        return
    if rank[ra] < rank[rb]:
        ra, rb = rb, ra
    parent[rb] = ra
    if rank[ra] == rank[rb]:
        rank[ra] += 1


def components(
    rows: list[dict[str, str]],
) -> tuple[list[list[int]], dict[str, Any]]:
    """Build endpoint-connected components and audit against component_id if present."""
    parent: dict[str, str] = {}
    rank: dict[str, int] = {}

    for i, row in enumerate(rows):
        cited = row["cited_id"].strip()
        citing = row["citing_id"].strip()
        if not cited or not citing:
            raise RuntimeError(
                f"row {i}: blank cited_id/citing_id prevents unambiguous component construction"
            )
        _union(parent, rank, cited, citing)

    grouped: dict[str, list[int]] = defaultdict(list)
    for i, row in enumerate(rows):
        root = _find(parent, row["cited_id"].strip())
        grouped[root].append(i)

    component_rows = sorted(grouped.values(), key=lambda idxs: min(idxs))

    finding: dict[str, Any] = {
        "reported_component_id_column_present": "component_id" in rows[0] if rows else False,
        "disagreement": None,
    }

    if rows and "component_id" in rows[0]:
        derived_component_for_row: dict[int, int] = {}
        for component_index, idxs in enumerate(component_rows):
            for i in idxs:
                derived_component_for_row[i] = component_index

        derived_to_reported: dict[int, set[str]] = defaultdict(set)
        reported_to_derived: dict[str, set[int]] = defaultdict(set)
        blank_reported_rows: list[int] = []

        for i, row in enumerate(rows):
            reported = row.get("component_id", "").strip()
            if not reported:
                blank_reported_rows.append(i)
                continue
            derived_id = derived_component_for_row[i]
            derived_to_reported[derived_id].add(reported)
            reported_to_derived[reported].add(derived_id)

        merged_reported_ids = {
            str(k): sorted(v)
            for k, v in derived_to_reported.items()
            if len(v) > 1
        }
        split_reported_ids = {
            k: sorted(v)
            for k, v in reported_to_derived.items()
            if len(v) > 1
        }
        disagreement = bool(
            blank_reported_rows or merged_reported_ids or split_reported_ids
        )

        finding["disagreement"] = disagreement
        finding["blank_reported_component_id_rows"] = blank_reported_rows[:20]
        finding["blank_reported_component_id_count"] = len(blank_reported_rows)
        finding["derived_components_spanning_multiple_reported_ids"] = merged_reported_ids
        finding["reported_ids_spanning_multiple_derived_components"] = split_reported_ids
    else:
        finding["disagreement"] = "not_assessed_missing_component_id_column"

    return component_rows, finding


def draws(n_components: int, corpus_key: int) -> list[np.ndarray]:
    """Generate 2,000 draws from one deterministic RNG stream per corpus."""
    rng = default_rng(
        SeedSequence(entropy=BOOTSTRAP_ENTROPY, spawn_key=(corpus_key,))
    )
    return [
        rng.integers(0, n_components, size=n_components)
        for _ in range(N_BOOTSTRAP)
    ]


def resample(
    rows: list[dict[str, str]],
    component_rows: list[list[int]],
    draw: np.ndarray,
) -> list[dict[str, str]]:
    """Concatenate rows for sampled components; repeated components repeat their rows."""
    sampled: list[dict[str, str]] = []
    for component_index in draw:
        sampled.extend(rows[i] for i in component_rows[int(component_index)])
    return sampled


def _finite_or_nan(value: float | None) -> float:
    if value is None:
        return float("nan")
    value = float(value)
    return value if np.isfinite(value) else float("nan")


def estimands(
    rows: list[dict[str, str]], corpus: str
) -> dict[str, Any]:
    """Evaluate row-35 estimands for the fixed eligible cells of one corpus."""
    derived = derive(rows)
    row33 = derived["row33"]

    background_count = int(row33["background_count"]["numerator"])
    yes_count = int(derived["uptake_components_within_background"]["yes"])

    overall = {
        "within_background_uptake_pct": _finite_or_nan(
            row33["within_background_uptake_pct"]["value"]
        ),
        "within_background_yes_only_pct": _finite_or_nan(
            pct(yes_count, background_count)
        ),
    }

    research_tab = contribution_cross_tab(rows, "research_contribution")
    by_type: dict[str, float] = {}
    for cell in BY_TYPE_CELLS[corpus]:
        details = research_tab.get(cell)
        value = None if details is None else details["within_background_uptake_pct"]
        by_type[cell] = _finite_or_nan(value)

    return {
        "overall": overall,
        "research_contribution_by_type": by_type,
    }


def interval(values: list[float]) -> dict[str, Any]:
    finite = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    count = int(finite.size)
    if count < MIN_FINITE_DRAWS:
        return {
            "finite_draws": count,
            "suppressed": True,
            "note": (
                f"interval suppressed: {count} finite draws; "
                f"requires at least {MIN_FINITE_DRAWS}"
            ),
            "lower_2_5": None,
            "upper_97_5": None,
        }

    q = np.quantile(finite, [0.025, 0.975], method="linear")
    return {
        "finite_draws": count,
        "suppressed": False,
        "note": None,
        "lower_2_5": float(q[0]),
        "upper_97_5": float(q[1]),
    }


def _validate_by_type_cells(rows_by_corpus: dict[str, list[dict[str, str]]]) -> None:
    expected_counts = {"hci": 3, "nature_oa": 4}

    if set(BY_TYPE_CELLS) != set(expected_counts):
        raise RuntimeError(
            "BY_TYPE_CELLS must define exactly the hci and nature_oa corpora."
        )

    total_cells = 0
    for corpus, expected_count in expected_counts.items():
        cells = BY_TYPE_CELLS[corpus]
        if len(cells) != expected_count:
            raise RuntimeError(
                f"{corpus}: expected {expected_count} fixed by-type cells, "
                f"found {len(cells)}."
            )
        if len(set(cells)) != len(cells):
            raise RuntimeError(f"{corpus}: BY_TYPE_CELLS contains duplicates.")

        labels = {
            row["research_contribution"].strip() or "<blank>"
            for row in rows_by_corpus[corpus]
        }
        absent = [cell for cell in cells if cell not in labels]
        if absent:
            raise RuntimeError(
                f"{corpus}: spec-named BY_TYPE_CELLS absent from the observed "
                f"full-sample research_contribution labels: {absent}"
            )
        total_cells += len(cells)

    if total_cells != 7:
        raise RuntimeError(
            f"Expected 7 admitted corpus-type cells in total, found {total_cells}."
        )


def _flatten_draw_estimates(
    estimates: list[dict[str, Any]], corpus: str
) -> dict[str, list[float]]:
    flattened: dict[str, list[float]] = {
        "within_background_uptake_pct": [],
        "within_background_yes_only_pct": [],
    }
    for cell in BY_TYPE_CELLS[corpus]:
        flattened[f"research_contribution::{cell}"] = []

    for estimate in estimates:
        flattened["within_background_uptake_pct"].append(
            estimate["overall"]["within_background_uptake_pct"]
        )
        flattened["within_background_yes_only_pct"].append(
            estimate["overall"]["within_background_yes_only_pct"]
        )
        for cell in BY_TYPE_CELLS[corpus]:
            flattened[f"research_contribution::{cell}"].append(
                estimate["research_contribution_by_type"][cell]
            )
    return flattened


def main() -> None:
    result: dict[str, Any] = {
        "purpose": "Independent I06 row-35 component-bootstrap reconstruction",
        "specification": SPECIFICATION,
        "comparison_status": (
            "not compared to I06 row 35 or protected measurement implementations/outputs"
        ),
        "bootstrap": {
            "level": "connected component",
            "draws": N_BOOTSTRAP,
            "entropy": BOOTSTRAP_ENTROPY,
            "rng": (
                "one numpy.default_rng per corpus from "
                "SeedSequence(entropy=20260913, spawn_key=(k_c,)); "
                "k_HCI=0, k_Nature_OA=1"
            ),
            "interval": "2.5th and 97.5th percentiles; np.quantile(method='linear')",
            "minimum_finite_draws": MIN_FINITE_DRAWS,
            "delta_definition": "paired draw b: HCI minus Nature OA",
            "by_type_eligibility": {
                "fixed_from_spec": True,
                "column": "research_contribution",
                "cells_by_corpus": {
                    corpus: list(cells)
                    for corpus, cells in BY_TYPE_CELLS.items()
                },
            },
        },
        "inputs": {},
        "corpora": {},
        "paired_deltas_hci_minus_nature_oa": {},
    }

    rows_by_corpus: dict[str, list[dict[str, str]]] = {}

    # Load + hash-check before any derivation.
    for corpus, source in INPUTS.items():
        path = source["path"]
        observed = verify_hash(path, source["sha256"])
        rows = load(path)
        rows_by_corpus[corpus] = rows
        result["inputs"][corpus] = {
            "path": str(path.relative_to(ROOT)),
            "sha256_expected": source["sha256"],
            "sha256_observed": observed,
        }

    _validate_by_type_cells(rows_by_corpus)

    # Safety assertions: identical derive() path must reproduce row-33 counts.
    for corpus, expected in EXPECTED_POINT_ESTIMATES.items():
        d = derive(rows_by_corpus[corpus])
        background_count = d["row33"]["background_count"]["numerator"]
        uptake_count = d["row33"]["background_with_uptake_count"]["numerator"]
        if background_count != expected["background_count"] or uptake_count != expected["uptake_count"]:
            raise RuntimeError(
                f"{corpus}: row-33 safety assertion failed; "
                f"expected background={expected['background_count']}, "
                f"uptake={expected['uptake_count']}; observed "
                f"background={background_count}, uptake={uptake_count}. "
                "Stop before bootstrapping."
            )

    flattened_by_corpus: dict[str, dict[str, list[float]]] = {}

    corpus_rng_keys = {"hci": 0, "nature_oa": 1}

    for corpus, rows in rows_by_corpus.items():
        component_rows, component_finding = components(rows)
        expected_components = EXPECTED_COMPONENT_COUNTS[corpus]
        if len(component_rows) != expected_components:
            raise RuntimeError(
                f"{corpus}: component count mismatch; "
                f"expected {expected_components}, observed {len(component_rows)}"
            )

        # Generate exactly once per corpus before any bootstrap statistic is computed.
        corpus_draws = draws(
            len(component_rows),
            corpus_rng_keys[corpus],
        )

        point = estimands(rows, corpus)
        bootstrap_estimates = [
            estimands(resample(rows, component_rows, draw), corpus)
            for draw in corpus_draws
        ]
        flattened = _flatten_draw_estimates(bootstrap_estimates, corpus)
        flattened_by_corpus[corpus] = flattened

        result["corpora"][corpus] = {
            "row_count": len(rows),
            "component_count": len(component_rows),
            "component_size_summary": {
                "min": min(len(x) for x in component_rows),
                "median": float(np.median([len(x) for x in component_rows])),
                "max": max(len(x) for x in component_rows),
            },
            "component_id_cross_check": component_finding,
            "row33_safety_assertion": {
                "background_count": derive(rows)["row33"]["background_count"]["numerator"],
                "uptake_count": derive(rows)["row33"]["background_with_uptake_count"]["numerator"],
                "passed": True,
            },
            "point_estimates": point,
            "bootstrap_intervals": {
                name: interval(values)
                for name, values in flattened.items()
            },
        }

    # Paired draw b: HCI minus Nature OA for estimands defined in both corpora.
    hci = flattened_by_corpus["hci"]
    nature = flattened_by_corpus["nature_oa"]
    shared_estimands = sorted(set(hci) & set(nature))

    for name in shared_estimands:
        deltas = [
            h - n if np.isfinite(h) and np.isfinite(n) else float("nan")
            for h, n in zip(hci[name], nature[name], strict=True)
        ]
        point_hci = (
            result["corpora"]["hci"]["point_estimates"]["overall"].get(name)
            if name in result["corpora"]["hci"]["point_estimates"]["overall"]
            else result["corpora"]["hci"]["point_estimates"]["research_contribution_by_type"][
                name.split("::", 1)[1]
            ]
        )
        point_nature = (
            result["corpora"]["nature_oa"]["point_estimates"]["overall"].get(name)
            if name in result["corpora"]["nature_oa"]["point_estimates"]["overall"]
            else result["corpora"]["nature_oa"]["point_estimates"]["research_contribution_by_type"][
                name.split("::", 1)[1]
            ]
        )
        point_delta = (
            float(point_hci - point_nature)
            if np.isfinite(point_hci) and np.isfinite(point_nature)
            else None
        )
        result["paired_deltas_hci_minus_nature_oa"][name] = {
            "point_estimate": point_delta,
            "interval": interval(deltas),
        }

    OUTPUT.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote independent row-35 reconstruction: {OUTPUT}")

    for corpus, details in result["corpora"].items():
        point = details["point_estimates"]["overall"]
        print(
            f"{corpus}: components={details['component_count']}; "
            f"uptake={point['within_background_uptake_pct']:.6f}%; "
            f"yes_only={point['within_background_yes_only_pct']:.6f}%"
        )


if __name__ == "__main__":
    main()
