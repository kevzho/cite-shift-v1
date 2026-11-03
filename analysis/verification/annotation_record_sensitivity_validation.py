#!/usr/bin/env python3
"""Measure annotation-record sensitivity of within-background uptake.

The 300 independently coded validation edges can be joined exactly to the two
canonical source sheets by the immutable edge identifier
``edge_id(corpus, cited_id, citing_id)``.  This program applies the same
within-background uptake estimand three times to those *same edges*: once to
the canonical record and once to each coder's multidimensional record.

This is a descriptive sensitivity analysis, not an adjudication. It does not
choose a gold annotation record, and it does not use the corpus-wide component bootstrap:
that bootstrap quantifies edge/component sampling variation conditional on the
canonical labels, whereas this analysis exposes annotation-record variation.

Run from ``citation_corpus``:

    python3 analysis/verification/annotation_record_sensitivity_validation.py
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "analysis/verification/annotation_record_sensitivity_validation_results.json"

CANONICAL_INPUTS = {
    "hci": ROOT / "analysis/derived/hci_contribution_coding_human_reviewed.csv",
    "nature_oa": ROOT / "analysis/derived/nature_oa_contribution_coding_human_reviewed.csv",
}
CODER_INPUTS = {
    "coder_a": ROOT / "annotations/coders/validation/annotator_a_validation_multidimensional_resubmit.csv",
    "coder_b": ROOT / "annotations/coders/validation/annotator_b_validation_multidimensional.csv",
}
EXPECTED_INPUT_SHA256 = {
    "analysis/derived/hci_contribution_coding_human_reviewed.csv": "b71c2c5c905a2f9aa91eb2a9da52a2340a483aba3ab5bf6fe3cf4545ff404e5e",
    "analysis/derived/nature_oa_contribution_coding_human_reviewed.csv": "acd497e933934bba6777867ee14649f7395d4f4ba18b134bb9f9273cf8a8780a",
    "annotations/coders/validation/annotator_a_validation_multidimensional_resubmit.csv": "4473daaa9502ef53f80978970392d5a36fa6a759202b72de9426329def50b245",
    "annotations/coders/validation/annotator_b_validation_multidimensional.csv": "ac8067805368affde540cecb3d88ad2362374b06242c4f82b9b308cbc7149262",
}

ANNOTATION_RECORDS = {
    "canonical": {
        "function_column": "citance_function",
        "realization_column": "realizes_contribution",
        "background_label": "background_bulk",
        "uptake_labels": ("partial", "yes"),
    },
    "coder_a": {
        "function_column": "citation_function",
        "realization_column": "contribution_realization",
        "background_label": "motivates_background",
        "uptake_labels": ("partial", "yes"),
    },
    "coder_b": {
        "function_column": "citation_function",
        "realization_column": "contribution_realization",
        "background_label": "motivates_background",
        "uptake_labels": ("partial", "yes"),
    },
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise RuntimeError(f"{path}: missing header")
        return [{key: (value or "") for key, value in row.items()} for row in reader]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def edge_id(corpus: str, row: dict[str, str]) -> str:
    key = "\x1f".join((corpus, row["cited_id"].strip(), row["citing_id"].strip()))
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:20]


def keyed(rows: list[dict[str, str]], source: Path) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    for row in rows:
        identifier = row.get("edge_id", "").strip()
        if not identifier:
            raise RuntimeError(f"{source}: blank edge_id")
        if identifier in result:
            raise RuntimeError(f"{source}: duplicate edge_id {identifier}")
        result[identifier] = row
    return result


def verify_input(path: Path, input_hashes: dict[str, str]) -> None:
    relative = str(path.relative_to(ROOT))
    observed = digest(path)
    expected = EXPECTED_INPUT_SHA256[relative]
    if observed != expected:
        raise RuntimeError(
            f"{relative}: SHA-256 mismatch; expected {expected}, observed {observed}"
        )
    input_hashes[relative] = observed


def summarise(
    rows: list[dict[str, str]], record: dict[str, Any]
) -> dict[str, Any]:
    function = record["function_column"]
    realization = record["realization_column"]
    background = record["background_label"]
    uptake_labels = set(record["uptake_labels"])
    selected = [row for row in rows if row[function].strip() == background]
    uptake = [row for row in selected if row[realization].strip() in uptake_labels]
    denominator = len(selected)
    numerator = len(uptake)
    function_by_realization: dict[str, Counter[str]] = {}
    for row in rows:
        function_label = row[function].strip() or "<blank>"
        function_by_realization.setdefault(function_label, Counter())[
            row[realization].strip() or "<blank>"
        ] += 1
    return {
        "edge_count": len(rows),
        "background_count": denominator,
        "uptake_count": numerator,
        "within_background_uptake_pct": None if not denominator else 100 * numerator / denominator,
        "function_distribution_all_edges": dict(
            sorted(Counter(row[function].strip() or "<blank>" for row in rows).items())
        ),
        "realization_distribution_all_edges": dict(
            sorted(Counter(row[realization].strip() or "<blank>" for row in rows).items())
        ),
        "function_by_realization_all_edges": {
            function_label: dict(sorted(counts.items()))
            for function_label, counts in sorted(function_by_realization.items())
        },
        "realization_distribution_within_background": dict(
            sorted(Counter(row[realization].strip() or "<blank>" for row in selected).items())
        ),
    }


def main() -> int:
    canonical: dict[str, dict[str, str]] = {}
    corpus_of: dict[str, str] = {}
    input_hashes: dict[str, str] = {}
    for corpus, path in CANONICAL_INPUTS.items():
        verify_input(path, input_hashes)
        for row in read_csv(path):
            identifier = edge_id(corpus, row)
            if identifier in canonical:
                raise RuntimeError(f"duplicate canonical edge_id {identifier}")
            canonical[identifier] = row
            corpus_of[identifier] = corpus

    coders: dict[str, dict[str, dict[str, str]]] = {}
    for name, path in CODER_INPUTS.items():
        verify_input(path, input_hashes)
        coders[name] = keyed(read_csv(path), path)

    canonical_ids = set(canonical)
    coder_sets = {name: set(rows) for name, rows in coders.items()}
    if coder_sets["coder_a"] != coder_sets["coder_b"]:
        raise RuntimeError("coder A and B validation edge_id sets differ")
    validation_ids = coder_sets["coder_a"]
    unmatched = validation_ids - canonical_ids
    if unmatched:
        raise RuntimeError(f"{len(unmatched)} validation edge(s) do not join to canonical data")
    if len(validation_ids) != 300:
        raise RuntimeError(f"expected 300 validation edges, found {len(validation_ids)}")

    ordered_ids = sorted(validation_ids)
    by_corpus = {
        corpus: [identifier for identifier in ordered_ids if corpus_of[identifier] == corpus]
        for corpus in CANONICAL_INPUTS
    }
    if {corpus: len(ids) for corpus, ids in by_corpus.items()} != {"hci": 150, "nature_oa": 150}:
        raise RuntimeError("validation sample is not 150 edges per corpus")

    records = {"canonical": canonical, **coders}
    estimates: dict[str, Any] = {}
    for name, record in ANNOTATION_RECORDS.items():
        source = records[name]
        estimates[name] = {
            "pooled": summarise([source[key] for key in ordered_ids], record),
            **{
                corpus: summarise([source[key] for key in identifiers], record)
                for corpus, identifiers in by_corpus.items()
            },
        }

    canonical_pct = estimates["canonical"]["pooled"]["within_background_uptake_pct"]
    results = {
        "analysis": "within-background uptake annotation-record sensitivity analysis",
        "unit": "one of the 300 paired validation edges",
        "join": {
            "key": "sha256(corpus + U+001F + cited_id + U+001F + citing_id)[:20]",
            "canonical_edges": len(canonical),
            "validation_edges": len(validation_ids),
            "matched_validation_edges": len(validation_ids),
            "unmatched_validation_edges": 0,
            "validation_edges_by_corpus": {corpus: len(ids) for corpus, ids in by_corpus.items()},
        },
        "definition": {
            "background": "canonical function == background_bulk; coder function == motivates_background",
            "uptake": "realization in {partial, yes}",
            "note": "Canonical background_bulk and no map directly to the project codebook's motivates_background and no_background under annotations/label_crosswalk.csv.",
        },
        "input_sha256": input_hashes,
        "estimates": estimates,
        "pooled_spread_percentage_points": {
            "minimum": min(estimates[name]["pooled"]["within_background_uptake_pct"] for name in ANNOTATION_RECORDS),
            "maximum": max(estimates[name]["pooled"]["within_background_uptake_pct"] for name in ANNOTATION_RECORDS),
            "range": max(estimates[name]["pooled"]["within_background_uptake_pct"] for name in ANNOTATION_RECORDS) - min(estimates[name]["pooled"]["within_background_uptake_pct"] for name in ANNOTATION_RECORDS),
            "canonical_minus_coder_a": canonical_pct - estimates["coder_a"]["pooled"]["within_background_uptake_pct"],
            "canonical_minus_coder_b": canonical_pct - estimates["coder_b"]["pooled"]["within_background_uptake_pct"],
        },
        "interpretation_boundary": "These are three annotation records on fixed edges, not three estimates to average and not evidence that any record is a gold standard.",
    }
    OUTPUT.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {OUTPUT}")
    for name in ANNOTATION_RECORDS:
        statistic = estimates[name]["pooled"]
        print(f"{name}: {statistic['uptake_count']}/{statistic['background_count']} = "
              f"{statistic['within_background_uptake_pct']:.2f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
