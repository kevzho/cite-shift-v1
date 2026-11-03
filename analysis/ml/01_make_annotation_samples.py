#!/usr/bin/env python3
"""Create deterministic blinded calibration and validation samples.

This script exposes only the text required for annotation. It intentionally
omits canonical contribution, function, realization, centrality, rationale,
domain, venue, and all model/proxy fields.

AI-policy notice: this is an AI-assisted internal scaffold. Do not use it to
produce submission evidence unless the human team independently reconstructs
and verifies the sampling design/code or obtains written venue clearance.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SOURCES = {
    "hci": ROOT / "yiren_artifacts/HCI contribution coding - hci coding (n=1040).csv",
    "nature_oa": ROOT
    / "yiren_artifacts/HCI contribution coding - nature_oa coding (n=1897).csv",
}
DEFAULT_OUTPUT = ROOT / "annotations/generated"
EVIDENCE_COLUMNS = (
    "edge_id",
    "cited_title",
    "citing_title",
    "citation_context",
)
COMPACT_COLUMNS = EVIDENCE_COLUMNS + (
    "compact_intent",
    "annotator_note",
)
MULTIDIMENSIONAL_COLUMNS = EVIDENCE_COLUMNS + (
    "cited_contribution_type",
    "citation_function",
    "contribution_realization",
    "centrality",
    "rhetorical_polarity",
    "partial_reason",
    "multi_reference",
    "annotator_note",
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def edge_id(corpus: str, row: dict[str, str]) -> str:
    key = "\x1f".join(
        (
            corpus,
            (row.get("cited_id") or "").strip(),
            (row.get("citing_id") or "").strip(),
        )
    )
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:20]


def evidence_row(corpus: str, row: dict[str, str]) -> dict[str, str]:
    return {
        "edge_id": edge_id(corpus, row),
        "cited_title": (row.get("cited_title") or "").strip(),
        "citing_title": (row.get("citing_title") or "").strip(),
        "citation_context": (row.get("citing_sentence(s)") or "").strip(),
    }


def compact_row(corpus: str, row: dict[str, str]) -> dict[str, str]:
    return {
        **evidence_row(corpus, row),
        "compact_intent": "",
        "annotator_note": "",
    }


def multidimensional_row(corpus: str, row: dict[str, str]) -> dict[str, str]:
    return {
        **evidence_row(corpus, row),
        "cited_contribution_type": "",
        "citation_function": "",
        "contribution_realization": "",
        "centrality": "",
        "rhetorical_polarity": "",
        "partial_reason": "",
        "multi_reference": "",
        "annotator_note": "",
    }


def choose(
    rows: list[dict[str, str]],
    *,
    calibration_n: int,
    validation_n: int,
    seed: int,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    if calibration_n < 0 or validation_n < 1:
        raise ValueError("sample sizes must be non-negative, with validation at least 1")
    if calibration_n + validation_n > len(rows):
        raise ValueError("requested samples exceed available rows")
    shuffled = list(rows)
    random.Random(seed).shuffle(shuffled)
    calibration = shuffled[:calibration_n]
    validation = shuffled[calibration_n : calibration_n + validation_n]
    return calibration, validation


def write_csv(
    path: Path, rows: list[dict[str, str]], columns: tuple[str, ...]
) -> None:
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration-per-corpus", type=int, default=20)
    parser.add_argument("--validation-per-corpus", type=int, default=150)
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    output = args.output_dir.resolve()
    if output.exists():
        raise SystemExit(f"refusing to overwrite existing output directory: {output}")
    output.mkdir(parents=True)

    calibration_source: list[tuple[str, dict[str, str]]] = []
    validation_source: list[tuple[str, dict[str, str]]] = []
    corpus_counts: dict[str, dict[str, int]] = {}
    for offset, (corpus, path) in enumerate(SOURCES.items()):
        rows = read_csv(path)
        ids = {edge_id(corpus, row) for row in rows}
        if len(ids) != len(rows):
            raise SystemExit(f"{corpus}: edge IDs are not unique")
        cal, val = choose(
            rows,
            calibration_n=args.calibration_per_corpus,
            validation_n=args.validation_per_corpus,
            seed=args.seed + offset,
        )
        calibration_source.extend((corpus, row) for row in cal)
        validation_source.extend((corpus, row) for row in val)
        corpus_counts[corpus] = {
            "source_rows": len(rows),
            "calibration_rows": len(cal),
            "validation_rows": len(val),
        }

    if {edge_id(corpus, row) for corpus, row in calibration_source} & {
        edge_id(corpus, row) for corpus, row in validation_source
    }:
        raise SystemExit("calibration and validation samples overlap")

    files: dict[str, dict[str, int | str]] = {}
    generated = (
        (
            "calibration_compact_blinded.csv",
            [compact_row(corpus, row) for corpus, row in calibration_source],
            COMPACT_COLUMNS,
            args.seed + 10_000,
        ),
        (
            "calibration_multidimensional_blinded.csv",
            [
                multidimensional_row(corpus, row)
                for corpus, row in calibration_source
            ],
            MULTIDIMENSIONAL_COLUMNS,
            args.seed + 20_000,
        ),
        (
            "validation_compact_blinded.csv",
            [compact_row(corpus, row) for corpus, row in validation_source],
            COMPACT_COLUMNS,
            args.seed + 30_000,
        ),
        (
            "validation_multidimensional_blinded.csv",
            [
                multidimensional_row(corpus, row)
                for corpus, row in validation_source
            ],
            MULTIDIMENSIONAL_COLUMNS,
            args.seed + 40_000,
        ),
    )
    for name, rows, columns, shuffle_seed in generated:
        random.Random(shuffle_seed).shuffle(rows)
        path = output / name
        write_csv(path, rows, columns)
        files[name] = {"rows": len(rows), "sha256": sha256(path)}

    manifest = {
        "status": "generated; not frozen until human approval is recorded",
        "seed": args.seed,
        "calibration_per_corpus": args.calibration_per_corpus,
        "validation_per_corpus": args.validation_per_corpus,
        "corpora": corpus_counts,
        "files": files,
        "compact_columns": list(COMPACT_COLUMNS),
        "multidimensional_columns": list(MULTIDIMENSIONAL_COLUMNS),
        "pass_separation": (
            "Complete compact and multidimensional files in separate, "
            "human-approved passes; do not merge or expose other labels."
        ),
        "hidden_fields": [
            "corpus",
            "domain",
            "venue",
            "section",
            "bulk_or_core",
            "n_citances",
            "cited_contribution",
            "contribution_subtype",
            "citance_function",
            "realizes_contribution",
            "rationale",
            "all model/proxy fields",
        ],
    }
    manifest_path = output / "sample_manifest.json"
    with manifest_path.open("x", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")

    print(f"Wrote four blinded pass files under {output}")
    print(f"Wrote manifest to {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
