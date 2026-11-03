#!/usr/bin/env python3
"""Freeze the annotation codebooks: validate, hash, and record a version.

Workstream A's gate is "hash recorded". This script does the mechanical half:

  1. validates `annotations/label_crosswalk.csv` against the canonical CSVs --
     every canonical value must be reachable from some codebook label, and no
     crosswalk row may point at a value the data does not contain;
  2. refuses to freeze while any crosswalk row is still NEEDS_DECISION, unless
     --allow-unresolved is passed (which records them in the manifest as open);
  3. hashes each codebook file and writes
     `annotations/CODEBOOK_FREEZE.json` with the SHA256s, sizes, UTC timestamp,
     and git commit.

It does NOT check that the codebooks say the right things. Construct
definitions, anchors, the `unclear` rule, and the reliability thresholds are
the human author's work (decision D40) and this script cannot verify them.

Usage:
    python3 analysis/measurement/23_freeze_codebooks.py            # validate only
    python3 analysis/measurement/23_freeze_codebooks.py --freeze   # write manifest

AI-ASSISTED CODE. Per decision D16 / issue I06 this script must be
independently reconstructed by a human author before its output is relied on.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ANN = ROOT / "annotations"
DERIVED = ROOT / "analysis" / "derived"

CODEBOOKS = [
    ANN / "ANNOTATION_CODEBOOK_DRAFT.md",
    ANN / "POLARITY_CODEBOOK_DRAFT.md",
]
CROSSWALK = ANN / "label_crosswalk.csv"
MANIFEST = ANN / "CODEBOOK_FREEZE.json"

# crosswalk dimension -> column in the canonical coding CSVs
CANONICAL_COLUMN = {
    "research_contribution": "research_contribution",
    "citance_function": "citance_function",
    "realizes_contribution": "realizes_contribution",
    "bulk_or_core": "bulk_or_core",
    "rhetorical_polarity": "rhetorical_polarity",
}

SOURCES = {
    "hci": DERIVED / "hci_contribution_coding_human_reviewed.csv",
    "nature_oa": DERIVED / "nature_oa_contribution_coding_human_reviewed.csv",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def canonical_values() -> dict[str, set[str]]:
    """Distinct non-null values per dimension, pooled across both corpora."""
    seen: dict[str, set[str]] = {d: set() for d in CANONICAL_COLUMN}
    for path in SOURCES.values():
        with path.open(newline="", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                for dim, col in CANONICAL_COLUMN.items():
                    v = (row.get(col) or "").strip()
                    if v:
                        seen[dim].add(v)
    return seen


def load_crosswalk() -> list[dict]:
    with CROSSWALK.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def validate(rows: list[dict], canon: dict[str, set[str]]) -> tuple[list[str], list[dict]]:
    problems: list[str] = []
    unresolved = [r for r in rows if r["status"].strip() == "NEEDS_DECISION"]

    dims_in_crosswalk = {r["dimension"] for r in rows}
    for dim in CANONICAL_COLUMN:
        if dim not in dims_in_crosswalk:
            problems.append(f"dimension '{dim}' missing from the crosswalk entirely")

    for r in rows:
        dim, target = r["dimension"], r["canonical_value"].strip()
        if dim not in canon:
            problems.append(f"unknown dimension '{dim}' in crosswalk")
            continue
        if target and target not in canon[dim]:
            problems.append(
                f"{dim}: crosswalk maps '{r['codebook_label']}' to "
                f"'{target}', which does not occur in the canonical data"
            )

    # every canonical value must be reachable, or the students can produce no
    # label that matches it and the canonical-vs-adjudicated comparison (plan
    # Workstream B item 3) is undefined for that class.
    for dim, values in canon.items():
        mapped = {r["canonical_value"].strip() for r in rows if r["dimension"] == dim}
        for v in sorted(values - mapped):
            problems.append(
                f"{dim}: canonical value '{v}' is unreachable — no codebook "
                f"label maps to it"
            )

    return problems, unresolved


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--freeze", action="store_true", help="write the manifest")
    ap.add_argument(
        "--allow-unresolved",
        action="store_true",
        help="freeze even with NEEDS_DECISION rows, recording them as open",
    )
    args = ap.parse_args()

    missing = [p for p in CODEBOOKS + [CROSSWALK] if not p.exists()]
    if missing:
        for p in missing:
            print(f"MISSING: {p.relative_to(ROOT)}", file=sys.stderr)
        return 2

    canon = canonical_values()
    rows = load_crosswalk()
    problems, unresolved = validate(rows, canon)

    print(f"crosswalk: {len(rows)} rows across {len({r['dimension'] for r in rows})} dimensions")
    for dim, values in sorted(canon.items()):
        print(f"  {dim}: {len(values)} canonical values")

    if problems:
        print(f"\n{len(problems)} validation problem(s):")
        for p in problems:
            print(f"  - {p}")

    if unresolved:
        print(f"\n{len(unresolved)} row(s) still NEEDS_DECISION — these are yours:")
        for r in unresolved:
            print(f"  - {r['dimension']}.{r['codebook_label']}: {r['note']}")

    if not args.freeze:
        print("\n(validate only; pass --freeze to write the manifest)")
        return 1 if problems else 0

    if problems:
        print("\nREFUSING TO FREEZE: fix the validation problems first.", file=sys.stderr)
        return 2
    if unresolved and not args.allow_unresolved:
        print(
            "\nREFUSING TO FREEZE: resolve the NEEDS_DECISION rows, or pass "
            "--allow-unresolved to record them as open in the manifest.",
            file=sys.stderr,
        )
        return 2

    manifest = {
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "files": {
            p.relative_to(ROOT).as_posix(): {
                "sha256": sha256(p),
                "bytes": p.stat().st_size,
            }
            for p in CODEBOOKS + [CROSSWALK]
        },
        "canonical_values": {k: sorted(v) for k, v in canon.items()},
        "unresolved_crosswalk_rows": [
            {"dimension": r["dimension"], "codebook_label": r["codebook_label"]}
            for r in unresolved
        ],
        "note": (
            "Mechanical freeze record only. It attests to file contents at a "
            "point in time, not to the adequacy of the construct definitions, "
            "anchors, unclear rule, or reliability thresholds."
        ),
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"\nwrote {MANIFEST.relative_to(ROOT)}")
    for name, info in manifest["files"].items():
        print(f"  {info['sha256'][:16]}…  {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
