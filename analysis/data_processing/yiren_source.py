"""Read Yiren's coding sheets without changing their coding vocabulary.

The sheets in ``yiren_artifacts/`` are the canonical human-coded inputs.
This module only adds convenience fields needed to join the legacy pilot
signals (S2/LLM) and to refer to the citation text consistently in code.  It
does not rename or replace Yiren's Tier-1, function, or realization labels.
"""
from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
YIREN_DIR = ROOT / "yiren_artifacts"
PILOT_PATH = ROOT / "hci_contribution_coding.csv"
HCI_PATH = YIREN_DIR / "HCI contribution coding - hci coding (n=1040).csv"
NATURE_PATH = YIREN_DIR / "HCI contribution coding - nature_oa coding (n=1897).csv"

TIER1 = "cited_contribution (Tier-1)"
TIER2 = "contribution_subtype (Tier-2)"
CITANCE = "citing_sentence(s)"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def pilot_lookup() -> dict[tuple[str, str], dict[str, str]]:
    if not PILOT_PATH.exists():
        return {}
    return {
        ((row.get("cited_id") or "").strip(), (row.get("citing_id") or "").strip()): row
        for row in read_csv(PILOT_PATH)
    }


def load(path: Path, *, attach_pilot_signals: bool = False) -> list[dict[str, str]]:
    """Load an original Yiren sheet, optionally attaching legacy signal fields.

    The original columns remain intact. ``citances`` is a convenience copy of
    Yiren's ``citing_sentence(s)`` for prompt/retrieval code; it is not a new
    annotation. Legacy fields are blank when no exact pilot edge matches.
    """
    rows = [dict(row) for row in read_csv(path)]
    lookup = pilot_lookup() if attach_pilot_signals else {}
    for row in rows:
        row["citances"] = row.get(CITANCE, "")
        if not attach_pilot_signals:
            continue
        pilot = lookup.get(((row.get("cited_id") or "").strip(), (row.get("citing_id") or "").strip()), {})
        for field in ("s2_intents", "isinfluential", "llm_is_genuine", "llm_buildon_type", "llm_grounded_in", "cited_venue", "component_id", "edge_idx"):
            row[field] = (pilot.get(field) or "").strip()
    return rows


def load_hci(*, attach_pilot_signals: bool = False) -> list[dict[str, str]]:
    return load(HCI_PATH, attach_pilot_signals=attach_pilot_signals)


def load_nature() -> list[dict[str, str]]:
    return load(NATURE_PATH)
