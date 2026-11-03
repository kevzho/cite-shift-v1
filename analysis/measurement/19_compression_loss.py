#!/usr/bin/env python3
"""Quantify realised contribution use absorbed by the background category.

For each corpus this reports how many human-coded citation edges carry the
``motivates_background`` function yet were judged to realise the cited
contribution (``yes`` or ``partial``). That intersection is the information a
compact background/method/result representation cannot express.

This is a crosswalk between two of Yiren's own human-coded columns. It
measures what the coded record shows; it is not a model evaluation and it is
not evidence that any published citation-intent system produced these labels.
The compact scheme is not separately applied to the corpus. This analysis
therefore measures distinctions the compact representation cannot express by
construction; it is not disagreement between two independent labellings.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "analysis" / "data_processing"))
from yiren_source import HCI_PATH, NATURE_PATH, TIER1, load

RESULTS = ROOT / "analysis" / "results" / "measurement"

BACKGROUND = "motivates_background"
REALISED = ("yes", "partial")


def norm(value: str | None) -> str:
    return " ".join((value or "").strip().lower().split()) or "unknown"


def pct(numerator: int, denominator: int) -> float:
    return round(100 * numerator / denominator, 2) if denominator else 0.0


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def overall_rows(corpora: dict[str, list[dict[str, str]]]) -> list[dict[str, object]]:
    output = []
    for corpus, rows in corpora.items():
        background = [r for r in rows if norm(r.get("citance_function")) == BACKGROUND]
        hidden = [r for r in background if norm(r.get("realizes_contribution")) in REALISED]
        core_hidden = [r for r in hidden if norm(r.get("bulk_or_core")) == "core"]
        output.append(
            {
                "corpus": corpus,
                "n_edges": len(rows),
                "n_background": len(background),
                "pct_background_of_corpus": pct(len(background), len(rows)),
                "n_hidden_realization": len(hidden),
                "pct_hidden_within_background": pct(len(hidden), len(background)),
                "pct_hidden_of_corpus": pct(len(hidden), len(rows)),
                "n_hidden_and_core": len(core_hidden),
                "pct_hidden_and_core_of_corpus": pct(len(core_hidden), len(rows)),
            }
        )
    return output


def by_contribution_rows(
    corpora: dict[str, list[dict[str, str]]]
) -> list[dict[str, object]]:
    output = []
    for corpus, rows in corpora.items():
        background = [r for r in rows if norm(r.get("citance_function")) == BACKGROUND]
        totals = Counter(norm(r.get(TIER1)) for r in background)
        hidden = Counter(
            norm(r.get(TIER1))
            for r in background
            if norm(r.get("realizes_contribution")) in REALISED
        )
        for label, total in sorted(totals.items(), key=lambda item: (-item[1], item[0])):
            output.append(
                {
                    "corpus": corpus,
                    "cited_contribution_tier1": label,
                    "n_background": total,
                    "n_hidden_realization": hidden[label],
                    "pct_hidden_within_background": pct(hidden[label], total),
                }
            )
    return output


def render_report(
    overall: list[dict[str, object]], by_type: list[dict[str, object]]
) -> str:
    lines = [
        "# Compression Loss In The Background Category",
        "",
        "Unit of analysis: one citing--cited citation edge. Source: Yiren's",
        "canonical human-coded sheets, read without relabelling.",
        "",
        "A citation is counted as *hidden realisation* when its coded function is",
        f"`{BACKGROUND}` and its coded realisation is `yes` or `partial`. A compact",
        "background/method/result representation has no slot for that combination.",
        "",
        "## Overall",
        "",
        "| Corpus | Edges | Background | Hidden realisation | Within background | Of corpus | Hidden and core |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in overall:
        lines.append(
            "| {corpus} | {n_edges} | {n_background} ({pct_background_of_corpus}%) | "
            "{n_hidden_realization} | {pct_hidden_within_background}% | "
            "{pct_hidden_of_corpus}% | {n_hidden_and_core} "
            "({pct_hidden_and_core_of_corpus}%) |".format(**row)
        )
    lines += [
        "",
        "## Hidden realisation within background edges, by cited contribution type",
        "",
        "| Corpus | Cited contribution (Tier-1) | Background | Hidden | Rate |",
        "|---|---|---:|---:|---:|",
    ]
    for row in by_type:
        lines.append(
            "| {corpus} | {cited_contribution_tier1} | {n_background} | "
            "{n_hidden_realization} | {pct_hidden_within_background}% |".format(**row)
        )
    lines += [
        "",
        "## Interpretation constraints",
        "",
        "- This is a crosswalk between two columns coded by the same annotator,",
        "  so the two judgements are not independent. Report it as coded-record",
        "  evidence until the dual-taxonomy audit provides independent passes.",
        "- Rare Tier-1 cells (`dataset`, `opinion`, `survey`) have very small",
        "  support and must be pooled or omitted rather than quoted as rates.",
        "- Edges are clustered within cited and citing papers; uncertainty must",
        "  be cluster-aware before any interval is published.",
        "- HCI and Nature OA differ in field, venue, sampling, and open-access",
        "  coverage. Read every contrast as observational and descriptive.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hci", type=Path, default=HCI_PATH)
    parser.add_argument("--nature", type=Path, default=NATURE_PATH)
    parser.add_argument("--results", type=Path, default=RESULTS)
    args = parser.parse_args()

    corpora = {"hci": load(args.hci), "nature_oa": load(args.nature)}

    overall = overall_rows(corpora)
    by_type = by_contribution_rows(corpora)

    write_csv(
        args.results / "compression_loss_overall.csv",
        overall,
        [
            "corpus",
            "n_edges",
            "n_background",
            "pct_background_of_corpus",
            "n_hidden_realization",
            "pct_hidden_within_background",
            "pct_hidden_of_corpus",
            "n_hidden_and_core",
            "pct_hidden_and_core_of_corpus",
        ],
    )
    write_csv(
        args.results / "compression_loss_by_contribution.csv",
        by_type,
        [
            "corpus",
            "cited_contribution_tier1",
            "n_background",
            "n_hidden_realization",
            "pct_hidden_within_background",
        ],
    )
    report = args.results / "compression_loss_report.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(render_report(overall, by_type), encoding="utf-8")

    for row in overall:
        print(
            f"{row['corpus']}: {row['n_hidden_realization']}/{row['n_edges']} edges "
            f"({row['pct_hidden_of_corpus']}%) are background-function but realise "
            f"the contribution; {row['pct_hidden_within_background']}% within background."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
