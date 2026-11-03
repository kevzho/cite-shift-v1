#!/usr/bin/env python3
"""Compare human-coded HCI and Nature OA citation edges on shared labels."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "analysis" / "data_processing"))
from yiren_source import HCI_PATH, NATURE_PATH, TIER1, load
RESULTS = ROOT / "analysis" / "results" / "measurement"
DEFAULT_HCI = HCI_PATH
DEFAULT_NATURE = NATURE_PATH


def norm(value: str | None) -> str:
    return " ".join((value or "").strip().lower().split()) or "unknown"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def pct(numerator: int, denominator: int) -> float:
    return round(100 * numerator / denominator, 2) if denominator else 0.0


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def distribution_rows(
    corpora: dict[str, list[dict[str, str]]], field: str
) -> list[dict[str, object]]:
    output = []
    for corpus, rows in corpora.items():
        counts = Counter(norm(row.get(field)) for row in rows)
        for label, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
            output.append({"corpus": corpus, "dimension": field, "label": label,
                           "n": count, "pct": pct(count, len(rows))})
    return output


def realization_by_contribution(
    corpora: dict[str, list[dict[str, str]]]
) -> list[dict[str, object]]:
    output = []
    for corpus, rows in corpora.items():
        labels = sorted({norm(row.get(TIER1)) for row in rows})
        for label in labels:
            selected = [row for row in rows if norm(row.get(TIER1)) == label]
            yes = sum(norm(row.get("realizes_contribution")) == "yes" for row in selected)
            partial = sum(norm(row.get("realizes_contribution")) == "partial" for row in selected)
            output.append({
                "corpus": corpus,
                "cited_contribution_tier1": label,
                "n": len(selected),
                "n_yes": yes,
                "n_partial": partial,
                "n_no": len(selected) - yes - partial,
                "pct_yes": pct(yes, len(selected)),
                "pct_yes_or_partial": pct(yes + partial, len(selected)),
            })
    return output


def function_realization_rows(
    corpora: dict[str, list[dict[str, str]]]
) -> list[dict[str, object]]:
    output = []
    for corpus, rows in corpora.items():
        counts = Counter(
            (norm(row.get("citance_function")), norm(row.get("realizes_contribution")))
            for row in rows
        )
        function_totals = Counter()
        for (function, _), count in counts.items():
            function_totals[function] += count
        for (function, realization), count in sorted(counts.items()):
            output.append({
                "corpus": corpus,
                "citance_function": function,
                "realizes_contribution": realization,
                "n": count,
                "pct_within_function": pct(count, function_totals[function]),
            })
    return output


def cramers_v(corpora: dict[str, list[dict[str, str]]], field: str) -> float:
    """Effect size for corpus × categorical-field association (no scipy)."""
    labels = sorted({norm(row.get(field)) for rows in corpora.values() for row in rows})
    table = [[sum(norm(row.get(field)) == label for row in rows) for label in labels]
             for rows in corpora.values()]
    total = sum(sum(row) for row in table)
    row_totals = [sum(row) for row in table]
    col_totals = [sum(table[i][j] for i in range(len(table))) for j in range(len(labels))]
    chi2 = 0.0
    for i, row in enumerate(table):
        for j, observed in enumerate(row):
            expected = row_totals[i] * col_totals[j] / total if total else 0
            if expected:
                chi2 += (observed - expected) ** 2 / expected
    denominator = total * min(len(table) - 1, len(labels) - 1)
    return round(math.sqrt(chi2 / denominator), 4) if denominator else 0.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hci", type=Path, default=DEFAULT_HCI)
    parser.add_argument("--nature", type=Path, default=DEFAULT_NATURE)
    args = parser.parse_args()
    for path in (args.hci, args.nature):
        if not path.exists():
            raise SystemExit(f"Missing Yiren source input: {path}")

    corpora = {"hci": load(args.hci), "nature_oa": load(args.nature)}
    required = {TIER1, "citance_function", "realizes_contribution", "bulk_or_core"}
    for corpus, rows in corpora.items():
        if not rows:
            raise SystemExit(f"{corpus} input has no rows")
        missing = required - set(rows[0])
        if missing:
            raise SystemExit(f"{corpus} input is missing columns: {sorted(missing)}")

    dimensions = ["realizes_contribution", "citance_function", TIER1, "bulk_or_core"]
    distributions = [row for field in dimensions for row in distribution_rows(corpora, field)]
    write_csv(
        RESULTS / "cross_domain_label_distributions.csv", distributions,
        ["corpus", "dimension", "label", "n", "pct"],
    )
    by_contribution = realization_by_contribution(corpora)
    write_csv(
        RESULTS / "cross_domain_realization_by_contribution.csv", by_contribution,
        ["corpus", "cited_contribution_tier1", "n", "n_yes", "n_partial", "n_no", "pct_yes", "pct_yes_or_partial"],
    )
    function_realization = function_realization_rows(corpora)
    write_csv(
        RESULTS / "cross_domain_function_by_realization.csv", function_realization,
        ["corpus", "citance_function", "realizes_contribution", "n", "pct_within_function"],
    )

    dist_lookup = {
        (row["corpus"], row["dimension"], row["label"]): row
        for row in distributions
    }
    def value(corpus: str, dimension: str, label: str) -> float:
        return float(dist_lookup.get((corpus, dimension, label), {}).get("pct", 0.0))

    report = [
        "# Cross-Domain Contribution Comparison",
        "",
        f"This edge-level comparison uses {len(corpora['hci']):,} human-coded HCI citations and {len(corpora['nature_oa']):,} human-coded Nature OA citations. It reads Yiren's original shared Tier-1/function/realization labels directly.",
        "",
        "## Headline differences",
        "",
        f"- Strong realization (`yes`): HCI {value('hci', 'realizes_contribution', 'yes')}% vs Nature OA {value('nature_oa', 'realizes_contribution', 'yes')}% ({round(value('nature_oa', 'realizes_contribution', 'yes') - value('hci', 'realizes_contribution', 'yes'), 2):+} percentage points Nature OA minus HCI).",
        f"- Motivates-background function: HCI {value('hci', 'citance_function', 'motivates_background')}% vs Nature OA {value('nature_oa', 'citance_function', 'motivates_background')}%.",
        f"- Core citations: HCI {value('hci', 'bulk_or_core', 'core')}% vs Nature OA {value('nature_oa', 'bulk_or_core', 'core')}%.",
        f"- Artifact contributions: HCI {value('hci', TIER1, 'artifact')}% vs Nature OA {value('nature_oa', TIER1, 'artifact')}%.",
        "",
        "## Association effect sizes",
        "",
        "Cramér's V summarizes the magnitude of association between corpus and each categorical dimension. It is an effect size, not a significance test.",
        "",
        "| Dimension | Cramér's V |",
        "|---|---:|",
        *[f"| `{field}` | {cramers_v(corpora, field)} |" for field in dimensions],
        "",
        "## Interpretation constraints",
        "",
        "- These are observational edge distributions, not causal estimates. Corpus sampling and venue composition differ.",
        "- Tier-2 subtypes are domain-specific and are intentionally not compared label-for-label.",
        "- Rhetorical polarity is project-inferred rather than human-coded and is excluded from the primary comparison.",
        "- HCI lacks S2 fields-of-study coverage, so no matched subfield analysis is attempted.",
        "- The full storyline dataset is still absent; these results do not measure chain-level trajectories.",
        "",
        "## Output tables",
        "",
        "- `cross_domain_label_distributions.csv`",
        "- `cross_domain_realization_by_contribution.csv`",
        "- `cross_domain_function_by_realization.csv`",
        "",
    ]
    (RESULTS / "cross_domain_comparison_report.md").write_text("\n".join(report), encoding="utf-8")
    print(f"Compared {len(corpora['hci'])} HCI and {len(corpora['nature_oa'])} Nature OA edges")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
