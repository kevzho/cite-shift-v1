#!/usr/bin/env python3
"""Deterministic tests for the retained iConference measurement rules."""

import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


def load(name: str, relative_path: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


cross_domain = load("cross_domain", "analysis/measurement/06_cross_domain_comparison.py")
compression = load("compression", "analysis/measurement/19_compression_loss.py")


def edge(function: str, realization: str, tier1: str = "empirical", hierarchy: str = "bulk"):
    return {
        "citance_function": function,
        "realizes_contribution": realization,
        "bulk_or_core": hierarchy,
        "cited_contribution (Tier-1)": tier1,
    }


class CrossDomainRules(unittest.TestCase):
    def test_distribution_keeps_corpora_separate(self):
        rows = cross_domain.distribution_rows(
            {"hci": [{"label": "a"}, {"label": "b"}], "nature_oa": [{"label": "a"}]},
            "label",
        )
        lookup = {(row["corpus"], row["label"]): row["pct"] for row in rows}
        self.assertEqual(lookup[("hci", "a")], 50.0)
        self.assertEqual(lookup[("nature_oa", "a")], 100.0)

    def test_cramers_v_zero_for_identical_distributions(self):
        corpora = {
            "hci": [{"label": "a"}, {"label": "b"}],
            "nature_oa": [{"label": "a"}, {"label": "b"}],
        }
        self.assertEqual(cross_domain.cramers_v(corpora, "label"), 0.0)


class CompressionLossRules(unittest.TestCase):
    """The headline measure: background-function edges that nonetheless realise."""

    def setUp(self):
        self.corpora = {
            "hci": [
                edge("motivates_background", "yes", "empirical", "core"),
                edge("motivates_background", "partial", "artifact", "bulk"),
                edge("motivates_background", "no_background", "empirical"),
                edge("uses_method", "yes", "methodological"),
            ]
        }

    def test_hidden_realization_requires_both_conditions(self):
        row = compression.overall_rows(self.corpora)[0]
        # Three background edges, two of which realise; the uses_method edge
        # realises but is not hidden because it is not filed under background.
        self.assertEqual(row["n_edges"], 4)
        self.assertEqual(row["n_background"], 3)
        self.assertEqual(row["n_hidden_realization"], 2)
        self.assertEqual(row["n_hidden_and_core"], 1)

    def test_counts_nest_inside_their_denominators(self):
        row = compression.overall_rows(self.corpora)[0]
        self.assertLessEqual(row["n_hidden_and_core"], row["n_hidden_realization"])
        self.assertLessEqual(row["n_hidden_realization"], row["n_background"])
        self.assertLessEqual(row["n_background"], row["n_edges"])

    def test_within_background_rate_uses_background_denominator(self):
        row = compression.overall_rows(self.corpora)[0]
        self.assertEqual(row["pct_hidden_within_background"], round(100 * 2 / 3, 2))
        self.assertEqual(row["pct_hidden_of_corpus"], 50.0)

    def test_by_contribution_partitions_the_background_edges(self):
        overall = compression.overall_rows(self.corpora)[0]
        by_type = compression.by_contribution_rows(self.corpora)
        self.assertEqual(
            sum(row["n_background"] for row in by_type), overall["n_background"]
        )
        self.assertEqual(
            sum(row["n_hidden_realization"] for row in by_type),
            overall["n_hidden_realization"],
        )

    def test_realization_labels_are_normalised(self):
        corpora = {"hci": [edge("Motivates_Background", " Partial ")]}
        self.assertEqual(compression.overall_rows(corpora)[0]["n_hidden_realization"], 1)


if __name__ == "__main__":
    unittest.main()
