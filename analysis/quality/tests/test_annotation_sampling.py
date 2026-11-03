#!/usr/bin/env python3
"""Tests for blinded, deterministic annotation sampling."""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "analysis/ml/01_make_annotation_samples.py"
SPEC = importlib.util.spec_from_file_location("annotation_sampling", SCRIPT)
sampling = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(sampling)


def row(i: int) -> dict[str, str]:
    return {
        "cited_id": f"cited-{i}",
        "citing_id": f"citing-{i}",
        "cited_title": f"Cited {i}",
        "citing_title": f"Citing {i}",
        "citing_sentence(s)": f"Context {i}",
        "citance_function": "hidden label",
        "realizes_contribution": "hidden label",
        "rationale": "hidden rationale",
    }


class AnnotationSamplingRules(unittest.TestCase):
    def test_sampling_is_deterministic_and_disjoint(self):
        rows = [row(i) for i in range(30)]
        first = sampling.choose(rows, calibration_n=5, validation_n=10, seed=7)
        second = sampling.choose(rows, calibration_n=5, validation_n=10, seed=7)
        self.assertEqual(first, second)
        calibration, validation = first
        self.assertFalse(
            {item["citing_id"] for item in calibration}
            & {item["citing_id"] for item in validation}
        )

    def test_blinding_excludes_existing_label_values_and_corpus(self):
        compact = sampling.compact_row("hci", row(1))
        multidimensional = sampling.multidimensional_row("hci", row(1))
        self.assertEqual(set(compact), set(sampling.COMPACT_COLUMNS))
        self.assertEqual(
            set(multidimensional), set(sampling.MULTIDIMENSIONAL_COLUMNS)
        )
        self.assertEqual(compact["compact_intent"], "")
        self.assertEqual(multidimensional["citation_function"], "")
        self.assertEqual(multidimensional["contribution_realization"], "")
        self.assertNotIn("rationale", multidimensional)
        self.assertNotIn("corpus", multidimensional)

    def test_edge_id_is_stable_and_corpus_specific(self):
        item = row(1)
        self.assertEqual(
            sampling.edge_id("hci", item), sampling.edge_id("hci", item)
        )
        self.assertNotEqual(
            sampling.edge_id("hci", item), sampling.edge_id("nature_oa", item)
        )

    def test_invalid_sample_sizes_fail(self):
        with self.assertRaises(ValueError):
            sampling.choose([row(1)], calibration_n=1, validation_n=1, seed=1)


if __name__ == "__main__":
    unittest.main()
