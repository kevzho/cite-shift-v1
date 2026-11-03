#!/usr/bin/env python3
"""Measure axis independence and pairwise agreement across annotation records.

The project's scheme separates *citation function* (why a work is cited) from
*contribution realisation* (whether the citing work uses it).  That separation
only exists in a record if the two columns carry independent information.  This
program measures that independence per record with Cramer's V and normalised
mutual information, and reports pairwise Cohen's kappa under the label
crosswalk.

Unlike kappa, the independence statistics require only ONE record, so they are
computable for the canonical corpus and for any single coder submission.

This is a descriptive measurement, not an adjudication: a low-independence
record is not thereby wrong, and no record is treated as gold.

Run from ``citation_corpus``:

    python3 analysis/verification/axis_independence_validation.py
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "analysis/verification/axis_independence_validation_results.json"

INPUTS = {
    "canonical_hci": ROOT / "analysis/derived/hci_contribution_coding_human_reviewed.csv",
    "canonical_nature_oa": ROOT / "analysis/derived/nature_oa_contribution_coding_human_reviewed.csv",
    "coder_a_validation": ROOT / "annotations/coders/validation/annotator_a_validation_multidimensional_resubmit.csv",
    "automated_pass_validation": ROOT / "annotations/coders/validation/annotator_b_validation_multidimensional.csv",
    "coder_a_round_two": ROOT / "annotations/coders/round_two/annotator_a_multidimensional.csv",
    "coder_b_round_two": ROOT / "annotations/coders/round_two/annotator_b_multidimensional.csv",
}

# Hash-pinned after the first clean derivation on 13 September 2026.
EXPECTED_INPUT_SHA256 = {
    "analysis/derived/hci_contribution_coding_human_reviewed.csv": "b71c2c5c905a2f9aa91eb2a9da52a2340a483aba3ab5bf6fe3cf4545ff404e5e",
    "analysis/derived/nature_oa_contribution_coding_human_reviewed.csv": "acd497e933934bba6777867ee14649f7395d4f4ba18b134bb9f9273cf8a8780a",
    "annotations/coders/validation/annotator_a_validation_multidimensional_resubmit.csv": "4473daaa9502ef53f80978970392d5a36fa6a759202b72de9426329def50b245",
    "annotations/coders/validation/annotator_b_validation_multidimensional.csv": "ac8067805368affde540cecb3d88ad2362374b06242c4f82b9b308cbc7149262",
    "annotations/coders/round_two/annotator_a_multidimensional.csv": "78ad81556be93daf00bdb8d0d64c77d8e0a68038114e791772c4d4da504b5748",
    "annotations/coders/round_two/annotator_b_multidimensional.csv": "e66fec094e0c202432a8bac154311ef673e6ab55418c6dba46f15a1f7498ef7e",
}

CANONICAL_COLUMNS = ("citance_function", "realizes_contribution")
CODEBOOK_COLUMNS = ("citation_function", "contribution_realization")
BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = 20260913

# annotations/label_crosswalk.csv, mechanical rows only.
FUNCTION_CROSSWALK = {
    "motivates_background": "background_bulk",
    "builds_on_finding": "uses_finding_as_evidence",
    "compares_against": "compares_or_contrasts",
    "applies_framework": "applies_framework",
    "reuses_artifact": "system_reuse_or_extension",
    "uses_method": "uses_method",
    "unclear": "unclear",
}
REALIZATION_CROSSWALK = {"yes": "yes", "partial": "partial", "no_background": "no"}
# `other` (function) and `unclear` (realisation) have no canonical target.
NON_MAPPABLE_NOTE = "Rows whose codebook label has no canonical target are excluded from human-canonical comparison; the exclusion count and rate are reported."


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def cramers_v(pairs: list[tuple[str, str]]) -> float:
    xs = sorted({a for a, _ in pairs})
    ys = sorted({b for _, b in pairs})
    n = len(pairs)
    k = min(len(xs), len(ys))
    if k < 2:
        return float("nan")
    joint = Counter(pairs)
    mx, my = Counter(a for a, _ in pairs), Counter(b for _, b in pairs)
    chi2 = 0.0
    for x in xs:
        for y in ys:
            expected = mx[x] * my[y] / n
            if expected > 0:
                chi2 += (joint.get((x, y), 0) - expected) ** 2 / expected
    return math.sqrt(chi2 / (n * (k - 1)))


def normalised_mutual_information(pairs: list[tuple[str, str]]) -> float:
    """I(X;Y) / min(H(X), H(Y)).  1.0 means one column determines the other."""
    n = len(pairs)
    joint = Counter(pairs)
    mx, my = Counter(a for a, _ in pairs), Counter(b for _, b in pairs)
    entropy = lambda counts: -sum(v / n * math.log(v / n) for v in counts.values() if v)
    information = sum(
        v / n * math.log((v / n) / ((mx[a] / n) * (my[b] / n)))
        for (a, b), v in joint.items()
        if v
    )
    floor = min(entropy(mx), entropy(my))
    return information / floor if floor > 0 else float("nan")


def modal_determinacy(pairs: list[tuple[str, str]]) -> dict[str, Any]:
    """Share of edges whose realisation equals the modal realisation of its function."""
    table: dict[str, Counter] = defaultdict(Counter)
    for function, realization in pairs:
        table[function][realization] += 1
    hits = sum(max(counts.values()) for counts in table.values())
    return {"determined": hits, "total": len(pairs), "pct": 100.0 * hits / len(pairs)}


def percentile(sorted_values: list[float], probability: float) -> float:
    """Linear-interpolated percentile, fixed before bootstrap computation."""
    if not sorted_values:
        return float("nan")
    position = (len(sorted_values) - 1) * probability
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def bootstrap_intervals(pairs: list[tuple[str, str]], *, seed: int) -> dict[str, Any]:
    """Within-record nonparametric percentile bootstrap for association metrics.

    The unit is one completed citation edge in the record.  These intervals
    describe sampling variation conditional on that annotation record; they do
    not compare records as if their sample sizes or protocols were matched.
    """
    generator = random.Random(seed)
    n = len(pairs)
    values_v: list[float] = []
    values_nmi: list[float] = []
    for _ in range(BOOTSTRAP_DRAWS):
        draw = [pairs[generator.randrange(n)] for _ in range(n)]
        v = cramers_v(draw)
        nmi = normalised_mutual_information(draw)
        if math.isfinite(v):
            values_v.append(v)
        if math.isfinite(nmi):
            values_nmi.append(nmi)
    values_v.sort()
    values_nmi.sort()
    return {
        "method": "nonparametric within-record percentile bootstrap",
        "unit": "completed citation edge",
        "draws": BOOTSTRAP_DRAWS,
        "seed": seed,
        "percentile_convention": "linear interpolation of sorted draws at 0.025 and 0.975",
        "cramers_v_95_ci": [percentile(values_v, 0.025), percentile(values_v, 0.975)],
        "normalised_mutual_information_95_ci": [
            percentile(values_nmi, 0.025), percentile(values_nmi, 0.975)
        ],
    }


def cohens_kappa(pairs: list[tuple[str, str]]) -> dict[str, Any]:
    n = len(pairs)
    categories = sorted({x for pair in pairs for x in pair})
    observed = sum(a == b for a, b in pairs) / n
    ma, mb = Counter(a for a, _ in pairs), Counter(b for _, b in pairs)
    expected = sum(ma[c] * mb[c] for c in categories) / n**2
    kappa = (observed - expected) / (1 - expected) if expected < 1 else float("nan")
    return {
        "n": n,
        "raw_agreement": observed,
        "expected_agreement": expected,
        "kappa": kappa,
        "marginals_a": dict(ma),
        "marginals_b": dict(mb),
    }


def note_templating(rows: list[dict[str, str]]) -> dict[str, Any]:
    """Provenance signal: generated notes collapse to few skeletons and repeat sentences."""
    notes = [(row.get("annotator_note") or "").strip() for row in rows]
    populated = [n for n in notes if n]
    if not populated:
        return {"notes_populated": 0}

    def skeleton(note: str) -> str:
        s = re.sub(r'"[^"]*"|“[^”]*”', ' "Q" ', note)
        s = re.sub(r"\b[a-z]+_[a-z_]+\b", "LABEL", s)
        s = re.sub(r"Confidence:\s*\w+", "Confidence: C", s)
        return re.sub(r"\s+", " ", s).strip()

    sentences: Counter = Counter()
    for note in populated:
        for part in re.split(r"(?<=[.;])\s+", note):
            part = re.sub(r'"[^"]*"', '"Q"', part).strip()
            if len(part) > 25:
                sentences[part] += 1
    top = sentences.most_common(1)[0] if sentences else ("", 0)
    return {
        "notes_populated": len(populated),
        "rows": len(rows),
        "distinct_skeletons": len(set(skeleton(n) for n in populated)),
        "mean_note_chars": sum(map(len, populated)) / len(populated),
        "max_sentence_repetition": top[1],
        "max_sentence_repetition_share": top[1] / len(populated),
        "most_repeated_sentence": top[0][:160],
    }


def edge_key(corpus: str, cited_id: str, citing_id: str) -> str:
    return hashlib.sha256(f"{corpus}\x1f{cited_id}\x1f{citing_id}".encode()).hexdigest()[:20]


def main() -> int:
    hashes = {}
    for name, path in INPUTS.items():
        digest = sha256(path)
        hashes[str(path.relative_to(ROOT))] = digest
        expected = EXPECTED_INPUT_SHA256.get(str(path.relative_to(ROOT)))
        if expected and expected != digest:
            raise SystemExit(f"input hash mismatch for {name}: {digest}")

    canonical_rows: dict[str, dict[str, str]] = {}
    canonical_pairs: list[tuple[str, str]] = []
    for corpus, key in (("hci", "canonical_hci"), ("nature_oa", "canonical_nature_oa")):
        for row in load(INPUTS[key]):
            canonical_rows[edge_key(corpus, row["cited_id"], row["citing_id"])] = row
            function, realization = row[CANONICAL_COLUMNS[0]], row[CANONICAL_COLUMNS[1]]
            if function and realization:
                canonical_pairs.append((function, realization))

    records: dict[str, list[tuple[str, str]]] = {"canonical_full_corpus": canonical_pairs}
    raw: dict[str, list[dict[str, str]]] = {}
    for name in ("coder_a_validation", "automated_pass_validation", "coder_a_round_two", "coder_b_round_two"):
        rows = load(INPUTS[name])
        raw[name] = rows
        records[name] = [
            (r[CODEBOOK_COLUMNS[0]], r[CODEBOOK_COLUMNS[1]])
            for r in rows
            if r.get(CODEBOOK_COLUMNS[0]) and r.get(CODEBOOK_COLUMNS[1])
        ]

    independence = {
        name: {
            "n": len(pairs),
            "cramers_v": cramers_v(pairs),
            "normalised_mutual_information": normalised_mutual_information(pairs),
            "modal_determinacy": modal_determinacy(pairs),
            "bootstrap": bootstrap_intervals(
                pairs, seed=BOOTSTRAP_SEED + index
            ),
            "function_by_realization": {
                f: dict(Counter(b for a, b in pairs if a == f))
                for f in sorted({a for a, _ in pairs})
            },
        }
        for index, (name, pairs) in enumerate(records.items())
    }

    # --- pairwise agreement -------------------------------------------------
    agreement: dict[str, Any] = {}

    for dimension, coder_column, canonical_column, crosswalk in (
        ("function", CODEBOOK_COLUMNS[0], CANONICAL_COLUMNS[0], FUNCTION_CROSSWALK),
        ("realization", CODEBOOK_COLUMNS[1], CANONICAL_COLUMNS[1], REALIZATION_CROSSWALK),
    ):
        pairs, excluded, unmatched = [], 0, 0
        for row in raw["coder_a_validation"]:
            canonical = canonical_rows.get(row["edge_id"])
            if canonical is None:
                unmatched += 1
                continue
            coder_value = (row.get(coder_column) or "").strip()
            canonical_value = (canonical.get(canonical_column) or "").strip()
            if coder_value not in crosswalk or not canonical_value:
                excluded += 1
                continue
            pairs.append((crosswalk[coder_value], canonical_value))
        agreement[f"coder_a_vs_canonical_{dimension}"] = {
            **cohens_kappa(pairs),
            "excluded_non_mappable": excluded,
            "unmatched_edges": unmatched,
            "note": NON_MAPPABLE_NOTE,
        }

    by_edge = {r["edge_id"]: r for r in raw["automated_pass_validation"]}
    for dimension, column in (("function", CODEBOOK_COLUMNS[0]), ("realization", CODEBOOK_COLUMNS[1])):
        pairs = [
            ((row.get(column) or "").strip(), (by_edge[row["edge_id"]].get(column) or "").strip())
            for row in raw["coder_a_validation"]
            if row["edge_id"] in by_edge
        ]
        agreement[f"coder_a_vs_automated_pass_{dimension}"] = cohens_kappa([p for p in pairs if all(p)])

    b_round_two = {r["edge_id"]: r for r in raw["coder_b_round_two"]}
    for dimension, column in (("function", CODEBOOK_COLUMNS[0]), ("realization", CODEBOOK_COLUMNS[1])):
        pairs = [
            ((row.get(column) or "").strip(), (b_round_two[row["edge_id"]].get(column) or "").strip())
            for row in raw["coder_a_round_two"]
            if row["edge_id"] in b_round_two
        ]
        agreement[f"human_human_round_two_{dimension}"] = cohens_kappa([p for p in pairs if all(p)])

    results = {
        "analysis": "axis independence and pairwise agreement across annotation records",
        "unit": "one citing-cited edge",
        "definition": {
            "axis_independence": "association between the function column and the realisation column within one record; NMI = I(function;realisation)/min(H(function),H(realisation)); 1.0 means the lower-entropy column is determined by the other. Modal determinacy separately reports the directed function-to-realisation mapping.",
            "bootstrap": "10,000-draw nonparametric percentile bootstrap within each annotation record; intervals describe sampling variation conditional on that record and are not precision-matched cross-record comparisons.",
            "modal_determinacy": "share of edges whose realisation equals the modal realisation of its own function label",
            "crosswalk": "annotations/label_crosswalk.csv, mechanical rows only",
        },
        "input_sha256": hashes,
        "axis_independence": independence,
        "pairwise_agreement": agreement,
        "note_provenance": {name: note_templating(rows) for name, rows in raw.items()},
        "interpretation_boundary": "Axis independence describes how a record was produced, not whether its judgements are correct. No record is treated as gold, and these values must not be averaged across records.",
    }
    OUTPUT.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {OUTPUT}\n")
    print(f"{'record':28s} {'n':>6s} {'V':>7s} {'NMI':>7s} {'determinacy':>12s}")
    for name, stats in independence.items():
        print(f"{name:28s} {stats['n']:6d} {stats['cramers_v']:7.3f} "
              f"{stats['normalised_mutual_information']:7.3f} {stats['modal_determinacy']['pct']:11.1f}%")
    print()
    for name, stats in agreement.items():
        print(f"{name:44s} n={stats['n']:4d} raw={stats['raw_agreement']:.3f} kappa={stats['kappa']:.3f}")
    print()
    for name, stats in results["note_provenance"].items():
        if stats.get("notes_populated"):
            print(f"{name:28s} skeletons={stats['distinct_skeletons']:4d} "
                  f"max_sentence_repeat={stats['max_sentence_repetition_share']:.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
