#!/usr/bin/env python3
"""Hand-coding materials for the codebook-v2 freeze (29-30 Sep 2026).

Writes to annotations/generated/v2_prep/ (gitignored):

  rationale_check_50.csv      Codebook v2 §9.4. A seeded random 50 of the canonical
                              background_bulk x {partial, yes} edges, drawn from
                              outside every existing sample. Shows the rationale
                              only; the canonical cell is in the key file, so the
                              answer is not read off the label.
  rationale_check_50_key.csv  item -> edge_id, corpus, canonical cell. Open only
                              after all 50 are coded.
  anchors_8.csv               The 8 codebook anchors with the FULL citance set,
                              blank v2 fields. No canonical label, no proposal.
  v2_seen_edges.txt           Every edge whose canonical label or rationale the
                              lead has now seen. The 1 Oct v2 sample must exclude
                              these, alongside the frozen samples.
  manifest.json               Seed, source hashes, output hashes.

Draws nothing that is compared with any label; no metric is computed.
"""
from __future__ import annotations

import csv
import hashlib
import json
import random
from pathlib import Path

SEED = 20260928
N = 50
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "annotations/generated/v2_prep"
SOURCES = {
    "hci": ROOT / "yiren_artifacts/HCI contribution coding - hci coding (n=1040).csv",
    "nature_oa": ROOT / "yiren_artifacts/HCI contribution coding - nature_oa coding (n=1897).csv",
}
ANCHORS = ROOT / "annotations/generated/codebook_v2_anchors.json"
TEST_EDGES = ROOT / "analysis/results/external_systems/test_edges.csv"


def edge_id(corpus: str, row: dict) -> str:
    """Identical to analysis/ml/01_make_annotation_samples.py."""
    key = "\x1f".join((corpus, (row.get("cited_id") or "").strip(),
                       (row.get("citing_id") or "").strip()))
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:20]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        w = csv.DictWriter(handle, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def excluded_ids() -> dict[str, str]:
    """edge_id -> reason, for every edge already in a sample or already seen."""
    out: dict[str, str] = {}
    for path in sorted((ROOT / "annotations/generated").rglob("*.csv")):
        if OUT in path.parents:
            continue
        for row in read_csv(path):
            if row.get("edge_id"):
                out.setdefault(row["edge_id"], str(path.relative_to(ROOT)))
    for group in json.loads(ANCHORS.read_text(encoding="utf-8")).values():
        for e in group:
            out[e["edge_id"]] = "codebook_v2_anchor"
    for row in read_csv(TEST_EDGES):
        out.setdefault(row["edge_id"], "external_systems_test_edge")
    return out


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    canon = {c: read_csv(p) for c, p in SOURCES.items()}
    by_id = {edge_id(c, r): (c, r) for c, rows in canon.items() for r in rows}
    excluded = excluded_ids()

    # §9.4 pool: background_bulk x {partial, yes}, outside every sample.
    pool = sorted(
        eid for eid, (c, r) in by_id.items()
        # The sheet spells the function `motivates_background`; project documents
        # call the same cell `background_bulk` (counts match codebook §0).
        if r["citance_function"].strip() == "motivates_background"
        and r["realizes_contribution"].strip() in {"partial", "yes"}
        and eid not in excluded)
    picked = random.Random(SEED).sample(pool, N)

    sheet, key = [], []
    for i, eid in enumerate(picked, 1):
        c, r = by_id[eid]
        sheet.append({"item": f"R{i:02d}", "rationale": r["rationale"].strip(),
                      "asserts_use": "", "asserted_mode": "", "note": ""})
        key.append({"item": f"R{i:02d}", "edge_id": eid, "corpus": c,
                    "canonical_function": r["citance_function"].strip(),
                    "canonical_realization": r["realizes_contribution"].strip()})
    write_csv(OUT / "rationale_check_50.csv", sheet)
    write_csv(OUT / "rationale_check_50_key.csv", key)

    anchors = []
    for code, group in json.loads(ANCHORS.read_text(encoding="utf-8")).items():
        for e in group:
            c, r = by_id[e["edge_id"]]
            anchors.append({
                "anchor_group": code, "edge_id": e["edge_id"], "corpus": c,
                "citing_title": r["citing_title"].strip(),
                "cited_title": r["cited_title"].strip(),
                "n_citances": r["n_citances"].strip(),
                "citances": r["citing_sentence(s)"].strip(),
                "cited_contribution_type": "", "citation_function": "",
                "credits_contribution": "", "uptake_mode": "", "uptake_degree": "",
                "claim_dependent": "", "centrality": "", "rhetorical_polarity": "",
                "strongest_citance": "", "annotator_note": ""})
    write_csv(OUT / "anchors_8.csv", anchors)

    seen = sorted(set(picked) | {a["edge_id"] for a in anchors}
                  | {r["edge_id"] for r in read_csv(TEST_EDGES)})
    (OUT / "v2_seen_edges.txt").write_text("\n".join(seen) + "\n", encoding="utf-8")

    manifest = {
        "script": "analysis/experiments/20_make_v2_handcoding_sheets.py",
        "seed": SEED, "pool_size": len(pool), "n_drawn": N,
        "pool_rule": "canonical citance_function=motivates_background (background_bulk) x realizes_contribution in {partial, yes}, "
                     "excluding annotations/generated/**/*.csv, anchors, external test edges",
        "n_excluded_edges": len(excluded),
        "pool_by_corpus": {c: sum(by_id[e][0] == c for e in pool) for c in SOURCES},
        "drawn_by_corpus": {c: sum(by_id[e][0] == c for e in picked) for c in SOURCES},
        "sources": {str(p.relative_to(ROOT)): sha256(p)
                    for p in [*SOURCES.values(), ANCHORS, TEST_EDGES]},
        "outputs": {p.name: sha256(p) for p in sorted(OUT.iterdir())
                    if p.name != "manifest.json"},
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n",
                                       encoding="utf-8")
    print(json.dumps({k: manifest[k] for k in
                      ("pool_size", "n_excluded_edges", "pool_by_corpus",
                       "drawn_by_corpus")}, indent=1))
    print(f"anchors: {len(anchors)}   seen edges: {len(seen)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
