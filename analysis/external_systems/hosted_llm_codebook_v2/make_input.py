#!/usr/bin/env python3
"""Build a --input CSV for run_hosted.py from the canonical edge files: edge_id, corpus, cited_id,
citing_id, citing_title, cited_title, citances. Copies NO label columns (pre-registration).

By default drops the eight codebook-v2 anchor edges. Which edges form the evaluation set is a
pre-registration decision; restrict with --only-ids (one edge_id per line) once it is fixed.
"""
import argparse, csv, hashlib, json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
KEEP = ["cited_id", "citing_id", "citing_title", "cited_title", "citances"]


def edge_id(corpus, cited, citing):  # scheme of analysis/measurement/28_cut_validation.py (edge_id_v1)
    return hashlib.sha256("\x1f".join((corpus, cited.strip(), citing.strip())).encode()).hexdigest()[:20]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "analysis/results/external_systems/hosted_llm_codebook_v2/input_canonical.csv"))
    ap.add_argument("--only-ids")
    ap.add_argument("--keep-anchors", action="store_true")
    a = ap.parse_args()
    anchors = set()
    if not a.keep_anchors:
        anchors = {x["edge_id"] for v in json.load(open(ROOT / "annotations/generated/codebook_v2_anchors.json")).values() for x in v}
    only = set(Path(a.only_ids).read_text().split()) if a.only_ids else None
    rows = []
    for corpus in ("hci", "nature_oa"):
        for r in csv.DictReader(open(ROOT / f"analysis/derived/{corpus}_contribution_coding_human_reviewed.csv", encoding="utf-8")):
            e = edge_id(corpus, r["cited_id"], r["citing_id"])
            if e in anchors or (only is not None and e not in only):
                continue
            rows.append({"edge_id": e, "corpus": corpus, **{k: r[k] for k in KEEP}})
    with open(a.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["edge_id", "corpus"] + KEEP)
        w.writeheader(); w.writerows(rows)
    print(f"{len(rows)} edges -> {a.out} (anchors dropped: {len(anchors) and not a.keep_anchors})")


if __name__ == "__main__":
    main()
