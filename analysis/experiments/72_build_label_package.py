#!/usr/bin/env python3
"""Build the public label package: the lead author's own human labels, keyed by edge, with no citation text, no
titles and no canonical labels. Edges are identified by corpus and Semantic Scholar corpus IDs of the citing and
cited paper, so others can re-fetch the text themselves.

Also included: the canonical record's categorical labels (2,937 edges, coded by Yiren Liu), labels only (no
rationales, notes, citances or titles), and the v2 codebook with the canonical rationale quotes withheld.

Included (coded by the lead author): codebook v2 sample B (Pass 1, 300 edges; Pass 2, 300 edges including the
stratum's first Pass 2 file), A250 (Pass 2), the v2 calibration set, the v1 500-edge expansion, and the three
retests. strongest_citance is kept as a citance number only.

    python3 analysis/experiments/72_build_label_package.py OUTDIR
Refuses to write into an existing OUTDIR.
"""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
A = ROOT / "annotations"
G = A / "generated"
V2 = G / "v2_coding_2026-10-02"
P1 = ["strongest_citance", "cited_contribution_type", "citation_function", "annotator_note"]
P2 = ["strongest_citance", "credits_contribution", "uptake_mode", "uptake_degree", "claim_dependent", "annotator_note"]


def read(p):
    with open(p, newline="", encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def ids():
    out = {}
    for c in ("hci", "nature_oa"):
        for r in read(ROOT / f"analysis/ml/splits/{c}_splits.csv"):
            rec = {"corpus": r["corpus"], "citing_s2_corpus_id": r["citing_id"], "cited_s2_corpus_id": r["cited_id"]}
            out[r["edge_id_v1"]] = out[r["edge_id_v2"]] = rec
    return out


CANON_LABELS = ["research_contribution", "contribution_subtype", "cited_contribution", "citance_function",
                "realizes_contribution", "rhetorical_polarity"]
RATIONALE = re.compile(r'(Rationale: )\*"[^"]*"\*', re.S)
EXAMPLE = re.compile(r'(Example, canonical `background_bulk` × `partial`: )\*"[^"]*"\*', re.S)
WITHHELD = "*[canonical rationale withheld from the public release]*"


def redact_codebook(text):
    """The frozen v2 codebook with the canonical record's rationale quotes withheld (labels and definitions kept)."""
    n_before = len(RATIONALE.findall(text)) + len(EXAMPLE.findall(text))
    text = RATIONALE.sub(lambda m: m.group(1) + WITHHELD, text)
    text = EXAMPLE.sub(lambda m: m.group(1) + WITHHELD, text)
    note = ("> Public copy of the codebook frozen on 3 Oct 2026 (full text sha256 "
            "2acafc2e0d9b72b3ade90527048656dab41a67361493fe45c6f2a7b943bc7639). Quotations of the canonical record's "
            f"rationales ({n_before}) are withheld; every definition, rule, label and anchor citance is unchanged.\n\n")
    return note + text


def canonical_labels():
    spec = importlib.util.spec_from_file_location("S", ROOT / "analysis/experiments/22_score_systems.py")
    S = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(S)
    rows = []
    for corpus, name in (("hci", "hci"), ("nature_oa", "nature_oa")):
        for r in read(ROOT / f"analysis/derived/{name}_contribution_coding_human_reviewed.csv"):
            rows.append({"edge_id": S.id_v1(corpus, r), "corpus": corpus, "citing_s2_corpus_id": r["citing_id"].strip(),
                         "cited_s2_corpus_id": r["cited_id"].strip(), **{c: (r.get(c) or "").strip() for c in CANON_LABELS}})
    return rows


def citance_no(v):
    m = re.match(r"\s*\[?(\d+)(?:\.0)?\]?", v or "")
    return str(int(m.group(1))) if m else ""


def coded(sheet, key, fields, idmap, extra=None):
    k = {r["item"]: r for r in read(key)}
    rows = []
    for r in read(sheet):
        e = k[r["item"]]["edge_id"]
        row = {"edge_id": e, **idmap[e], "item": r["item"]}
        for f in fields:
            v = (r.get(f) or "").strip()
            row[f] = citance_no(v) if f == "strongest_citance" else v
        if extra:
            row.update(extra)
        rows.append(row)
    return rows


def write(p, rows):
    with open(p, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    out = Path(sys.argv[1])
    if out.exists():
        sys.exit(f"refusing to write into existing {out}")
    (out / "labels").mkdir(parents=True)
    (out / "codebooks").mkdir()
    idm = ids()
    files = {
        "v2_B_pass1.csv": coded(V2 / "B_stratum_pass1.csv", V2 / "B_stratum_pass1_key.csv", P1, idm, {"subset": "stratum"})
        + coded(V2 / "B_rest_pass1.csv", V2 / "B_rest_pass1_key.csv", P1, idm, {"subset": "rest"}),
        "v2_B_pass2.csv": coded(V2 / "received/B_stratum_pass2_as_submitted_2026-10-05_2223.csv", V2 / "B_stratum_pass2_key.csv",
                                P2, idm, {"subset": "stratum"})
        + coded(V2 / "B_rest_pass2.csv", V2 / "B_rest_pass2_key.csv", P2, idm, {"subset": "rest"}),
        "v2_A250_pass2.csv": coded(V2 / "A250_pass2.csv", V2 / "A250_pass2_key.csv", P2, idm),
        "v2_retest_stratum_pass1.csv": coded(G / "retest_2026-10-31/stratum_pass1_retest_coded.csv",
                                             G / "retest_2026-10-31/stratum_pass1_retest_key.csv", P1, idm),
        "v2_retest_stratum_pass2.csv": coded(G / "retest_2026-10-31/stratum_pass2_retest_coded.csv",
                                             G / "retest_2026-10-31/stratum_pass2_retest_key.csv", P2, idm),
        "v1_retest_function.csv": coded(G / "retest_2026-10-24/function_retest_coded.csv",
                                        G / "retest_2026-10-24/function_retest_key.csv", ["citation_function", "note"], idm),
    }
    v1 = []
    for r in read(A / "coders/expansion_2026-09-26/annotator_kz_expansion_multidimensional.csv"):
        e = r["edge_id"].strip()
        v1.append({"edge_id": e, **idm[e], **{f: (r.get(f) or "").strip() for f in
                   ("cited_contribution_type", "citation_function", "contribution_realization", "centrality",
                    "rhetorical_polarity", "partial_reason", "multi_reference", "annotator_note")}})
    files["v1_expansion_500.csv"] = v1
    files["canonical_labels_2937.csv"] = canonical_labels()
    for name, rows in files.items():
        write(out / "labels" / name, rows)
    (out / "codebooks" / "CODEBOOK_v2_PUBLIC.md").write_text(
        redact_codebook((A / "CODEBOOK_v2_DRAFT.md").read_text(encoding="utf-8")), encoding="utf-8")
    for f in ("ANNOTATION_CODEBOOK_DRAFT.md",):
        shutil.copy2(A / f, out / "codebooks" / f)
    manifest = {p.relative_to(out).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted(out.rglob("*")) if p.is_file()}
    counts = {n: len(r) for n, r in files.items()}
    (out / "MANIFEST.json").write_text(json.dumps({"files": manifest, "rows": counts}, indent=2) + "\n")
    print(json.dumps(counts, indent=1))


if __name__ == "__main__":
    main()
