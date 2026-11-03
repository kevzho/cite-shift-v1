#!/usr/bin/env python3
"""Blind coding folders for two additional coders (inter-rater reliability of codebook v2), one folder per coder.

Items: the 60-edge B retest stratum plus 60 B-rest edges (30 HCI, 30 Nature OA; seed 20261103) = 120 main items, and
the 40-edge calibration set as a pilot. Evidence shown = the columns the lead author saw (both titles, numbered
citances); no labels of any kind. Each coder gets an own random order and own item codes per file, and Pass 1 and
Pass 2 use different codes, as in the lead author's protocol. Keys (item -> edge) stay in OUT/keys/, never in a
coder's folder. The codebook is the v2 text frozen on 3 Oct 2026 (hash checked).

    python3 analysis/experiments/73_make_interrater_files.py CODER [CODER ...]
Writes annotations/generated/interrater_2026-11-03/ (refuses to overwrite a coder's folder).
"""
from __future__ import annotations

import csv
import hashlib
import json
import random
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
G = ROOT / "annotations/generated"
V2, CAL = G / "v2_coding_2026-10-02", G / "v2_calibration_2026-10-02"
OUT = G / "interrater_2026-11-03"
CODEBOOK = ROOT / "annotations/CODEBOOK_v2_DRAFT.md"
CODEBOOK_SHA = "2acafc2e0d9b72b3ade90527048656dab41a67361493fe45c6f2a7b943bc7639"
SAMPLE_SEED = 20261103
SHOWN = ["citing_title", "cited_title", "n_citances", "citances"]
P1 = ["strongest_citance", "cited_contribution_type", "citation_function", "annotator_note"]
P2 = ["strongest_citance", "credits_contribution", "uptake_mode", "uptake_degree", "claim_dependent", "annotator_note"]

INSTRUCTIONS = """# Citation coding: {coder}

Thank you for helping. You will label how research papers cite each other, using a written codebook. Your labels are
compared with other coders' labels to measure how reliably the codebook can be applied. Work on your own.

## Rules (please read first)

- **Use only the files in this folder.** Do not look at anyone else's labels, and do not discuss items with the other
  coder until everything is returned.
- **No AI or LLM tools at any point**: no chatbots, no AI autocomplete, no AI features in your spreadsheet, not to read
  the citations and not to check or polish your answers or notes.
- **Do not look up the papers.** Use only the titles and citing sentences in each row.
- Work **top to bottom, in file order**. No sorting, no skipping, no going back to change an earlier row.
- Write any judgment call in `annotator_note`, briefly, in your own words. Apply it forward.
- Save as **CSV UTF-8**. Do not add, remove or reorder rows or columns.
- **Log every sitting** in `SITTING_LOG.md` when it happens (date, start, end, rows, breaks), and at the end of each
  sitting save a copy named with the sitting number, e.g. `main_pass1_sitting1.csv`. Return these copies too.

## Order and timing

| Step | File | Rows | When |
|---|---|---|---|
| 1 | `calibration_pass1.csv` | 40 | first |
| 2 | `calibration_pass2.csv` | 40 | at least 24 h after you finish step 1 |
| - | Return steps 1-2; short call about codebook questions (no main-set items discussed) | | |
| 3 | `main_pass1.csv` | 120 | after the call |
| 4 | `main_pass2.csv` | 120 | at least 24 h after you finish step 3 |

Expect roughly 1-1.5 minutes per row per pass (about 6-8 hours in total), in several sittings.

## Codebook

`CODEBOOK_v2.md` (frozen 3 Oct 2026). Read sections 1-5 and the anchors in section 8 before starting. Each row is one
citation link: a citing paper, a cited paper, and every sentence in the citing paper that cites it ("citances",
numbered [1], [2], ...).

## Pass 1 files

1. `strongest_citance`: the number of the strongest (most specific) citance. If one sentence cites several papers,
   judge only what it says about this cited paper.
2. `cited_contribution_type`: `empirical` · `artifact` · `methodological` · `theoretical` · `dataset` · `survey` ·
   `opinion` · `unclear`.
3. `citation_function`: `background` · `uses_method` · `applies_framework` · `builds_on_finding` · `reuses_artifact` ·
   `compares_against` · `other` (say what in the note) · `unclear`. Executed *and* thought with -> `uses_method`.

## Pass 2 files

1. `strongest_citance`: choose again, without looking at Pass 1.
2. `credits_contribution`: `specific` · `generic` · `unclear`. Specific is not use.
3. `uptake_mode` (one primary; any secondary in the note): `method` · `artifact` · `finding` · `extension` ·
   `framing` · `design_knowledge` · `contrast` · `none` · `unclear`. On background-looking citances, look deliberately
   for `framing` and `design_knowledge`.
4. `uptake_degree`: only when the mode is not `none` or `unclear`; otherwise leave it **blank**. `full`, or `partial`
   only if you can **name the part** in `annotator_note`.
5. `claim_dependent`: delete the cited paper; would the citing paper's own claim, method or artifact have to change?
   `yes` · `no` · `unclear`.

## Returning your work

Send back the whole folder: the four coded files, your sitting copies and `SITTING_LOG.md`. Keep a copy yourself.
"""

LOG = """# Sitting log: {coder}

| Date | Start | End | File | Rows | Breaks / notes |
|---|---|---|---|---|---|
| | | | | | |
"""


def read(p):
    with open(p, newline="", encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def write(p, rows):
    with open(p, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def evidence(sheet, key):
    k = {r["item"]: r for r in read(key)}
    return {k[r["item"]]["edge_id"]: {"corpus": k[r["item"]].get("corpus", ""), **{c: r[c] for c in SHOWN}}
            for r in read(sheet)}


def main():
    coders = sys.argv[1:]
    if not coders:
        sys.exit(__doc__)
    if hashlib.sha256(CODEBOOK.read_bytes()).hexdigest() != CODEBOOK_SHA:
        sys.exit("codebook does not match the 3 Oct 2026 freeze")
    stratum = evidence(V2 / "B_stratum_pass1.csv", V2 / "B_stratum_pass1_key.csv")
    rest = evidence(V2 / "B_rest_pass1.csv", V2 / "B_rest_pass1_key.csv")
    rng = random.Random(SAMPLE_SEED)
    extra = []
    for corpus in ("hci", "nature_oa"):
        extra += rng.sample(sorted(e for e, r in rest.items() if r["corpus"] == corpus), 30)
    main_ev = {**stratum, **{e: rest[e] for e in extra}}
    cal_ev = evidence(CAL / "calibration_pass1.csv", CAL / "calibration_pass1_key.csv")
    assert len(main_ev) == 120 and len(cal_ev) == 40
    (OUT / "keys").mkdir(parents=True, exist_ok=True)
    manifest = {"script": "analysis/experiments/73_make_interrater_files.py", "sample_seed": SAMPLE_SEED,
                "main_edges": sorted(main_ev), "stratum_edges": sorted(stratum), "rest_edges": sorted(extra),
                "codebook_sha256": CODEBOOK_SHA, "coders": {}}
    for ci, coder in enumerate(coders):
        folder = OUT / coder
        if folder.exists():
            sys.exit(f"refusing to overwrite {folder}")
        folder.mkdir()
        seed = SAMPLE_SEED + 1 + ci
        files = {}
        for fname, ev, labels, prefix in (("calibration_pass1", cal_ev, P1, "C1"), ("calibration_pass2", cal_ev, P2, "C2"),
                                          ("main_pass1", main_ev, P1, "M1"), ("main_pass2", main_ev, P2, "M2")):
            edges = sorted(ev)
            random.Random(f"{seed}-{fname}").shuffle(edges)
            sheet, key = [], []
            for i, e in enumerate(edges, 1):
                item = f"{prefix}-{i:03d}"
                sheet.append({"item": item, **{c: ev[e][c] for c in SHOWN}, **{c: "" for c in labels}})
                key.append({"item": item, "edge_id": e})
            write(folder / f"{fname}.csv", sheet)
            write(OUT / "keys" / f"{coder}_{fname}_key.csv", key)
            files[fname] = hashlib.sha256((folder / f"{fname}.csv").read_bytes()).hexdigest()
        shutil.copy2(CODEBOOK, folder / "CODEBOOK_v2.md")
        (folder / "INSTRUCTIONS.md").write_text(INSTRUCTIONS.format(coder=coder.capitalize()), encoding="utf-8")
        (folder / "SITTING_LOG.md").write_text(LOG.format(coder=coder.capitalize()), encoding="utf-8")
        manifest["coders"][coder] = {"order_seed": seed, "sheet_sha256": files}
        shutil.make_archive(str(OUT / f"{coder}_coding_folder"), "zip", OUT, coder)
        print(f"{coder}: {folder.relative_to(ROOT)} + {coder}_coding_folder.zip")
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
