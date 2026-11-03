#!/usr/bin/env python3
"""Coding files for codebook v2 (docs/sts/V2_CODING_PROCEDURE.md, Parts A4 and B).

Subcommands
-----------
  calibration   40 calibration items (20 HCI + 20 Nature OA), Pass 1 and Pass 2
                files, into annotations/generated/v2_calibration_2026-10-02/.
  sample        Sample B (150 + 150), its 60-edge retest stratum, and the A file
                (v2 Pass 2 on the 500 v1 edges, function hidden), into
                annotations/generated/v2_coding_2026-10-02/.
  check FILE    Validate a coded Pass 1 or Pass 2 file against codebook v2.
  cells         Codebook §7 critical cells on the coded calibration files.

Pool. Calibration and B are drawn from edges in the `frozen` split of
analysis/ml/splits/ (component-mates of earlier samples) that have never been
in any sample, anchor, rationale check or external test set. They are already
held out, so drawing them costs no training edges and leaves the splits
unchanged.

Blinding. Coding files carry only an item code, both titles and the numbered
citances. Item codes differ between passes, so a Pass 2 row cannot be matched
to its Pass 1 labels while coding. The item -> edge_id map is in *_key.csv;
the canonical function used to stratify the retest is in the key only.

Never overwrites an existing output file: coded work is not regenerated over.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
GEN = ROOT / "annotations/generated"
CAL_DIR = GEN / "v2_calibration_2026-10-02"
COD_DIR = GEN / "v2_coding_2026-10-02"
DERIVED = {c: ROOT / f"analysis/derived/{c}_contribution_coding_human_reviewed.csv"
           for c in ("hci", "nature_oa")}
SPLITS = {c: ROOT / f"analysis/ml/splits/{c}_splits.csv" for c in DERIVED}
A_SOURCE = GEN / "expansion_2026-09-16/expansion_batch_kz.csv"
SEEN = GEN / "v2_prep/v2_seen_edges.txt"
ANCHORS = GEN / "codebook_v2_anchors.json"
TEST_EDGES = ROOT / "analysis/results/external_systems/test_edges.csv"

SEED_CAL = 20261002
SEED_B = 20261003
N_CAL = 20          # per corpus
N_B = 150           # per corpus
N_STRATUM = 30      # per corpus

PASS1_FIELDS = ["strongest_citance", "cited_contribution_type", "citation_function",
                "annotator_note"]
PASS2_FIELDS = ["strongest_citance", "credits_contribution", "uptake_mode",
                "uptake_degree", "claim_dependent", "annotator_note"]

LABELS = {
    "cited_contribution_type": {"empirical", "artifact", "methodological", "theoretical",
                                "dataset", "survey", "opinion", "unclear"},
    "citation_function": {"background", "uses_method", "applies_framework",
                          "builds_on_finding", "reuses_artifact", "compares_against",
                          "other", "unclear"},
    "credits_contribution": {"specific", "generic", "unclear"},
    "uptake_mode": {"method", "artifact", "finding", "extension", "framing",
                    "design_knowledge", "contrast", "none", "unclear"},
    "uptake_degree": {"full", "partial"},
    "claim_dependent": {"yes", "no", "unclear"},
}


# --------------------------------------------------------------------------
# io

def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_new(path: Path, rows: list[dict]) -> None:
    if path.exists():
        raise SystemExit(f"refusing to overwrite {path.relative_to(ROOT)}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        w = csv.DictWriter(handle, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def edge_id(corpus: str, row: dict) -> str:
    """Round-one scheme, as 01_make_annotation_samples.py and expansion_batch_kz."""
    key = "\x1f".join((corpus, (row.get("cited_id") or "").strip(),
                       (row.get("citing_id") or "").strip()))
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:20]


# --------------------------------------------------------------------------
# pool

def canonical() -> dict[str, tuple[str, dict]]:
    return {edge_id(c, r): (c, r) for c, p in DERIVED.items() for r in read_csv(p)}


def excluded() -> set[str]:
    """Every edge already in a sample, seen, an anchor or an external test edge."""
    out: set[str] = set()
    for base in (ROOT / "annotations", ROOT / "private/history"):
        for path in base.rglob("*.csv"):
            for row in read_csv(path):
                if row.get("edge_id"):
                    out.add(row["edge_id"].strip())
    out |= set(SEEN.read_text(encoding="utf-8").split())
    for group in json.loads(ANCHORS.read_text(encoding="utf-8")).values():
        out |= {a["edge_id"] for a in group}
    out |= {r["edge_id"] for r in read_csv(TEST_EDGES)}
    return out


def pool() -> dict[str, list[str]]:
    """corpus -> sorted v1 edge ids: frozen split, never sampled."""
    ex = excluded()
    out = {}
    for c, path in SPLITS.items():
        out[c] = sorted(s["edge_id_v1"] for s in read_csv(path)
                        if s["split"] == "frozen"
                        and s["edge_id_v1"] not in ex and s["edge_id_v2"] not in ex)
    return out


# --------------------------------------------------------------------------
# files

def numbered(citances: str) -> tuple[int, str]:
    parts = [p.strip() for p in citances.split("|||") if p.strip()]
    return len(parts), "\n".join(f"[{i}] {p}" for i, p in enumerate(parts, 1))


def coding_rows(ids: list[str], canon: dict, prefix: str, fields: list[str],
                rng: random.Random) -> tuple[list[dict], list[dict]]:
    order = list(ids)
    rng.shuffle(order)
    sheet, key = [], []
    for i, eid in enumerate(order, 1):
        c, r = canon[eid]
        n, text = numbered(r["citances"])
        item = f"{prefix}{i:03d}"
        sheet.append({"item": item, "citing_title": r["citing_title"].strip(),
                      "cited_title": r["cited_title"].strip(), "n_citances": n,
                      "citances": text, **{f: "" for f in fields}})
        key.append({"item": item, "edge_id": eid, "corpus": c,
                    "canonical_function": r["citance_function"].strip()})
    return sheet, key


def stratified(ids: list[str], canon: dict, n: int, rng: random.Random) -> list[str]:
    """n of ids, allocated across canonical function by largest remainder."""
    by_fn: dict[str, list[str]] = {}
    for eid in ids:
        by_fn.setdefault(canon[eid][1]["citance_function"].strip(), []).append(eid)
    quota = {f: n * len(v) / len(ids) for f, v in by_fn.items()}
    take = {f: int(q) for f, q in quota.items()}
    for f in sorted(quota, key=lambda f: (-(quota[f] - take[f]), f))[: n - sum(take.values())]:
        take[f] += 1
    out = []
    for f in sorted(by_fn):
        out += rng.sample(sorted(by_fn[f]), take[f])
    return sorted(out)


def manifest(path: Path, payload: dict) -> None:
    if path.exists():
        raise SystemExit(f"refusing to overwrite {path.relative_to(ROOT)}")
    payload["outputs"] = {p.name: sha256(p) for p in sorted(path.parent.glob("*.csv"))}
    payload["sources"] = {str(p.relative_to(ROOT)): sha256(p)
                          for p in [*DERIVED.values(), *SPLITS.values(), A_SOURCE,
                                    SEEN, ANCHORS, TEST_EDGES]}
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def cmd_calibration() -> int:
    canon, pl = canonical(), pool()
    rng = random.Random(SEED_CAL)
    picked = sorted(e for c in sorted(pl) for e in rng.sample(pl[c], N_CAL))
    for prefix, fields, name in (("C1-", PASS1_FIELDS, "pass1"),
                                 ("C2-", PASS2_FIELDS, "pass2")):
        sheet, key = coding_rows(picked, canon, prefix, fields, rng)
        write_new(CAL_DIR / f"calibration_{name}.csv", sheet)
        write_new(CAL_DIR / f"calibration_{name}_key.csv", key)
    manifest(CAL_DIR / "manifest.json", {
        "script": "analysis/experiments/21_make_v2_coding_files.py calibration",
        "seed": SEED_CAL, "per_corpus": N_CAL,
        "pool": {c: len(v) for c, v in pl.items()},
        "rule": "frozen-split component-mates never in any sample, anchor, rationale "
                "check or external test set; calibration labels never become data"})
    print(f"calibration: {len(picked)} items -> {CAL_DIR.relative_to(ROOT)}")
    return 0


def cmd_sample() -> int:
    canon, pl = canonical(), pool()          # pool() now excludes calibration
    rng = random.Random(SEED_B)
    b = {c: sorted(rng.sample(pl[c], N_B)) for c in sorted(pl)}
    stratum = sorted(e for c in sorted(b) for e in stratified(b[c], canon, N_STRATUM, rng))
    rest = sorted(set(b["hci"] + b["nature_oa"]) - set(stratum))
    files = [("B_stratum", stratum, "S"), ("B_rest", rest, "R")]
    for stem, ids, tag in files:
        for prefix, fields, name in ((f"{tag}1-", PASS1_FIELDS, "pass1"),
                                     (f"{tag}2-", PASS2_FIELDS, "pass2")):
            sheet, key = coding_rows(ids, canon, prefix, fields, rng)
            write_new(COD_DIR / f"{stem}_{name}.csv", sheet)
            write_new(COD_DIR / f"{stem}_{name}_key.csv", key)

    a_ids = sorted(r["edge_id"].strip() for r in read_csv(A_SOURCE))
    missing = [e for e in a_ids if e not in canon]
    if missing:
        raise SystemExit(f"{len(missing)} A edges not found in canonical")
    sheet, key = coding_rows(a_ids, canon, "A2-", PASS2_FIELDS, rng)
    write_new(COD_DIR / "A_pass2.csv", sheet)
    write_new(COD_DIR / "A_pass2_key.csv", key)

    strat_fn = Counter((canon[e][0], canon[e][1]["citance_function"].strip())
                       for e in stratum)
    manifest(COD_DIR / "manifest.json", {
        "script": "analysis/experiments/21_make_v2_coding_files.py sample",
        "seed": SEED_B,
        "pool_after_calibration": {c: len(v) for c, v in pl.items()},
        "B": {c: len(v) for c, v in b.items()},
        "stratum": {"n": len(stratum), "per_corpus": N_STRATUM,
                    "stratified_on": "corpus x canonical citance_function "
                                     "(Pass 1 function does not exist yet; disclosed)",
                    "cells": {f"{c}/{f}": n for (c, f), n in sorted(strat_fn.items())}},
        "B_rest": len(rest), "A": len(a_ids),
        "A_note": "v2 Pass 2 on the 500 v1 edges; no function column; new order"})
    print(f"B: {len(stratum)} stratum + {len(rest)} rest; A: {len(a_ids)} "
          f"-> {COD_DIR.relative_to(ROOT)}")
    return 0


# --------------------------------------------------------------------------
# checks

def problems(rows: list[dict]) -> list[str]:
    out = []
    pass2 = "uptake_mode" in rows[0]
    for r in rows:
        item, note = r["item"], (r.get("annotator_note") or "").strip()
        sc = (r.get("strongest_citance") or "").strip()
        if not (sc.isdigit() and 1 <= int(sc) <= int(r["n_citances"])):
            out.append(f"{item}: strongest_citance '{sc}' not in 1..{r['n_citances']}")
        fields = (["credits_contribution", "uptake_mode", "claim_dependent"] if pass2
                  else ["cited_contribution_type", "citation_function"])
        for f in fields:
            v = (r.get(f) or "").strip()
            if v not in LABELS[f]:
                out.append(f"{item}: {f} '{v}' not allowed")
        if pass2:
            mode = (r.get("uptake_mode") or "").strip()
            deg = (r.get("uptake_degree") or "").strip()
            if mode in {"none", "unclear"} and deg:
                out.append(f"{item}: uptake_degree must be blank when mode is {mode}")
            if mode in LABELS["uptake_mode"] - {"none", "unclear"} and deg not in LABELS["uptake_degree"]:
                out.append(f"{item}: uptake_degree '{deg}' must be full or partial")
            if deg == "partial" and not note:
                out.append(f"{item}: partial must name the part in annotator_note")
        elif (r.get("citation_function") or "").strip() == "other" and not note:
            out.append(f"{item}: function 'other' needs a note")
    return out


def cmd_check(path: Path) -> int:
    rows = read_csv(path)
    bad = problems(rows)
    done = sum(1 for r in rows if (r.get("strongest_citance") or "").strip())
    print(f"{path.name}: {len(rows)} rows, {done} started, {len(bad)} problem(s)")
    for line in bad[:60]:
        print("  " + line)
    if not bad:
        print(f"  sha256 {sha256(path)}")
    return 1 if bad else 0


def cmd_cells() -> int:
    def joined(name: str) -> dict[str, dict]:
        key = {k["item"]: k["edge_id"] for k in read_csv(CAL_DIR / f"calibration_{name}_key.csv")}
        return {key[r["item"]]: r for r in read_csv(CAL_DIR / f"calibration_{name}.csv")}
    p1, p2 = joined("pass1"), joined("pass2")
    bad = problems(list(p1.values())) + problems(list(p2.values()))
    if bad:
        print(f"{len(bad)} validation problem(s); run `check` on each file first")
        return 1
    bg = [e for e, r in p1.items() if r["citation_function"].strip() == "background"]
    cells = {
        "background x uptake_mode in {finding, framing, design_knowledge}":
            sum(p2[e]["uptake_mode"].strip() in {"finding", "framing", "design_knowledge"} for e in bg),
        "background x specific x none":
            sum(p2[e]["credits_contribution"].strip() == "specific"
                and p2[e]["uptake_mode"].strip() == "none" for e in bg),
        "claim_dependent=no x uptake_mode not in {none, unclear}":
            sum(r["claim_dependent"].strip() == "no"
                and r["uptake_mode"].strip() not in {"none", "unclear"} for r in p2.values()),
    }
    for name, n in cells.items():
        print(f"  {'ok  ' if n else 'ZERO'} {n:>3}  {name}")
    print("gate 4:", "MET" if all(cells.values()) else "NOT MET - codebook defect (§7)")
    return 0 if all(cells.values()) else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("calibration")
    sub.add_parser("sample")
    ck = sub.add_parser("check")
    ck.add_argument("file", type=Path)
    sub.add_parser("cells")
    args = ap.parse_args()
    if args.cmd == "calibration":
        return cmd_calibration()
    if args.cmd == "sample":
        return cmd_sample()
    if args.cmd == "check":
        return cmd_check(args.file)
    return cmd_cells()


if __name__ == "__main__":
    sys.exit(main())
