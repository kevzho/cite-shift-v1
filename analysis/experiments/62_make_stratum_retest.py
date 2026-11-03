#!/usr/bin/env python3
"""Registered v2 stability retest of the 60-edge stratum (pre-registration §3 and §7 Stability; coding procedure
Part D3): Pass 1 and Pass 2, blind, new order, new item codes, no prior labels, each item >= 28 days after its first
coding. First coding: Pass 1 on 3 Oct 2026; Pass 2 = the third stratum submission, created 5 Oct 2026 22:23.
So Pass 1 may start on 31 Oct; Pass 2 on 2 Nov at the earliest and >= 24 h after the Pass 1 retest is finished.

Evidence shown = the same columns the coder saw at first coding (titles, numbered citances). The keys map
item -> edge_id and the first-coding item code; they are never opened while coding.

    python3 analysis/experiments/62_make_stratum_retest.py
Writes annotations/generated/retest_2026-10-31/ (refuses to overwrite).
"""
import csv
import hashlib
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
V2 = ROOT / "annotations/generated/v2_coding_2026-10-02"
OUT = ROOT / "annotations/generated/retest_2026-10-31"
PASSES = {
    1: dict(src="B_stratum_pass1", seed=20261031, prefix="R1",
            labels=["strongest_citance", "cited_contribution_type", "citation_function", "annotator_note"],
            earliest="2026-10-31 (first coded 2026-10-03)"),
    2: dict(src="B_stratum_pass2", seed=20261102, prefix="R2",
            labels=["strongest_citance", "credits_contribution", "uptake_mode", "uptake_degree", "claim_dependent",
                    "annotator_note"],
            earliest="2026-11-02 (first coded 2026-10-05 22:23) and >= 24 h after the Pass 1 retest is finished"),
}
SHOWN = ["citing_title", "cited_title", "n_citances", "citances"]


def read(p):
    with open(p, newline="", encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def write(p, rows):
    with open(p, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    targets = [OUT / f"stratum_pass{n}_retest{s}.csv" for n in PASSES for s in ("", "_key")]
    clash = [p for p in targets if p.exists()]
    if clash:
        raise SystemExit(f"refusing to overwrite {clash}")
    manifest = {"script": "analysis/experiments/62_make_stratum_retest.py", "passes": {}}
    edge_sets = []
    for n, cfg in PASSES.items():
        key = {r["item"]: r for r in read(V2 / f"{cfg['src']}_key.csv")}
        text = {r["item"]: r for r in read(V2 / f"{cfg['src']}.csv")}  # the blank template: evidence columns only
        items = sorted(key)
        random.Random(cfg["seed"]).shuffle(items)
        sheet, out_key = [], []
        for i, old in enumerate(items, 1):
            new = f"{cfg['prefix']}-{i:03d}"
            sheet.append({"item": new, **{c: text[old][c] for c in SHOWN}, **{c: "" for c in cfg["labels"]}})
            out_key.append({"item": new, "edge_id": key[old]["edge_id"], "first_item": old})
        edge_sets.append({r["edge_id"] for r in out_key})
        sp, kp = OUT / f"stratum_pass{n}_retest.csv", OUT / f"stratum_pass{n}_retest_key.csv"
        write(sp, sheet)
        write(kp, out_key)
        manifest["passes"][f"pass{n}"] = {"source": f"{cfg['src']}.csv", "seed": cfg["seed"], "n": len(sheet),
                                          "earliest_start": cfg["earliest"],
                                          "sheet_sha256": hashlib.sha256(sp.read_bytes()).hexdigest(),
                                          "key_sha256": hashlib.sha256(kp.read_bytes()).hexdigest()}
        print(f"wrote {len(sheet)} items -> {sp.relative_to(ROOT)}")
    assert edge_sets[0] == edge_sets[1], "Pass 1 and Pass 2 stratum keys cover different edges"
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
