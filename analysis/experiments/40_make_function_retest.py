#!/usr/bin/env python3
"""Registered function-only retest (pre-registration §3, §7 Stability): 150 of the lead author's v1 500,
seed 20261024, stratified by the v1 citation_function label (largest remainder), on/after 24 Oct 2026
(>= 28 days after the v1 file was saved, 26 Sep 2026 11:31). Blind: the sheet carries no labels; the key carries
only item -> edge_id. Evidence shown = the v1 file's citation_context, the same text seen at first coding.
Writes annotations/generated/retest_2026-10-24/ (refuses to overwrite).
"""
import csv
import hashlib
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
V1 = ROOT / "annotations/coders/expansion_2026-09-26/annotator_kz_expansion_multidimensional.csv"
OUT = ROOT / "annotations/generated/retest_2026-10-24"
SEED, N = 20261024, 150


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    sheet_p, key_p = OUT / "function_retest.csv", OUT / "function_retest_key.csv"
    for p in (sheet_p, key_p):
        if p.exists():
            raise SystemExit(f"refusing to overwrite {p}")
    rows = list(csv.DictReader(open(V1, encoding="utf-8-sig")))
    rng = random.Random(SEED)
    by = {}
    for r in rows:
        by.setdefault(r["citation_function"].strip(), []).append(r["edge_id"].strip())
    quota = {f: N * len(v) / len(rows) for f, v in by.items()}
    take = {f: int(q) for f, q in quota.items()}
    for f in sorted(quota, key=lambda f: (-(quota[f] - take[f]), f))[: N - sum(take.values())]:
        take[f] += 1
    chosen = []
    for f in sorted(by):
        chosen += rng.sample(sorted(by[f]), take[f])
    rng.shuffle(chosen)
    rec = {r["edge_id"].strip(): r for r in rows}
    sheet, key = [], []
    for i, e in enumerate(chosen, 1):
        r = rec[e]
        parts = [p.strip() for p in r["citation_context"].split("|||") if p.strip()]
        item = f"F-{i:03d}"
        sheet.append({"item": item, "citing_title": r["citing_title"].strip(), "cited_title": r["cited_title"].strip(),
                      "n_citances": len(parts), "citances": "\n".join(f"[{j}] {p}" for j, p in enumerate(parts, 1)),
                      "citation_function": "", "note": ""})
        key.append({"item": item, "edge_id": e})
    for p, data in ((sheet_p, sheet), (key_p, key)):
        with open(p, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(data[0]))
            w.writeheader()
            w.writerows(data)
    man = {"script": "analysis/experiments/39_make_function_retest.py", "seed": SEED, "n": len(chosen),
           "stratified_on": "v1 citation_function (largest remainder)", "source_v1_file_saved": "2026-09-26 11:31",
           "sheet_sha256": hashlib.sha256(sheet_p.read_bytes()).hexdigest(),
           "key_sha256": hashlib.sha256(key_p.read_bytes()).hexdigest()}
    (OUT / "manifest.json").write_text(json.dumps(man, indent=2) + "\n")
    print(f"wrote {len(chosen)} items -> {sheet_p.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
