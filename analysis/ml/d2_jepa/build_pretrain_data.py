#!/usr/bin/env python3
"""D2 (exploratory) pretraining pairs: context = citances of a CHI edge, target = cited paper title.

Spec: private/preregistration/JEPA_EXPLORATORY_SPEC_2026-10-07.md.
Exclusion = the D1 rule (docs/results/2026-10-04_d1_silver_chi2000.md): drop every CHI edge whose citing
paper's S2 CorpusId appears as the citing OR cited id of any canonical edge (HCI or Nature OA). That keeps
every v1, v2 (B, A250), calibration, anchor and test-split paper out of pretraining. Also dropped: edges
with an empty cited title or no non-empty citance. The 2,000 D1 CHI edges are training-side and are kept.

Output (gitignored, contains citance text): analysis/results/ml/d2_jepa/data/{train,val}.jsonl, 99/1 split
by citing paper (so no paper's citances straddle train/val), seed 20261007; manifest.json with counts
and sha256.
"""
from __future__ import annotations

import csv
import hashlib
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "external_data/chi_chiea_grobid_2026-09-15/citances/chi_chiea_citances.jsonl"
OUT = ROOT / "analysis/results/ml/d2_jepa/data"
SEED = 20261007


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    for name in ("train.jsonl", "val.jsonl"):
        if (OUT / name).exists():
            raise SystemExit(f"refusing to overwrite {OUT / name}")
    canon_ids = set()
    for c in ("hci", "nature_oa"):
        with open(ROOT / f"analysis/derived/{c}_contribution_coding_human_reviewed.csv", encoding="utf-8-sig") as h:
            for r in csv.DictReader(h):
                canon_ids.add(r["cited_id"].strip())
                canon_ids.add(r["citing_id"].strip())

    counts = {"edges": 0, "dropped_canonical_paper": 0, "dropped_empty_title": 0, "dropped_no_citance": 0}
    by_paper: dict[str, list[dict]] = {}
    for line in open(SRC, encoding="utf-8"):
        d = json.loads(line)
        counts["edges"] += 1
        pid = str(d["citing_paper_id"])
        if pid in canon_ids:
            counts["dropped_canonical_paper"] += 1
            continue
        title = (d.get("cited_title") or "").strip()
        if not title:
            counts["dropped_empty_title"] += 1
            continue
        cits = [c.strip() for c in d["citances"] if c and c.strip()]
        if not cits:
            counts["dropped_no_citance"] += 1
            continue
        by_paper.setdefault(pid, []).append({"citing_paper_id": pid, "cited_ref_id": d["cited_ref_id"],
                                             "venue": d.get("venue", ""), "context": " ".join(cits),
                                             "target": title})
    papers = sorted(by_paper)
    random.Random(SEED).shuffle(papers)
    n_val = max(1, round(0.01 * len(papers)))
    val_p = set(papers[:n_val])
    n = {"train": 0, "val": 0}
    with open(OUT / "train.jsonl", "w", encoding="utf-8") as tr, open(OUT / "val.jsonl", "w", encoding="utf-8") as va:
        for pid in sorted(by_paper):
            f, k = (va, "val") if pid in val_p else (tr, "train")
            for r in by_paper[pid]:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
                n[k] += 1
    manifest = {"source": str(SRC.relative_to(ROOT)), "source_sha256": sha256(SRC), "seed": SEED,
                "rule": "D1 exclusion (citing paper S2 id in any canonical citing/cited id); empty title / no citance dropped",
                **counts, "kept_pairs": n["train"] + n["val"], "train_pairs": n["train"], "val_pairs": n["val"],
                "citing_papers": len(papers), "val_papers": n_val,
                "sha256": {p.name: sha256(p) for p in (OUT / "train.jsonl", OUT / "val.jsonl")}}
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
