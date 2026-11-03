#!/usr/bin/env python3
"""
Extract a deterministic seed set of S2AG corpusIds from the multi-venue
inventory DuckDB, optionally merging extra seed files (e.g. the CHI'22
GROBID-derived seeds, which include outgoing-reference corpusids).

Metadata-first note: inventory seeds are the venue papers themselves only.
Incoming citations (with contexts/intents) are available immediately from the
S2AG citations dump because build_snapshot_db.py keeps edges where EITHER
endpoint is a seed. Outgoing-reference seeds get added later by the GROBID +
S2-augmentation stage; re-run this script and the snapshot build then.

Example:
  uv run python extract_inventory_seed_ids.py \
    --inventory data/inventory/hci_inventory_2026-01-27.duckdb \
    --extra-seed-file data/corpus_seeds/chi22_seed_corpus_ids.txt \
    --tiers a,b,c,d \
    --out data/corpus_seeds/hci_v1_seed_corpus_ids.txt
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import duckdb


def _load_seed_file(path: Path) -> set[int]:
    seed: set[int] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            seed.add(int(line))
        except Exception:
            continue
    return seed


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inventory", type=Path, default=Path("data/inventory/hci_inventory_2026-01-27.duckdb"))
    ap.add_argument("--tiers", type=str, default="a,b,c,d")
    ap.add_argument("--extra-seed-file", type=Path, action="append", default=[],
                    help="Additional seed file(s) to merge in (repeatable).")
    ap.add_argument("--out", type=Path, default=Path("data/corpus_seeds/hci_v1_seed_corpus_ids.txt"))
    ap.add_argument("--manifest", type=Path, default=None,
                    help="Default: <out>.manifest.json next to --out.")
    args = ap.parse_args()

    tiers = [t.strip() for t in args.tiers.split(",") if t.strip()]
    con = duckdb.connect(str(args.inventory), read_only=True)
    tier_list = ", ".join("'" + t.replace("'", "''") + "'" for t in tiers)
    rows = con.execute(f"SELECT corpusid FROM inventory WHERE tier IN ({tier_list});").fetchall()
    inventory_ids = {int(r[0]) for r in rows}

    per_venue = con.execute(
        f"SELECT venue_key, count(*) FROM inventory WHERE tier IN ({tier_list}) GROUP BY 1 ORDER BY 2 DESC;"
    ).fetchall()

    seed = set(inventory_ids)
    extra_stats = []
    for p in args.extra_seed_file:
        ids = _load_seed_file(p)
        new = len(ids - seed)
        seed |= ids
        extra_stats.append({"file": str(p), "ids": len(ids), "new_ids_added": new})

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(str(x) for x in sorted(seed)) + "\n", encoding="utf-8")

    manifest_path = args.manifest or (args.out.parent / (args.out.stem + ".manifest.json"))
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "inventory": str(args.inventory),
        "tiers": tiers,
        "out": str(args.out),
        "n_seed_ids": len(seed),
        "n_inventory_ids": len(inventory_ids),
        "inventory_per_venue": {k: int(n) for k, n in per_venue},
        "extra_seed_files": extra_stats,
        "note": "Inventory seeds are venue papers only (metadata-first); outgoing-reference seeds added later via GROBID+S2 augmentation.",
    }
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(json.dumps({"out": str(args.out), "n_seed_ids": len(seed),
                      "n_inventory_ids": len(inventory_ids),
                      "extra_seed_files": extra_stats}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
