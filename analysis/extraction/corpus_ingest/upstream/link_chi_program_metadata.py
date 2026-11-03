#!/usr/bin/env python3
"""
Join SIGCHI program metadata (rows_chi22..25 JSONs) onto the inventory DuckDB.

- Match by normalized title within the program year (fallback: year +/- 1),
  against inventory rows with venue_key in ('chi', 'chiea').
- On match: set has_program_metadata=TRUE and program_type/award/url.
- Tier-a flag: every inventory 'chi' row with year in the registry's
  program_years gets tier='a' (CHI 2022-2025), matched or not.
- Writes an unmatched report (per year/type match rates + unmatched titles).
- Cross-check: CHI'22 matched corpusids vs data/chi22_grobid_s2 target ids.

Example:
  uv run python link_chi_program_metadata.py \
    --db data/inventory/hci_inventory_2026-01-27.duckdb \
    --report data/inventory/chi_program_link_report.json
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path

import duckdb

DEFAULT_PROGRAM_FILES = {
    2022: "data/chi22_25_with_urls_fixed/rows_chi22_with_urls.json",
    2023: "data/chi22_25_with_urls_fixed/rows_chi23_with_urls.json",
    2024: "data/chi22_25_with_urls_fixed/rows_chi24_with_urls.json",
    2025: "data/chi22_25_with_urls_fixed/rows_chi25_with_urls_fixed.json",
}


def norm_title(t: str) -> str:
    t = unicodedata.normalize("NFKD", t or "")
    t = "".join(c for c in t if not unicodedata.combining(c))
    t = t.lower()
    t = re.sub(r"[^a-z0-9]+", " ", t)
    return t.strip()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", type=Path, default=Path("data/inventory/hci_inventory_2026-01-27.duckdb"))
    ap.add_argument("--report", type=Path, default=Path("data/inventory/chi_program_link_report.json"))
    ap.add_argument("--grobid-s2-dir", type=Path, default=Path("data/chi22_grobid_s2"),
                    help="Cross-check CHI'22 matches against existing target_corpusId values.")
    ap.add_argument("--program-years", type=str, default="2022-2025")
    args = ap.parse_args()

    y_lo, y_hi = (int(x) for x in args.program_years.split("-"))
    con = duckdb.connect(str(args.db))

    # Inventory lookup: normalized title -> [(corpusid, year, venue_key)]
    inv_rows = con.execute(
        f"""
        SELECT corpusid, title, year, venue_key FROM inventory
        WHERE venue_key IN ('chi', 'chiea') AND year BETWEEN {y_lo - 1} AND {y_hi + 1};
        """
    ).fetchall()
    by_title: dict[str, list[tuple[int, int, str]]] = {}
    by_prefix: dict[str, list[tuple[int, int, str]]] = {}
    for cid, title, year, vkey in inv_rows:
        nt = norm_title(title)
        by_title.setdefault(nt, []).append((int(cid), int(year), vkey))
        by_prefix.setdefault(" ".join(nt.split()[:8]), []).append((int(cid), int(year), vkey))

    report: dict = {"per_year": {}, "unmatched": []}
    updates: list[tuple[str, str, str, int]] = []  # (type, award, url, corpusid)
    matched_cids_2022: set[int] = set()

    for year, fname in DEFAULT_PROGRAM_FILES.items():
        if not (y_lo <= year <= y_hi):
            continue
        rows = json.loads(Path(fname).read_text(encoding="utf-8"))
        per_type: dict[str, dict[str, int]] = {}
        for r in rows:
            ptype = r.get("type") or ""
            stats = per_type.setdefault(ptype, {"rows": 0, "matched": 0})
            stats["rows"] += 1
            nt = norm_title(r.get("title", ""))
            cands = by_title.get(nt, [])
            if not cands:
                # Fuzzy fallback: unique match on the first 8 normalized words
                # (handles subtitle truncation / minor title edits).
                prefix_cands = by_prefix.get(" ".join(nt.split()[:8]), [])
                if len(prefix_cands) == 1:
                    cands = prefix_cands
            # Prefer exact year, then +/- 1; prefer chi over chiea on ties.
            best = None
            for cid, y, vkey in cands:
                score = (abs(y - year), 0 if vkey == "chi" else 1)
                if abs(y - year) <= 1 and (best is None or score < best[0]):
                    best = (score, cid)
            if best is None:
                report["unmatched"].append({"year": year, "type": ptype, "title": r.get("title")})
                continue
            stats["matched"] += 1
            updates.append((ptype, r.get("award") or None, r.get("url") or None, best[1]))
            if year == 2022:
                matched_cids_2022.add(best[1])
        report["per_year"][year] = {
            "file": fname,
            "rows": len(rows),
            "matched": sum(s["matched"] for s in per_type.values()),
            "per_type": per_type,
        }

    # Idempotent: reset program columns before applying this run's matches.
    con.execute(
        """
        UPDATE inventory
        SET has_program_metadata = FALSE, program_type = NULL,
            program_award = NULL, program_url = NULL
        WHERE has_program_metadata;
        """
    )
    con.executemany(
        """
        UPDATE inventory
        SET has_program_metadata = TRUE, program_type = ?, program_award = ?, program_url = ?
        WHERE corpusid = ?;
        """,
        updates,
    )
    n_tier_a = con.execute(
        f"""
        UPDATE inventory SET tier = 'a'
        WHERE venue_key = 'chi' AND year BETWEEN {y_lo} AND {y_hi}
        RETURNING corpusid;
        """
    ).fetchall()
    report["tier_a_rows"] = len(n_tier_a)

    # Cross-check vs existing CHI'22 GROBID+S2 target corpusids.
    if args.grobid_s2_dir.is_dir():
        targets: set[int] = set()
        for p in args.grobid_s2_dir.glob("*.citations.s2.json"):
            try:
                tid = json.loads(p.read_text(encoding="utf-8")).get("meta", {}).get("target_corpusId")
                if tid is not None:
                    targets.add(int(tid))
            except Exception:
                continue
        report["chi22_crosscheck"] = {
            "grobid_s2_targets": len(targets),
            "program_matched_2022": len(matched_cids_2022),
            "overlap": len(targets & matched_cids_2022),
            "grobid_targets_not_program_matched": len(targets - matched_cids_2022),
        }

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    summary = {y: f"{d['matched']}/{d['rows']}" for y, d in report["per_year"].items()}
    print(json.dumps({"matched_per_year": summary,
                      "tier_a_rows": report["tier_a_rows"],
                      "chi22_crosscheck": report.get("chi22_crosscheck"),
                      "report": str(args.report)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
