#!/usr/bin/env python3
"""
Sanity checks for the multi-venue inventory DuckDB.

1) Per-venue/year counts, with optional DBLP stream counts for comparison
   (--dblp). DBLP counts the whole stream (incl. extended abstracts /
   companion volumes the registry excludes), so DBLP is an UPPER bound;
   flag venue-years where inventory > DBLP (over-matching) or
   inventory < 40% of DBLP (likely recall problem).
2) Match-signal agreement: share of rows matched by >= 2 independent signals
   (uuid/name/dblp) per venue. Low agreement = check the registry aliases.
3) Random sample rows per venue for manual precision spot-checks.

Example:
  uv run python verify_inventory.py --dblp --out data/inventory/verify_report.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import duckdb
import requests

DBLP_API = "https://dblp.org/search/publ/api"


def dblp_stream_count(stream: str, year: int, *, session: requests.Session) -> int | None:
    try:
        resp = session.get(
            DBLP_API,
            params={"q": f"stream:streams/{stream.rstrip('/')}: year:{year}:", "format": "json", "h": 0},
            timeout=30,
        )
        resp.raise_for_status()
        return int(resp.json()["result"]["hits"]["@total"])
    except Exception:
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", type=Path, default=Path("data/inventory/hci_inventory_2026-01-27.duckdb"))
    ap.add_argument("--registry", type=Path, default=Path("data/venue_registry/venues.json"))
    ap.add_argument("--dblp", action="store_true", help="Fetch DBLP per-venue/year counts (slow, ~1 req/s).")
    ap.add_argument("--sample-per-venue", type=int, default=5)
    ap.add_argument("--out", type=Path, default=Path("data/inventory/verify_report.json"))
    args = ap.parse_args()

    reg = json.loads(args.registry.read_text(encoding="utf-8"))
    dblp_prefix = {v["key"]: (v.get("dblp_key_prefixes") or [None])[0] for v in reg["venues"]}

    con = duckdb.connect(str(args.db), read_only=True)
    report: dict = {"per_venue_year": [], "signal_agreement": [], "samples": {}, "flags": []}

    rows = con.execute(
        """
        SELECT venue_key, year, count(*) AS n
        FROM inventory GROUP BY 1, 2 ORDER BY 1, 2;
        """
    ).fetchall()

    session = requests.Session()
    session.headers["User-Agent"] = "citation-impact-agent verify_inventory (academic research)"
    for venue_key, year, n in rows:
        entry = {"venue_key": venue_key, "year": int(year), "inventory": int(n)}
        if args.dblp:
            prefix = dblp_prefix.get(venue_key)
            if prefix:
                stream = prefix.rstrip("/")
                cnt = dblp_stream_count(stream, int(year), session=session)
                entry["dblp_stream_total"] = cnt
                if cnt is not None and cnt > 0:
                    ratio = n / cnt
                    entry["inventory_over_dblp"] = round(ratio, 3)
                    if n > cnt:
                        report["flags"].append(f"{venue_key} {year}: inventory {n} > DBLP {cnt} (over-match?)")
                    elif ratio < 0.4:
                        report["flags"].append(f"{venue_key} {year}: inventory {n} < 40% of DBLP {cnt} (recall?)")
                time.sleep(1.0)  # DBLP is rate-limited; be polite.
        report["per_venue_year"].append(entry)

    agree = con.execute(
        """
        SELECT venue_key,
               count(*) AS n,
               sum(CASE WHEN length(match_signals) - length(replace(match_signals, ',', '')) >= 1
                        THEN 1 ELSE 0 END) AS n_multi_signal
        FROM inventory GROUP BY 1 ORDER BY 1;
        """
    ).fetchall()
    for venue_key, n, n_multi in agree:
        report["signal_agreement"].append(
            {"venue_key": venue_key, "n": int(n), "multi_signal_share": round(int(n_multi) / int(n), 3)}
        )

    for (venue_key,) in con.execute("SELECT DISTINCT venue_key FROM inventory ORDER BY 1;").fetchall():
        sample = con.execute(
            f"""
            SELECT corpusid, year, title, s2_venue_string, dblp_key, match_signals
            FROM inventory WHERE venue_key = '{venue_key}'
            ORDER BY hash(corpusid) LIMIT {int(args.sample_per_venue)};
            """
        ).fetchall()
        report["samples"][venue_key] = [
            {"corpusid": r[0], "year": r[1], "title": r[2], "venue": r[3], "dblp": r[4], "signals": r[5]}
            for r in sample
        ]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"Wrote {args.out}")
    print("\nPer-venue totals:")
    totals = con.execute("SELECT venue_key, tier, count(*) FROM inventory GROUP BY 1,2 ORDER BY 2,3 DESC;").fetchall()
    for venue_key, tier, n in totals:
        print(f"  tier {tier}  {venue_key:18s} {n}")
    print("\nSignal agreement (share matched by >=2 signals):")
    for e in report["signal_agreement"]:
        print(f"  {e['venue_key']:18s} n={e['n']:<7} multi={e['multi_signal_share']}")
    if report["flags"]:
        print("\nFLAGS:")
        for f in report["flags"]:
            print(f"  - {f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
