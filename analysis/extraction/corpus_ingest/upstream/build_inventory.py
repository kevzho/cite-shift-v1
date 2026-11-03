#!/usr/bin/env python3
"""
Build a multi-venue HCI paper inventory from the on-disk S2AG `papers` dump.

Three subcommands (run in order):

1) scan    One pass over `papers/*.gz` (DuckDB native JSON reader, resumable
           per shard). Keeps a broad superset of rows matching ANY registry
           signal (DBLP key prefix, venue-name alias, pinned venue UUID)
           within the global year range -> `candidates` table.
2) report  Aggregate candidates per venue key: which (venue string,
           publicationvenueid) combos each signal found, with counts.
           Use this to pin `s2_publication_venue_ids` / fix aliases in the
           registry, then re-run `build` (no rescan needed).
3) build   Apply the precise registry matching rule to candidates ->
           `memberships` (all matches) + `inventory` (one row per corpusid,
           highest-priority tier) + per-venue-year JSONL exports.

Matching rule (per venue): match if ANY of
  (a) publicationvenueid in s2_publication_venue_ids
  (b) lower(venue) equals a lowercased s2_venue_names alias
  (c) externalids.DBLP starts with a dblp_key_prefixes entry
AND year within year_range; rejected outright if lower(venue) is in
s2_venue_names_exclude (e.g. CHI EA shares conf/chi/ DBLP keys with CHI).

Example:
  uv run python build_inventory.py scan \
    --papers-dir data/s2ag/2026-01-27/papers \
    --registry data/venue_registry/venues.json \
    --db data/inventory/hci_inventory_2026-01-27.duckdb --resume
  uv run python build_inventory.py report --db ... --registry ...
  uv run python build_inventory.py build --db ... --registry ...
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Iterable

import duckdb
from tqdm import tqdm


# ----------------------------- registry -----------------------------


def load_registry(registry_path: Path) -> dict[str, Any]:
    reg = json.loads(registry_path.read_text(encoding="utf-8"))
    if not isinstance(reg.get("venues"), list):
        raise ValueError(f"Registry {registry_path} has no 'venues' list")
    return reg


def _sql_quote(s: str) -> str:
    return "'" + s.replace("'", "''") + "'"


def _global_year_range(venues: list[dict]) -> tuple[int, int]:
    los = [v["year_range"][0] for v in venues if v.get("year_range")]
    his = [v["year_range"][1] for v in venues if v.get("year_range")]
    return (min(los), max(his))


def _all_scan_terms(venues: list[dict]) -> tuple[list[str], list[str], list[str]]:
    """Broad scan terms across ALL venues (incl. disabled, so enabling later
    needs no rescan): names (incl. excludes -- we want those rows for the
    report), dblp prefixes, pinned uuids."""
    names: set[str] = set()
    prefixes: set[str] = set()
    uuids: set[str] = set()
    for v in venues:
        for n in v.get("s2_venue_names", []) + v.get("s2_venue_names_exclude", []):
            names.add(n.lower())
        for p in v.get("dblp_key_prefixes", []):
            prefixes.add(p)
        for u in v.get("s2_publication_venue_ids", []):
            uuids.add(u)
    return sorted(names), sorted(prefixes), sorted(uuids)


# ----------------------------- progress -----------------------------


def _iter_gz_shards(dir_path: Path) -> list[Path]:
    return sorted(dir_path.glob("*.gz"))


def _load_progress(progress_path: Path) -> set[str]:
    if not progress_path.exists():
        return set()
    try:
        obj = json.loads(progress_path.read_text(encoding="utf-8"))
    except Exception:
        return set()
    done = obj.get("done_shards")
    return {str(x) for x in done} if isinstance(done, list) else set()


def _save_progress(progress_path: Path, *, done_shards: Iterable[str]) -> None:
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"updated_at_unix": time.time(), "done_shards": sorted(set(done_shards))}
    progress_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


# ----------------------------- scan -----------------------------

CANDIDATES_DDL = """
CREATE TABLE IF NOT EXISTS candidates (
  corpusid BIGINT,
  doi VARCHAR,
  dblp_key VARCHAR,
  title VARCHAR,
  year SMALLINT,
  venue VARCHAR,
  publicationvenueid VARCHAR,
  citationcount BIGINT,
  referencecount BIGINT,
  influentialcitationcount BIGINT,
  isopenaccess BOOLEAN,
  publicationdate VARCHAR,
  journal_name VARCHAR,
  s2_fields VARCHAR,
  url VARCHAR,
  authors_json VARCHAR
);
"""


def cmd_scan(args: argparse.Namespace) -> None:
    reg = load_registry(args.registry)
    venues = reg["venues"]
    names, prefixes, uuids = _all_scan_terms(venues)
    y_lo, y_hi = _global_year_range(venues)

    con = duckdb.connect(str(args.db))
    if args.threads > 0:
        con.execute(f"PRAGMA threads={int(args.threads)};")
    con.execute(CANDIDATES_DDL)

    progress_path = args.db.parent / (args.db.stem + ".scan_progress.json")
    done = _load_progress(progress_path) if args.resume else set()
    if not args.resume:
        con.execute("DELETE FROM candidates;")

    shards = _iter_gz_shards(args.papers_dir)
    if args.max_shards > 0:
        shards = shards[: args.max_shards]

    name_list = ", ".join(_sql_quote(n) for n in names)
    clauses = [f"lower(venue) IN ({name_list})"]
    if prefixes:
        clauses.append(
            "(" + " OR ".join(f"dblp_key LIKE {_sql_quote(p + '%')}" for p in prefixes) + ")"
        )
    if uuids:
        uuid_list = ", ".join(_sql_quote(u) for u in uuids)
        clauses.append(f"publicationvenueid IN ({uuid_list})")
    match_clause = " OR ".join(clauses)

    done_mut = set(done)
    for shard in tqdm(shards, desc="S2AG papers shards", unit="shard", dynamic_ncols=True):
        if shard.name in done_mut:
            continue
        # Explicit column types: robust against per-shard schema inference drift.
        con.execute(
            f"""
            INSERT INTO candidates
            WITH src AS (
              SELECT
                try_cast(corpusid AS BIGINT) AS corpusid,
                json_extract_string(externalids, '$.DOI') AS doi,
                json_extract_string(externalids, '$.DBLP') AS dblp_key,
                title,
                try_cast(year AS SMALLINT) AS year,
                venue,
                publicationvenueid,
                try_cast(citationcount AS BIGINT) AS citationcount,
                try_cast(referencecount AS BIGINT) AS referencecount,
                try_cast(influentialcitationcount AS BIGINT) AS influentialcitationcount,
                try_cast(isopenaccess AS BOOLEAN) AS isopenaccess,
                publicationdate,
                json_extract_string(journal, '$.name') AS journal_name,
                CASE WHEN s2fieldsofstudy IS NULL THEN NULL
                     ELSE array_to_string(list_distinct(list_transform(s2fieldsofstudy, x -> x.category)), ',')
                END AS s2_fields,
                url,
                CAST(authors AS VARCHAR) AS authors_json
              FROM read_json(
                '{shard.as_posix()}',
                format='newline_delimited',
                columns={{
                  'corpusid': 'BIGINT', 'externalids': 'JSON', 'title': 'VARCHAR',
                  'authors': 'JSON', 'venue': 'VARCHAR', 'publicationvenueid': 'VARCHAR',
                  'year': 'BIGINT', 'referencecount': 'BIGINT', 'citationcount': 'BIGINT',
                  'influentialcitationcount': 'BIGINT', 'isopenaccess': 'BOOLEAN',
                  'publicationdate': 'VARCHAR', 'journal': 'JSON', 'url': 'VARCHAR',
                  's2fieldsofstudy': 'STRUCT(category VARCHAR, "source" VARCHAR)[]'
                }}
              )
            )
            SELECT * FROM src
            WHERE corpusid IS NOT NULL
              AND year BETWEEN {y_lo} AND {y_hi}
              AND ({match_clause});
            """
        )
        done_mut.add(shard.name)
        _save_progress(progress_path, done_shards=done_mut)

    n = con.execute("SELECT COUNT(*) FROM candidates;").fetchone()[0]
    print(json.dumps({"candidates": int(n), "shards_done": len(done_mut)}, indent=2))


# ----------------------------- precise matching -----------------------------


def _venue_match_sql(v: dict, *, dblp_only_blocklist: list[str]) -> tuple[str, str, str] | None:
    """Return (where_clause, signals_expr, reject_clause) for one venue."""
    uuids = v.get("s2_publication_venue_ids", [])
    names = [n.lower() for n in v.get("s2_venue_names", [])]
    excludes = [n.lower() for n in v.get("s2_venue_names_exclude", [])]
    prefixes = v.get("dblp_key_prefixes", [])
    y_lo, y_hi = v["year_range"]

    sig_parts: list[tuple[str, str]] = []
    if uuids:
        sig_parts.append(("uuid", "publicationvenueid IN (%s)" % ", ".join(_sql_quote(u) for u in uuids)))
    if names:
        sig_parts.append(("name", "lower(venue) IN (%s)" % ", ".join(_sql_quote(n) for n in names)))
    if prefixes:
        dblp_clause = "(" + " OR ".join(f"dblp_key LIKE {_sql_quote(p + '%')}" for p in prefixes) + ")"
        if dblp_only_blocklist:
            # A dblp-prefix hit alone must not look like a workshop/companion/
            # adjunct volume sharing the conference's DBLP key prefix.
            blocked = " OR ".join(f"lower(venue) LIKE {_sql_quote(p)}" for p in dblp_only_blocklist)
            dblp_clause = f"({dblp_clause} AND NOT (coalesce({blocked}, FALSE)))"
        sig_parts.append(("dblp", dblp_clause))
    if not sig_parts:
        return None

    any_clause = " OR ".join(c for _, c in sig_parts)
    signals_expr = (
        "array_to_string(list_filter(["
        + ", ".join(f"CASE WHEN {c} THEN '{tag}' END" for tag, c in sig_parts)
        + "], x -> x IS NOT NULL), ',')"
    )
    reject = ""
    if excludes:
        reject = "AND lower(venue) NOT IN (%s)" % ", ".join(_sql_quote(n) for n in excludes)
    where = f"({any_clause}) AND year BETWEEN {y_lo} AND {y_hi} {reject}"
    return where, signals_expr, reject


def cmd_build(args: argparse.Namespace) -> None:
    reg = load_registry(args.registry)
    venues = [v for v in reg["venues"] if v.get("enabled", True)]
    blocklist = [p.lower() for p in reg.get("dblp_only_venue_blocklist", [])]
    tier_rank = {"a": 0, "b": 0, "c": 1, "d": 2}  # tier a is a chi-subset flag, not a key

    con = duckdb.connect(str(args.db))
    con.execute(
        """
        CREATE OR REPLACE TABLE memberships (
          corpusid BIGINT,
          venue_key VARCHAR,
          tier VARCHAR,
          year SMALLINT,
          match_signals VARCHAR,
          priority INTEGER
        );
        """
    )
    for i, v in enumerate(venues):
        parts = _venue_match_sql(v, dblp_only_blocklist=blocklist)
        if parts is None:
            print(f"WARN: venue {v['key']} has no matching signals; skipped")
            continue
        where, signals_expr, _ = parts
        priority = tier_rank.get(v["tier"], 9) * 1000 + i
        con.execute(
            f"""
            INSERT INTO memberships
            SELECT corpusid, {_sql_quote(v['key'])}, {_sql_quote(v['tier'])},
                   year, {signals_expr}, {priority}
            FROM candidates
            WHERE {where};
            """
        )

    # Deduplicate: a corpusid can be in candidates once per scan (it is unique
    # within the dump), but defensively dedup memberships per (corpusid, venue_key).
    con.execute(
        """
        CREATE OR REPLACE TABLE memberships AS
        SELECT corpusid, venue_key, tier, year,
               max(match_signals) AS match_signals, min(priority) AS priority
        FROM memberships GROUP BY 1, 2, 3, 4;
        """
    )

    con.execute(
        """
        CREATE OR REPLACE TABLE inventory AS
        WITH best AS (
          SELECT corpusid, venue_key, tier, match_signals,
                 row_number() OVER (PARTITION BY corpusid ORDER BY priority) AS rn
          FROM memberships
        )
        SELECT
          c.corpusid,
          c.doi,
          c.dblp_key,
          c.title,
          c.year,
          b.venue_key,
          b.tier,
          c.venue AS s2_venue_string,
          c.publicationvenueid,
          b.match_signals,
          c.citationcount,
          c.referencecount,
          c.influentialcitationcount,
          c.isopenaccess,
          c.publicationdate,
          c.journal_name,
          c.s2_fields,
          c.url,
          c.authors_json,
          'tier-a flag set by link_chi_program_metadata.py' AS _comment,
          FALSE AS has_program_metadata,
          NULL::VARCHAR AS program_type,
          NULL::VARCHAR AS program_award,
          NULL::VARCHAR AS program_url,
          NULL::VARCHAR AS oa_pdf_url
        FROM best b
        JOIN candidates c USING (corpusid)
        WHERE b.rn = 1;
        """
    )
    con.execute("ALTER TABLE inventory DROP COLUMN _comment;")

    counts = con.execute(
        """
        SELECT venue_key, tier, count(*) AS n
        FROM inventory GROUP BY 1, 2 ORDER BY tier, n DESC;
        """
    ).fetchall()
    total = con.execute("SELECT count(*) FROM inventory;").fetchone()[0]

    if args.jsonl_dir is not None:
        args.jsonl_dir.mkdir(parents=True, exist_ok=True)
        pairs = con.execute("SELECT DISTINCT venue_key, year FROM inventory ORDER BY 1, 2;").fetchall()
        for venue_key, year in tqdm(pairs, desc="JSONL exports", unit="file", dynamic_ncols=True):
            out = args.jsonl_dir / f"{venue_key}_{year}.jsonl"
            con.execute(
                f"""
                COPY (
                  SELECT * EXCLUDE (authors_json), authors_json AS authors
                  FROM inventory
                  WHERE venue_key = {_sql_quote(venue_key)} AND year = {int(year)}
                  ORDER BY corpusid
                ) TO '{out.as_posix()}' (FORMAT JSON);
                """
            )

    print(json.dumps({"inventory_total": int(total)}, indent=2))
    for venue_key, tier, n in counts:
        print(f"  tier {tier}  {venue_key:20s} {n}")


# ----------------------------- report -----------------------------


def cmd_report(args: argparse.Namespace) -> None:
    reg = load_registry(args.registry)
    con = duckdb.connect(str(args.db), read_only=True)

    blocklist = [p.lower() for p in reg.get("dblp_only_venue_blocklist", [])]
    report: dict[str, Any] = {"generated_at_unix": time.time(), "venues": {}}
    for v in reg["venues"]:
        parts = _venue_match_sql(v, dblp_only_blocklist=blocklist)
        if parts is None:
            continue
        where, signals_expr, _ = parts
        rows = con.execute(
            f"""
            SELECT venue, publicationvenueid, {signals_expr} AS signals,
                   count(*) AS n, min(year) AS y_min, max(year) AS y_max
            FROM candidates
            WHERE {where}
            GROUP BY 1, 2, 3 ORDER BY n DESC LIMIT 50;
            """
        ).fetchall()
        report["venues"][v["key"]] = [
            {
                "venue": r[0],
                "publicationvenueid": r[1],
                "signals": r[2],
                "n": int(r[3]),
                "years": [r[4], r[5]],
            }
            for r in rows
        ]

    out = args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {out}")
    for key, rows in report["venues"].items():
        total = sum(r["n"] for r in rows)
        print(f"\n== {key} (~{total} rows, top combos) ==")
        for r in rows[:6]:
            print(f"  n={r['n']:<7} signals={r['signals']:<15} uuid={r['publicationvenueid']} venue={r['venue']!r}")


# ----------------------------- main -----------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--registry", type=Path, default=Path("data/venue_registry/venues.json"))
    common.add_argument("--db", type=Path, default=Path("data/inventory/hci_inventory_2026-01-27.duckdb"))

    ap_scan = sub.add_parser("scan", parents=[common], help="Scan papers dump into candidates table")
    ap_scan.add_argument("--papers-dir", type=Path, default=Path("data/s2ag/2026-01-27/papers"))
    ap_scan.add_argument("--resume", action="store_true")
    ap_scan.add_argument("--max-shards", type=int, default=0, help="Debug: only first N shards (0 = all)")
    ap_scan.add_argument("--threads", type=int, default=0, help="DuckDB threads (0 = default)")
    ap_scan.set_defaults(func=cmd_scan)

    ap_report = sub.add_parser("report", parents=[common], help="Aggregate candidates per venue for registry pinning")
    ap_report.add_argument("--out", type=Path, default=Path("data/inventory/venue_report.json"))
    ap_report.set_defaults(func=cmd_report)

    ap_build = sub.add_parser("build", parents=[common], help="Build memberships + inventory + JSONL exports")
    ap_build.add_argument("--jsonl-dir", type=Path, default=Path("data/inventory/jsonl"))
    ap_build.add_argument("--no-jsonl", dest="jsonl_dir", action="store_const", const=None)
    ap_build.set_defaults(func=cmd_build)

    args = ap.parse_args()
    args.db.parent.mkdir(parents=True, exist_ok=True)
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
