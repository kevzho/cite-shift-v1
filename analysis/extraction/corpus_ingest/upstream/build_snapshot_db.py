#!/usr/bin/env python3
"""
Build a snapshot-ready, filtered citation edge store (DuckDB).

Filtering rule:
  Keep a citation edge if either endpoint (citing or cited corpusId) is in a seed set S.

Seed set S is typically produced by `extract_seed_corpus_ids.py` from `data/chi22_grobid_s2`,
but you can point it to any seed list later when you expand beyond CHI'22.

Temporal snapshots:
  The S2AG `citations` dataset has no timestamp per edge.
  For snapshotting, we attach each edge a time key derived from the *citing* paper's publication year
  (from the S2AG `papers` dataset). Then snapshot_at(year=Y) is `WHERE citing_pub_year <= Y`.

This script supports:
- Building `edge_core` from S2AG `citations/*.gz` (JSONL in gzip).
- (Optional) Populating `paper_meta` and updating `edge_core.citing_pub_year` from S2AG `papers/*.gz`.
- A resumable progress file to skip already processed shards.

Examples:
  # 1) Build seed list (CHI'22)
  uv run python extract_seed_corpus_ids.py --in-dir data/chi22_grobid_s2 --out data/corpus_seeds/chi22_seed_corpus_ids.txt

  # 2) Build filtered edge store (no pub-year yet)
  uv run python build_snapshot_db.py \
    --seed-file data/corpus_seeds/chi22_seed_corpus_ids.txt \
    --citations-dir data/s2ag/2026-01-27/citations \
    --out-db data/snapshots/chi22_mentions_2026-01-27.duckdb

  # 3) Download papers dataset (once) then update pub years
  uv run python download_s2ag_dataset.py --dataset papers --release 2026-01-27 --out data/s2ag
  uv run python build_snapshot_db.py \
    --seed-file data/corpus_seeds/chi22_seed_corpus_ids.txt \
    --citations-dir data/s2ag/2026-01-27/citations \
    --papers-dir data/s2ag/2026-01-27/papers \
    --out-db data/snapshots/chi22_mentions_2026-01-27.duckdb \
    --update-pubyear
"""

from __future__ import annotations

import argparse
import gzip
import json
import time
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Iterable

import duckdb
from tqdm import tqdm


@dataclass
class BuildStats:
    shards_seen: int = 0
    shards_skipped_already_done: int = 0
    lines_seen: int = 0
    edges_kept: int = 0
    edges_dropped_null_cited: int = 0
    json_errors: int = 0
    engine: str = "python"


def _as_int(x: Any) -> int | None:
    if x is None:
        return None
    if isinstance(x, bool):
        return None
    if isinstance(x, int):
        return x
    if isinstance(x, str):
        s = x.strip()
        if not s:
            return None
        try:
            return int(s)
        except Exception:
            return None
    return None


def _pub_year_from_paper(obj: dict[str, Any]) -> int | None:
    # Common fields in S2AG papers:
    # - year (int)
    # - publicationDate / publicationdate (YYYY-MM-DD)
    y = obj.get("year")
    yi = _as_int(y)
    if yi is not None and 1000 <= yi <= 3000:
        return yi

    for k in ("publicationDate", "publicationdate", "publication_date"):
        v = obj.get(k)
        if isinstance(v, str) and len(v) >= 4:
            try:
                y2 = int(v[:4])
                if 1000 <= y2 <= 3000:
                    return y2
            except Exception:
                continue
    return None


def _iter_gz_shards(dir_path: Path) -> list[Path]:
    return sorted(dir_path.glob("*.gz"))


def _load_seed_file(seed_file: Path) -> set[int]:
    seed: set[int] = set()
    for line in seed_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            seed.add(int(line))
        except Exception:
            continue
    return seed


def _load_progress(progress_path: Path) -> set[str]:
    if not progress_path.exists():
        return set()
    try:
        obj = json.loads(progress_path.read_text(encoding="utf-8"))
    except Exception:
        return set()
    done = obj.get("done_shards")
    if isinstance(done, list):
        return {str(x) for x in done}
    return set()


def _save_progress(progress_path: Path, *, done_shards: Iterable[str]) -> None:
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "updated_at_unix": time.time(),
        "done_shards": sorted(set(done_shards)),
    }
    progress_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _init_db(con: duckdb.DuckDBPyConnection) -> None:
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS seed_corpus (
          corpusid BIGINT PRIMARY KEY
        );
        """
    )
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS paper_meta (
          corpusid BIGINT PRIMARY KEY,
          pub_year SMALLINT
        );
        """
    )
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS edge_core (
          citationid BIGINT,
          citingcorpusid BIGINT,
          citedcorpusid BIGINT,
          isinfluential BOOLEAN,
          citing_pub_year SMALLINT
        );
        """
    )


def _upsert_seeds(con: duckdb.DuckDBPyConnection, seed: set[int]) -> None:
    con.execute("DELETE FROM seed_corpus;")
    rows = [(x,) for x in sorted(seed)]
    con.executemany("INSERT INTO seed_corpus VALUES (?);", rows)


def build_edges_from_citations(
    *,
    con: duckdb.DuckDBPyConnection,
    citations_dir: Path,
    seed: set[int],
    progress_path: Path,
    resume: bool,
    drop_null_cited: bool,
    max_shards: int,
    max_edges: int,
    engine: str,
) -> BuildStats:
    if engine == "duckdb":
        return _build_edges_from_citations_duckdb(
            con=con,
            citations_dir=citations_dir,
            progress_path=progress_path,
            resume=resume,
            drop_null_cited=drop_null_cited,
            max_shards=max_shards,
            max_edges=max_edges,
        )
    if engine != "python":
        raise ValueError(f"Unknown engine: {engine!r}")

    stats = BuildStats()
    stats.engine = "python"
    done = _load_progress(progress_path) if resume else set()

    shards = _iter_gz_shards(citations_dir)
    if max_shards and max_shards > 0:
        shards = shards[:max_shards]

    # Make builds repeatable if re-run: clear edges unless resuming.
    if not resume:
        con.execute("DROP TABLE IF EXISTS edge_core;")
        con.execute(
            """
            CREATE TABLE edge_core (
              citationid BIGINT,
              citingcorpusid BIGINT,
              citedcorpusid BIGINT,
              isinfluential BOOLEAN,
              citing_pub_year SMALLINT
            );
            """
        )

    kept_total = 0
    done_mut = set(done)

    for shard in tqdm(shards, desc="S2AG citations shards", unit="shard", dynamic_ncols=True):
        stats.shards_seen += 1
        shard_key = shard.name
        if resume and shard_key in done_mut:
            stats.shards_skipped_already_done += 1
            continue

        batch: list[tuple[int | None, int | None, int | None, bool | None, int | None]] = []
        with gzip.open(shard, "rt", encoding="utf-8") as f:
            for line in f:
                stats.lines_seen += 1
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    stats.json_errors += 1
                    continue

                citing = _as_int(obj.get("citingcorpusid"))
                cited = _as_int(obj.get("citedcorpusid"))
                if citing is None:
                    continue

                if drop_null_cited and cited is None:
                    if citing in seed:
                        stats.edges_dropped_null_cited += 1
                    continue

                if (citing not in seed) and (cited is None or cited not in seed):
                    continue

                citationid = _as_int(obj.get("citationid"))
                isinfl = obj.get("isinfluential")
                isinfl_b = bool(isinfl) if isinstance(isinfl, bool) else None

                batch.append((citationid, citing, cited, isinfl_b, None))
                stats.edges_kept += 1
                kept_total += 1

                if len(batch) >= 50_000:
                    con.executemany(
                        "INSERT INTO edge_core VALUES (?, ?, ?, ?, ?);",
                        batch,
                    )
                    batch.clear()

                if max_edges and max_edges > 0 and kept_total >= max_edges:
                    break

        if batch:
            con.executemany("INSERT INTO edge_core VALUES (?, ?, ?, ?, ?);", batch)

        done_mut.add(shard_key)
        _save_progress(progress_path, done_shards=done_mut)

        if max_edges and max_edges > 0 and kept_total >= max_edges:
            break

    return stats


def _build_edges_from_citations_duckdb(
    *,
    con: duckdb.DuckDBPyConnection,
    citations_dir: Path,
    progress_path: Path,
    resume: bool,
    drop_null_cited: bool,
    max_shards: int,
    max_edges: int,
) -> BuildStats:
    """
    Faster path: let DuckDB parse/filter newline-delimited JSON in C++ (and use threads).

    Notes:
    - Stats like `lines_seen`/`json_errors` are not tracked in this engine.
    - `edges_kept` is approximated by rowcount deltas per shard.
    """
    stats = BuildStats(engine="duckdb")
    done = _load_progress(progress_path) if resume else set()

    shards = _iter_gz_shards(citations_dir)
    if max_shards and max_shards > 0:
        shards = shards[:max_shards]

    # Make builds repeatable if re-run: clear edges unless resuming.
    if not resume:
        con.execute("DROP TABLE IF EXISTS edge_core;")
        con.execute(
            """
            CREATE TABLE edge_core (
              citationid BIGINT,
              citingcorpusid BIGINT,
              citedcorpusid BIGINT,
              isinfluential BOOLEAN,
              citing_pub_year SMALLINT
            );
            """
        )

    done_mut = set(done)
    kept_total = con.execute("SELECT COUNT(*) FROM edge_core;").fetchone()[0]

    for shard in tqdm(shards, desc="S2AG citations shards", unit="shard", dynamic_ncols=True):
        stats.shards_seen += 1
        shard_key = shard.name
        if resume and shard_key in done_mut:
            stats.shards_skipped_already_done += 1
            continue

        before = kept_total

        # Build and run a single INSERT..SELECT per shard.
        # We join to seed_corpus to implement "either endpoint in seed set".
        where_null_cited = "AND src.citedcorpusid IS NOT NULL" if drop_null_cited else ""
        con.execute(
            f"""
            WITH src AS (
              SELECT
                try_cast(citationid AS BIGINT) AS citationid,
                try_cast(citingcorpusid AS BIGINT) AS citingcorpusid,
                try_cast(citedcorpusid AS BIGINT) AS citedcorpusid,
                CASE
                  WHEN isinfluential IS NULL THEN NULL
                  ELSE cast(isinfluential AS BOOLEAN)
                END AS isinfluential
              FROM read_json_auto('{shard.as_posix()}', format='newline_delimited')
            )
            INSERT INTO edge_core
            SELECT
              src.citationid,
              src.citingcorpusid,
              src.citedcorpusid,
              src.isinfluential,
              NULL::SMALLINT AS citing_pub_year
            FROM src
            LEFT JOIN seed_corpus s1 ON s1.corpusid = src.citingcorpusid
            LEFT JOIN seed_corpus s2 ON s2.corpusid = src.citedcorpusid
            WHERE src.citingcorpusid IS NOT NULL
              {where_null_cited}
              AND (s1.corpusid IS NOT NULL OR s2.corpusid IS NOT NULL);
            """
        )

        kept_total = con.execute("SELECT COUNT(*) FROM edge_core;").fetchone()[0]
        inserted = int(kept_total - before)
        stats.edges_kept += max(inserted, 0)

        done_mut.add(shard_key)
        _save_progress(progress_path, done_shards=done_mut)

        if max_edges and max_edges > 0 and kept_total >= max_edges:
            break

    return stats


def update_pub_years_from_papers(
    *,
    con: duckdb.DuckDBPyConnection,
    papers_dir: Path,
    max_shards: int,
) -> dict[str, int]:
    # Identify which citing papers we actually need years for.
    needed_rows = con.execute(
        "SELECT DISTINCT citingcorpusid FROM edge_core WHERE citingcorpusid IS NOT NULL;"
    ).fetchall()
    needed = {int(r[0]) for r in needed_rows if r and r[0] is not None}

    if not needed:
        return {"needed": 0, "paper_meta_added": 0, "edges_updated": 0}

    # Avoid re-adding; we overwrite paper_meta for determinism.
    con.execute("DELETE FROM paper_meta;")

    shards = _iter_gz_shards(papers_dir)
    if max_shards and max_shards > 0:
        shards = shards[:max_shards]

    added = 0
    batch: list[tuple[int, int]] = []

    for shard in tqdm(shards, desc="S2AG papers shards", unit="shard", dynamic_ncols=True):
        with gzip.open(shard, "rt", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue

                cid = _as_int(obj.get("corpusid"))
                if cid is None:
                    cid = _as_int(obj.get("corpusId"))
                if cid is None or cid not in needed:
                    continue

                py = _pub_year_from_paper(obj)
                if py is None:
                    continue

                batch.append((cid, py))
                added += 1
                if len(batch) >= 50_000:
                    con.executemany("INSERT INTO paper_meta VALUES (?, ?);", batch)
                    batch.clear()

        if len(batch) >= 50_000:
            con.executemany("INSERT INTO paper_meta VALUES (?, ?);", batch)
            batch.clear()

    if batch:
        con.executemany("INSERT INTO paper_meta VALUES (?, ?);", batch)

    # Update edge_core.
    con.execute(
        """
        UPDATE edge_core
        SET citing_pub_year = pm.pub_year
        FROM paper_meta pm
        WHERE edge_core.citingcorpusid = pm.corpusid;
        """
    )
    updated = con.execute("SELECT COUNT(*) FROM edge_core WHERE citing_pub_year IS NOT NULL;").fetchone()[0]
    return {"needed": len(needed), "paper_meta_added": added, "edges_updated": int(updated)}


def _create_indexes(con: duckdb.DuckDBPyConnection) -> None:
    # Index creation is best-effort: if unsupported in a particular build, ignore.
    try:
        con.execute("CREATE INDEX IF NOT EXISTS edge_core_citing_idx ON edge_core(citingcorpusid);")
        con.execute("CREATE INDEX IF NOT EXISTS edge_core_cited_idx ON edge_core(citedcorpusid);")
        con.execute("CREATE INDEX IF NOT EXISTS edge_core_pubyear_idx ON edge_core(citing_pub_year);")
    except Exception:
        pass


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed-file", type=Path, required=True)
    ap.add_argument("--citations-dir", type=Path, required=True)
    ap.add_argument("--out-db", type=Path, required=True)
    ap.add_argument("--papers-dir", type=Path, default=None, help="Optional: S2AG papers directory for pub years.")
    ap.add_argument("--update-pubyear", action="store_true", help="Populate/update citing_pub_year via papers dataset.")
    ap.add_argument("--resume", action="store_true", help="Resume edge build using a progress file.")
    ap.add_argument("--drop-null-cited", action="store_true", help="Drop edges with null citedcorpusid (recommended).")
    ap.add_argument("--keep-null-cited", action="store_true", help="Keep edges with null citedcorpusid.")
    ap.add_argument(
        "--engine",
        choices=["python", "duckdb"],
        default="duckdb",
        help="Edge build engine. 'duckdb' is usually much faster for S2AG shards.",
    )
    ap.add_argument(
        "--threads",
        type=int,
        default=0,
        help="DuckDB threads (0 = default). Only affects --engine duckdb and SQL operations.",
    )
    ap.add_argument("--max-citation-shards", type=int, default=0, help="Debug: only process first N citation shards.")
    ap.add_argument("--max-paper-shards", type=int, default=0, help="Debug: only process first N paper shards.")
    ap.add_argument("--max-edges", type=int, default=0, help="Debug: stop after keeping N edges total.")
    args = ap.parse_args()

    seed = _load_seed_file(args.seed_file)
    if not seed:
        raise SystemExit(f"Seed file {args.seed_file} produced 0 ids.")

    args.out_db.parent.mkdir(parents=True, exist_ok=True)
    progress_path = args.out_db.with_suffix(args.out_db.suffix + ".progress.json")

    drop_null_cited = True
    if args.keep_null_cited:
        drop_null_cited = False
    elif args.drop_null_cited:
        drop_null_cited = True

    con = duckdb.connect(str(args.out_db))
    if args.threads and args.threads > 0:
        con.execute(f"PRAGMA threads={int(args.threads)};")
    _init_db(con)
    _upsert_seeds(con, seed)

    # Edge build
    t0 = time.time()
    stats = build_edges_from_citations(
        con=con,
        citations_dir=args.citations_dir,
        seed=seed,
        progress_path=progress_path,
        resume=args.resume,
        drop_null_cited=drop_null_cited,
        max_shards=args.max_citation_shards,
        max_edges=args.max_edges,
        engine=args.engine,
    )
    t1 = time.time()

    # Optional pub-year update
    pub_year_result: dict[str, int] | None = None
    if args.update_pubyear:
        if not args.papers_dir:
            raise SystemExit("--update-pubyear requires --papers-dir")
        pub_year_result = update_pub_years_from_papers(
            con=con,
            papers_dir=args.papers_dir,
            max_shards=args.max_paper_shards,
        )

    _create_indexes(con)

    # Summarize
    n_edges = con.execute("SELECT COUNT(*) FROM edge_core;").fetchone()[0]
    n_seed = con.execute("SELECT COUNT(*) FROM seed_corpus;").fetchone()[0]
    n_with_year = con.execute("SELECT COUNT(*) FROM edge_core WHERE citing_pub_year IS NOT NULL;").fetchone()[0]

    out = {
        "out_db": str(args.out_db),
        "progress": str(progress_path),
        "n_seed": int(n_seed),
        "n_edges": int(n_edges),
        "n_edges_with_citing_pub_year": int(n_with_year),
        "build_seconds": round(t1 - t0, 3),
        "edge_build_stats": stats.__dict__,
        "pub_year_update": pub_year_result,
    }
    print(json.dumps(out, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

