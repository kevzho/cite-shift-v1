#!/usr/bin/env python3
"""
Extract a deterministic seed set of S2AG corpusIds from `data/chi22_grobid_s2`.

Seed definition (current project convention):
- Include each file's `meta.target_corpusId` (the CHI'22 paper).
- Include each cited paper's `citations[].s2.paper.corpusId` (when present).

Output:
- A text file with one corpusId per line (sorted numeric).
- Optionally a small JSON manifest describing the extraction.

Example:
  uv run python extract_seed_corpus_ids.py \
    --in-dir data/chi22_grobid_s2 \
    --out data/corpus_seeds/chi22_seed_corpus_ids.txt
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from tqdm import tqdm


@dataclass
class ExtractStats:
    files_seen: int = 0
    files_ok: int = 0
    files_failed: int = 0
    target_ids_added: int = 0
    cited_ids_added: int = 0
    citations_seen: int = 0
    citations_with_s2: int = 0
    citations_with_corpus_id: int = 0


def _iter_citation_files(in_dir: Path) -> list[Path]:
    return sorted(in_dir.glob("*.citations.s2.json"))


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


def _safe_get(d: Any, *keys: str) -> Any:
    cur = d
    for k in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(k)
    return cur


def extract_seed_ids(
    files: Iterable[Path],
    *,
    stats: ExtractStats,
    show_progress: bool = True,
) -> set[int]:
    seed: set[int] = set()
    files_list = list(files)
    it = tqdm(files_list, desc="CHI22 citation files", unit="file", dynamic_ncols=True) if show_progress else files_list
    for path in it:
        stats.files_seen += 1
        try:
            obj = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            stats.files_failed += 1
            continue

        stats.files_ok += 1

        target_id = _as_int(_safe_get(obj, "meta", "target_corpusId"))
        if target_id is not None and target_id not in seed:
            seed.add(target_id)
            stats.target_ids_added += 1

        citations = obj.get("citations")
        if not isinstance(citations, list):
            continue
        for c in citations:
            stats.citations_seen += 1
            s2 = c.get("s2") if isinstance(c, dict) else None
            if not isinstance(s2, dict):
                continue
            stats.citations_with_s2 += 1
            cid = _as_int(_safe_get(s2, "paper", "corpusId"))
            if cid is None:
                continue
            stats.citations_with_corpus_id += 1
            if cid not in seed:
                seed.add(cid)
                stats.cited_ids_added += 1

    return seed


def _write_seed_file(seed: set[int], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = "\n".join(str(x) for x in sorted(seed)) + "\n"
    out_path.write_text(lines, encoding="utf-8")


def _write_manifest(
    *,
    manifest_path: Path,
    in_dir: Path,
    out_path: Path,
    files: list[Path],
    stats: ExtractStats,
) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "in_dir": str(in_dir),
        "out": str(out_path),
        "n_seed_ids": (stats.target_ids_added + stats.cited_ids_added),
        "stats": stats.__dict__,
        "files": [str(p) for p in files],
    }
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", type=Path, default=Path("data/chi22_grobid_s2"))
    ap.add_argument("--out", type=Path, default=Path("data/corpus_seeds/chi22_seed_corpus_ids.txt"))
    ap.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/corpus_seeds/chi22_seed_corpus_ids.manifest.json"),
        help="Write a small JSON manifest (determinism + provenance).",
    )
    ap.add_argument("--no-manifest", action="store_true", help="Do not write manifest JSON.")
    ap.add_argument("--max-files", type=int, default=0, help="Debug: only read first N files (0 = all).")
    args = ap.parse_args()

    files = _iter_citation_files(args.in_dir)
    if args.max_files and args.max_files > 0:
        files = files[: args.max_files]

    stats = ExtractStats()
    seed = extract_seed_ids(files, stats=stats, show_progress=True)

    _write_seed_file(seed, args.out)
    if not args.no_manifest:
        _write_manifest(
            manifest_path=args.manifest,
            in_dir=args.in_dir,
            out_path=args.out,
            files=files,
            stats=stats,
        )

    print(
        json.dumps(
            {
                "out": str(args.out),
                "n_seed_ids": len(seed),
                "files_ok": stats.files_ok,
                "files_failed": stats.files_failed,
                "target_ids_added": stats.target_ids_added,
                "cited_ids_added": stats.cited_ids_added,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

