#!/usr/bin/env python3
"""Freeze the measurement study's full local record (the repository tracks code only).

Hashes every file under the record directories into MANIFEST.json, packs them (minus model weights, which are
hashed but not packed) into one tar.gz, and writes freeze/FREEZE_<date>.json with the archive's and manifest's
sha256 so a git commit pins the private record without publishing it.

    python3 freeze/make_freeze.py OUTDIR [--extra LABEL=PATH ...]
"""
import argparse
import hashlib
import json
import os
import sys
import tarfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RECORD = ["annotations", "docs", "private/preregistration", "private/plans", "analysis", "paper_iconf",
          "paper_sts", "reproduce", "README.md", "requirements.txt"]
WEIGHTS = {".safetensors", ".bin", ".pt", ".pth", ".ckpt"}
SKIP = {".DS_Store"}


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def files(base: Path):
    if base.is_file():
        yield base
        return
    for dp, dn, fn in os.walk(base):
        dn[:] = sorted(d for d in dn if d != "__pycache__")
        for n in sorted(fn):
            if n not in SKIP:
                yield Path(dp) / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("outdir", type=Path)
    ap.add_argument("--extra", action="append", default=[], help="LABEL=PATH, packed under extra/LABEL/")
    a = ap.parse_args()
    date = datetime.now().strftime("%Y-%m-%d")
    entries = []
    for r in RECORD:
        for p in files(ROOT / r):
            entries.append((str(p.relative_to(ROOT)), p))
    for spec in a.extra:
        label, path = spec.split("=", 1)
        for p in files(Path(path)):
            entries.append((f"extra/{label}/{p.relative_to(Path(path)) if Path(path).is_dir() else p.name}", p))
    manifest, packed = [], 0
    a.outdir.mkdir(parents=True, exist_ok=True)
    arc = a.outdir / f"measurement_record_{date}.tar.gz"
    if arc.exists():
        sys.exit(f"refusing to overwrite {arc}")
    with tarfile.open(arc, "w:gz") as tar:
        for rel, p in entries:
            st = p.stat()
            is_w = p.suffix in WEIGHTS
            manifest.append({"path": rel, "bytes": st.st_size, "sha256": sha(p),
                             "mtime": datetime.fromtimestamp(st.st_mtime, timezone.utc).isoformat(timespec="seconds"),
                             "packed": not is_w})
            if not is_w:
                tar.add(p, arcname=rel, recursive=False)
                packed += 1
    mp = a.outdir / "MANIFEST.json"
    mp.write_text(json.dumps(manifest, indent=1) + "\n")
    summary = {"date": date, "archive": arc.name, "archive_sha256": sha(arc), "archive_bytes": arc.stat().st_size,
               "manifest_sha256": sha(mp), "files": len(manifest), "files_packed": packed,
               "weights_hashed_not_packed": len(manifest) - packed,
               "total_bytes": sum(m["bytes"] for m in manifest),
               "record_dirs": RECORD + [f"extra/{s.split('=', 1)[0]}" for s in a.extra],
               "note": "Code is tracked in git; data, labels, logs, results and manuscripts are private and pinned here by hash."}
    (ROOT / "freeze" / f"FREEZE_{date}.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
