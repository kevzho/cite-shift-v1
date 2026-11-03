#!/usr/bin/env python3
"""One-off: replace machine-specific absolute paths in runners and harnesses with environment variables
that default to the same locations relative to the repository (2026-10-27).

  CITATION_ROOT   repository root            (default: computed from the file's own location)
  CITATION_VENVS  per-system virtualenvs     (default: ~/.cache/citation_corpus/venvs)
  PYTHON          main interpreter           (default: python3 on PATH)
  UV              uv binary                  (default: uv on PATH)

Usage: python3 reproduce/tools/fix_abs_paths.py [--dry-run]
"""
from __future__ import annotations

import ast
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REPO_ABS = "/Volumes/kz_extended/SALT-UIUC/citation_iConf/citation_corpus"
VENVS_ABS = "/Users/kevinzhou/.cache/citation_corpus/venvs"
PY_ABS = "/Users/kevinzhou/miniconda3/bin/python3"
UV_ABS = "/Users/kevinzhou/.local/bin/uv"
SKIP = ("open_llm_intent", "pst_bench")  # environments no longer exist; not used by the paper


def rel_to_root(f: Path) -> str:
    return os.path.relpath(ROOT, f.parent)


def fix_sh(f: Path, s: str) -> str:
    lines = s.split("\n")
    defs = []
    if REPO_ABS in s:
        defs.append(f'CITATION_ROOT="${{CITATION_ROOT:-$(cd "$(dirname "${{BASH_SOURCE[0]}}")/{rel_to_root(f)}" && pwd)}}"')
    if VENVS_ABS in s:
        defs.append('CITATION_VENVS="${CITATION_VENVS:-$HOME/.cache/citation_corpus/venvs}"')
    if PY_ABS in s:
        defs.append('PYTHON="${PYTHON:-python3}"')
    if UV_ABS in s:
        defs.append('UV_BIN="${UV:-uv}"')
    body = "\n".join(lines[1:]) if lines[0].startswith("#!") else "\n".join(lines)
    # python heredocs inside runners are quoted ('EOF'), so read the exported variable instead
    body = body.replace(f'Path("{REPO_ABS}")', 'Path(__import__("os").environ["CITATION_ROOT"])')
    body = (body.replace(REPO_ABS, "${CITATION_ROOT}").replace(VENVS_ABS, "${CITATION_VENVS}")
            .replace(PY_ABS, "${PYTHON}").replace(UV_ABS, "${UV_BIN}"))
    head = lines[0] if lines[0].startswith("#!") else "#!/bin/bash"
    # export so that python heredocs and child scripts see the same values
    exports = [d for d in defs] + ["export " + " ".join(d.split("=")[0] for d in defs)] if defs else []
    return "\n".join([head, *exports, body])


def insert_point(src: str) -> int:
    """Line index after the module docstring and any `from __future__` imports."""
    tree = ast.parse(src)
    idx = 0
    for node in tree.body:
        if isinstance(node, ast.Expr) and isinstance(getattr(node, "value", None), ast.Constant) and isinstance(node.value.value, str) and idx == 0:
            idx = node.end_lineno
            continue
        if isinstance(node, ast.ImportFrom) and node.module == "__future__":
            idx = node.end_lineno
            continue
        break
    return idx


def fix_py(f: Path, s: str) -> str:
    depth = len(f.relative_to(ROOT).parts) - 1
    need_root, need_venvs = REPO_ABS in s, VENVS_ABS in s
    # quoted literals: "<abs>" , "<abs>/rest" , '<abs>/rest'
    for abs_, var in ((REPO_ABS, "_CITATION_ROOT"), (VENVS_ABS, "_CITATION_VENVS")):
        s = re.sub(r'(["\'])' + re.escape(abs_) + r'\1', var, s)
        s = re.sub(r'(["\'])' + re.escape(abs_) + r'(/[^"\']*)\1', lambda m: f'{var} + {m.group(1)}{m.group(2)}{m.group(1)}', s)
    if REPO_ABS in s or VENVS_ABS in s:
        raise SystemExit(f"unhandled literal in {f}")
    defs = ["import os as _os", "from pathlib import Path as _Path"]
    if need_root:
        defs.append(f'_CITATION_ROOT = _os.environ.get("CITATION_ROOT", str(_Path(__file__).resolve().parents[{depth}]))')
    if need_venvs:
        defs.append('_CITATION_VENVS = _os.environ.get("CITATION_VENVS", _os.path.expanduser("~/.cache/citation_corpus/venvs"))')
    lines = s.split("\n")
    i = insert_point(s)
    return "\n".join(lines[:i] + defs + lines[i:])


def main():
    dry = "--dry-run" in sys.argv
    files = [p for p in (ROOT / "analysis").rglob("*") if p.suffix in (".py", ".sh") and p.is_file()
             and not any(k in p.parts for k in SKIP) and "__pycache__" not in p.parts]
    changed = []
    for f in sorted(files):
        s = f.read_text(encoding="utf-8")
        if not any(a in s for a in (REPO_ABS, VENVS_ABS, PY_ABS, UV_ABS)):
            continue
        new = fix_sh(f, s) if f.suffix == ".sh" else fix_py(f, s)
        changed.append(f.relative_to(ROOT))
        if not dry:
            f.write_text(new, encoding="utf-8")
    print("\n".join(map(str, changed)))
    print(f"{len(changed)} files {'would change' if dry else 'changed'}")


if __name__ == "__main__":
    main()
