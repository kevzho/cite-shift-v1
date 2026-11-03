#!/usr/bin/env python3
"""Build the adjudication worksheet from two coders' calibration files.

Lists only the rows where the coders differ, grouped by dimension and ordered
by how diagnostic the dimension is -- the dimensions where a whole category
failed to be co-assigned come first, because those are boundary defects in the
codebook rather than disagreements about cases.

Deliberately omits kappa, agreement rates, and every other aggregate. Coders
are blind to those under CONTRACT.md, and a number on the page would anchor the
discussion before the cases have been read.

Usage:
    python3 analysis/measurement/25_adjudication_worksheet.py \
        --repo annotations/coders/round_one --out annotations/ADJUDICATION_WORKSHEET.md
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

# Ordered most diagnostic first. The rationale in each entry is what the
# session is trying to resolve, not a summary of the disagreement.
DIMENSIONS = [
    ("calibration_multidimensional_blinded", "contribution_realization",
     "`partial` was used by both coders and co-assigned zero times. Decide first "
     "whether `no_background` is a degree on the same scale as `yes`/`partial`, "
     "or a different kind of judgement that should not compete with it."),
    ("calibration_multidimensional_blinded", "citation_function",
     "Five of seven categories were never co-assigned. The scheme is operating "
     "as a two-category one. Which categories are real and which are unreachable "
     "from a citance?"),
    ("calibration_multidimensional_blinded", "cited_contribution_type",
     "The dimension asks about the cited paper, which is not in the evidence. "
     "Separate 'cannot tell from this citance' from 'the categories do not "
     "separate here' -- they need different tokens."),
    ("calibration_compact_blinded", "compact_intent",
     "One citance doing two things. Does the multi-label rule extend here, and "
     "if it does, is the pass still comparable to SciCite and ACL-ARC?"),
    ("calibration_multidimensional_blinded", "rhetorical_polarity",
     "`supports` against `neutral_use` carries nearly all of it. Where is the "
     "line between using a finding and endorsing it?"),
    ("calibration_multidimensional_blinded", "centrality",
     "Highest agreement of the six. Review quickly unless a case is instructive."),
]

KEY = "edge_id"
CONTEXT_CHARS = 400


def read(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def pick(repo: Path, coder: str, stem: str) -> Path:
    folder = repo / "coders" / coder
    norm = folder / f"{stem}_normalized.csv"
    return norm if norm.exists() else folder / f"{stem}.csv"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--coder-a", default="annotator_a")
    ap.add_argument("--coder-b", default="annotator_b")
    args = ap.parse_args()

    cache: dict[str, tuple[dict, dict]] = {}
    for stem in {s for s, _, _ in DIMENSIONS}:
        a = {r[KEY]: r for r in read(pick(args.repo, args.coder_a, stem))}
        b = {r[KEY]: r for r in read(pick(args.repo, args.coder_b, stem))}
        cache[stem] = (a, b)

    lines = [
        "# Adjudication worksheet — calibration set",
        "",
        "Only the rows where the two coders differ. Ordered by how diagnostic the",
        "dimension is, not by edge number.",
        "",
        "**How to run the session.** Work the cases, not the counts — no agreement",
        "statistics appear here on purpose. For each disagreement the question is not",
        "*who was right* but *what wording would have made both of you choose the same",
        "token*. Record the answer as a codebook anchor. \"Edge 47 is `partial`\" is worth",
        "nothing next week; \"a citation realises the contribution partially when ___\" is",
        "the deliverable.",
        "",
        "Both coders' original files are committed unchanged and stay that way.",
        "Adjudication produces a new file; it never overwrites a coder's judgement.",
        "",
    ]

    for stem, dim, rationale in DIMENSIONS:
        a, b = cache[stem]
        shared = [k for k in a if k in b]
        diffs = [k for k in shared if a[k].get(dim, "") != b[k].get(dim, "")]

        lines += [f"## {dim}", "", f"*{rationale}*", "",
                  f"**{len(diffs)} of {len(shared)} rows differ.**", ""]
        if not diffs:
            lines += ["No disagreements.", ""]
            continue

        # Category pairs that recur are boundary problems; single instances are cases.
        pairs: dict[tuple[str, str], int] = {}
        for k in diffs:
            pair = tuple(sorted((a[k][dim], b[k][dim])))
            pairs[pair] = pairs.get(pair, 0) + 1
        recurring = sorted(pairs.items(), key=lambda kv: -kv[1])
        lines += ["| pair | n |", "|---|---|"]
        lines += [f"| `{x}` vs `{y}` | {n} |" for (x, y), n in recurring]
        lines += ["", "Start with the most frequent pair — a boundary that fails "
                      "repeatedly is a wording defect, not a hard case.", ""]

        order = {pair: i for i, (pair, _) in enumerate(recurring)}
        diffs.sort(key=lambda k: order[tuple(sorted((a[k][dim], b[k][dim])))])

        for k in diffs:
            ra, rb = a[k], b[k]
            context = (ra.get("citation_context") or "").strip().replace("\n", " ")
            if len(context) > CONTEXT_CHARS:
                context = context[:CONTEXT_CHARS] + "…"
            lines += [
                f"### `{k[:12]}` — A: `{ra[dim]}`  ·  B: `{rb[dim]}`",
                "",
                f"**Cited:** {ra.get('cited_title', '').strip()}",
                "",
                f"**Citing:** {ra.get('citing_title', '').strip()}",
                "",
                f"> {context}",
                "",
            ]
            for who, row in (("A", ra), ("B", rb)):
                note = (row.get("annotator_note") or "").strip()
                if note:
                    lines += [f"*{who} noted:* {note}", ""]
            lines += ["**Agreed wording:**", "", "---", ""]

    args.out.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {args.out} ({len(lines)} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
