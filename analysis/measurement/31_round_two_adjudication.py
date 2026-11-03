#!/usr/bin/env python3
"""Build the round-two adjudication worksheet from the recalibration files.

Ordered by what the session can actually fix, which round two makes visible in
a way round one did not.

Two dimensions missed their declared threshold, and they failed differently.
Polarity's disagreements all run one direction -- one coder consistently reads
less endorsement into the same sentence than the other. That is a threshold
offset and a session can move it. Centrality's disagreements are symmetric on
balanced marginals, which is what independent noise looks like; there is no
offset to correct, and talking longer will not produce one. Ordering the
worksheet by kappa would have put centrality first. Ordering it by whether
disagreement has a direction puts polarity first, which is where the hour is
worth spending.

Directional pair counts are shown; agreement rates and kappa are not. Coders
are blind to those under CONTRACT.md, and a reliability number on the page
would anchor the discussion before the cases have been read. A pair count says
which boundary keeps failing, which is what the session is for.

Usage:
    python3 analysis/measurement/31_round_two_adjudication.py \
        --repo private/history/citation-annotation-2027 \
        --out annotations/ADJUDICATION_WORKSHEET_2026-08-24.md
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import io
import subprocess
from collections import Counter
from pathlib import Path

KEY = "edge_id"
CONTEXT_CHARS = 400
BLANK = "(blank)"

# Same refs as 30_round_two_agreement.py; the coder branches are not merged.
SOURCES = {
    "compact": {
        "A": ("origin/annotator-a", "coders/annotator_a/compact.csv"),
        "B": ("origin/annotator-b",
              "coders/annotator_b/recalibration_compact_blinded_completed.csv"),
    },
    "multidimensional": {
        "A": ("origin/annotator-a", "coders/annotator_a/mult.csv"),
        "B": ("origin/annotator-b",
              "coders/annotator_b/recalibration_multidimensional_blinded_completed.csv"),
    },
}

# Ordered by adjudicability, not by how badly the dimension scored.
DIMENSIONS = [
    ("multidimensional", "rhetorical_polarity",
     "**Start here.** Every disagreement runs the same way: A reads endorsement "
     "where B reads use. Not one case goes the other direction. That is not two "
     "people finding the sentence hard -- it is two people holding different "
     "thresholds for the same word, and a threshold is something this session "
     "can actually set. The deliverable is a sentence completing 'a citance "
     "`supports` rather than `neutral_use` when ___' that both of you would have "
     "applied to every case below."),
    ("multidimensional", "contribution_realization",
     "`partial` was co-assigned twice out of seven uses -- better than round one, "
     "where it was never co-assigned at all, but still the weakest category here. "
     "Both of you gave `part` as the reason every single time and neither ever "
     "used `type_mismatch`. Decide whether `type_mismatch` describes anything "
     "real, or whether it should be cut."),
    ("multidimensional", "citation_function",
     "`compares_against` is the live boundary: A used it three times where B said "
     "`builds_on_finding`. Both readings are defensible for a citance that reports "
     "someone else's result next to your own. Settle which one wins."),
    ("multidimensional", "cited_contribution_type",
     "Where you both committed to a type you mostly agreed. The gap is in when to "
     "stop: B declined to call it more than twice as often as A. `unclear` is not "
     "a wrong answer here, but it needs to mean the same thing to both of you."),
    ("compact", "compact_intent",
     "Narrowest miss of the six. The disagreements are `background` against "
     "everything else, which is the boundary this pass shares with SciCite and "
     "ACL-ARC -- so how you draw it determines whether the comparison holds."),
    ("multidimensional", "centrality",
     "**Read this last and keep it short.** Disagreements here are symmetric and "
     "the split is near even, which is what it looks like when the citance simply "
     "does not carry the answer -- unlike polarity, there is no consistent "
     "direction to correct. The useful outcome is not a higher number. It is "
     "deciding whether a coder can tell bulk from core from the sentence alone, "
     "and if not, saying so in the paper."),
]


def git_show(repo: Path, ref: str, path: str) -> str:
    out = subprocess.run(["git", "-C", str(repo), "show", f"{ref}:{path}"],
                         capture_output=True, text=True)
    if out.returncode != 0:
        raise SystemExit(f"cannot read {ref}:{path} -- {out.stderr.strip()}")
    return out.stdout


def load_validator(repo: Path):
    """Reuse the coders' own validator so the worksheet cannot normalise a token
    differently from the tool they are told to run."""
    path = repo / "tools" / "validate_submission.py"
    spec = importlib.util.spec_from_file_location("validate_submission", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def normalise(value: str, column: str, validator) -> str:
    raw = (value or "").strip()
    if not raw:
        return BLANK
    fix, _ = validator.check_cell(raw, column)
    return fix if fix is not None else raw


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    repo = args.repo.resolve()
    validator = load_validator(repo)

    cache: dict[str, tuple[dict, dict]] = {}
    for pass_name, sources in SOURCES.items():
        loaded = {}
        for who, (ref, path) in sources.items():
            rows = list(csv.DictReader(io.StringIO(git_show(repo, ref, path))))
            loaded[who] = {r[KEY]: r for r in rows}
        cache[pass_name] = (loaded["A"], loaded["B"])

    lines = [
        "# Adjudication worksheet — round two (recalibration set)",
        "",
        "Only the rows where the two coders differ, ordered by what the session can",
        "fix rather than by how badly a dimension scored.",
        "",
        "**How to run the session.** Work the cases, not the counts. For each",
        "disagreement the question is not *who was right* but *what wording would have",
        "made both of you choose the same token*. Record the answer as a codebook",
        "anchor: \"edge 47 is `supports`\" is worth nothing next week; \"a citance",
        "`supports` rather than `neutral_use` when ___\" is the deliverable.",
        "",
        "No agreement statistics appear here, on purpose. The pair counts show which",
        "boundary keeps failing; a reliability figure would only tell you how to feel",
        "about it before you have read the cases.",
        "",
        "Both coders' original files are committed unchanged and stay that way.",
        "Adjudication produces a new file; it never overwrites a coder's judgement.",
        "",
        "Tokens below are shown in their codebook spelling (`neutral use` →",
        "`neutral_use`). Nothing else about a judgement was altered.",
        "",
    ]

    for pass_name, dim, rationale in DIMENSIONS:
        a, b = cache[pass_name]
        shared = [k for k in a if k in b]
        va = {k: normalise(a[k].get(dim, ""), dim, validator) for k in shared}
        vb = {k: normalise(b[k].get(dim, ""), dim, validator) for k in shared}
        diffs = [k for k in shared if va[k] != vb[k]]

        lines += [f"## {dim}", "", rationale, "",
                  f"**{len(diffs)} of {len(shared)} rows differ.**", ""]
        if not diffs:
            lines += ["No disagreements.", ""]
            continue

        # Directional, not sorted: for polarity the direction IS the finding.
        pairs = Counter((va[k], vb[k]) for k in diffs)
        recurring = pairs.most_common()
        lines += ["| A said | B said | n |", "|---|---|---|"]
        lines += [f"| `{x}` | `{y}` | {n} |" for (x, y), n in recurring]

        reverse_seen = any((y, x) in pairs for (x, y) in pairs)
        if not reverse_seen and len(diffs) > 2:
            lines += ["", "Every disagreement runs one direction and none runs the "
                          "other. This is an offset between two coders, not a hard "
                          "boundary in the material."]
        lines += ["", "Start with the most frequent pair — a boundary that fails "
                      "repeatedly is a wording defect, not a hard case.", ""]

        order = {pair: i for i, (pair, _) in enumerate(recurring)}
        diffs.sort(key=lambda k: order[(va[k], vb[k])])

        for k in diffs:
            ra, rb = a[k], b[k]
            context = (ra.get("citation_context") or "").strip().replace("\n", " ")
            if len(context) > CONTEXT_CHARS:
                context = context[:CONTEXT_CHARS] + "…"
            lines += [
                f"### `{k[:12]}` — A: `{va[k]}`  ·  B: `{vb[k]}`",
                "",
                f"**Cited:** {ra.get('cited_title', '').strip()}",
                "",
                f"**Citing:** {ra.get('citing_title', '').strip()}",
                "",
                f"> {context}",
                "",
            ]
            if dim == "contribution_realization":
                pa = (ra.get("partial_reason") or "").strip() or "—"
                pb = (rb.get("partial_reason") or "").strip() or "—"
                lines += [f"*partial_reason:* A `{pa}` · B `{pb}`", ""]
            flags = [w for w, r in (("A", ra), ("B", rb))
                     if (r.get("multi_reference") or "").strip()]
            if flags:
                lines += [f"*`multi_reference` ticked by:* {', '.join(flags)} — the "
                          f"citance names several papers, so decide the label for the "
                          f"one edge in front of you, not for the group.", ""]
            for who, row in (("A", ra), ("B", rb)):
                note = (row.get("annotator_note") or "").strip()
                if note:
                    lines += [f"*{who} noted:* {note}", ""]
            lines += ["**Agreed wording:**", "", "---", ""]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines), encoding="utf-8")
    total = sum(1 for line in lines if line.startswith("### `"))
    print(f"wrote {args.out} -- {total} disagreement(s) across "
          f"{len(DIMENSIONS)} dimensions")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
