#!/usr/bin/env python3
"""Generate one written adjudication questionnaire per coder.

Async substitute for the live session. Each coder sees only their own labels --
the other coder's judgements never appear -- so independence is preserved and
the two documents can be sent simultaneously.

The questions are forced-choice on purpose. The central finding from the
calibration is that one coder gates ("did they use it at all?") and the other
grades a scale ("how much of it?"), and neither can report that in answer to an
open question, because neither knows they are doing it. Asking them to choose
between two stated questions recovers what a live follow-up would have.

Usage:
    python3 analysis/measurement/26_coder_questionnaire.py \
        --repo annotations/coders/round_one --out-dir local/
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

MULTI = "calibration_multidimensional_blinded"
COMPACT = "calibration_compact_blinded"
KEY = "edge_id"
CONTEXT_CHARS = 450

FUNCTIONS = ["motivates_background", "builds_on_finding", "compares_against",
             "applies_framework", "reuses_artifact", "uses_method", "other", "unclear"]


def read(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def pick(repo: Path, coder: str, stem: str) -> Path:
    folder = repo / "coders" / coder
    norm = folder / f"{stem}_normalized.csv"
    return norm if norm.exists() else folder / f"{stem}.csv"


def case(row: dict, dim: str) -> list[str]:
    context = (row.get("citation_context") or "").strip().replace("\n", " ")
    if len(context) > CONTEXT_CHARS:
        context = context[:CONTEXT_CHARS] + "…"
    out = [f"**Cited:** {row.get('cited_title', '').strip()}", "",
           f"> {context}", "",
           f"**You labelled this `{row[dim]}`.**", ""]
    note = (row.get("annotator_note") or "").strip()
    if note:
        out += [f"*Your note at the time:* {note}", ""]
    return out


def build(name: str, multi: list[dict], compact: list[dict]) -> str:
    L = [
        f"# Calibration follow-up — {name}",
        "",
        "Ten to fifteen minutes. Answer in this document and send it back to me only.",
        "",
        "**This is not a test and there are no wrong answers here.** You and the other",
        "coder disagreed on about a third of the labels, which tells me the codebook was",
        "underspecified — that is exactly what the calibration round was for. I am trying",
        "to find out *what the instructions failed to say*, and only you can tell me that.",
        "",
        "Everything below is your own work. You will not see anyone else's labels.",
        "",
        "---",
        "",
        "## 1. When you chose `partial`",
        "",
        "Below are rows where you used `partial` for **contribution realization**.",
        "",
    ]

    partials = [r for r in multi if r.get("contribution_realization") == "partial"][:3]
    if not partials:
        partials = [r for r in multi if r.get("contribution_realization") == "no_background"][:3]
    for i, r in enumerate(partials, 1):
        L += [f"### Case {i}", ""] + case(r, "contribution_realization")

    L += [
        "**Pick one. Which question were you actually answering?**",
        "",
        "- [ ] **(a)** Did the citing paper use the cited contribution *at all*, or was it",
        "  only mentioned as background?",
        "- [ ] **(b)** *How much* of the cited contribution did the citing paper use — all",
        "  of it, or part of it?",
        "- [ ] **(c)** Something else — please describe:",
        "",
        "If your answer differs between the three cases, say so. That is useful.",
        "",
        "---",
        "",
        "## 2. Citation function — which values do you trust?",
        "",
        "Tick every value you have **never felt confident** assigning. Be honest; a value",
        "nobody can apply is a defect in my scheme, not in your coding.",
        "",
    ]
    L += [f"- [ ] `{f}`" for f in FUNCTIONS]
    L += ["", "For any you ticked: what makes it hard?", "", "---", "",
          "## 3. When you chose `unclear`", ""]

    unclear = [r for r in multi
               if "unclear" in (r.get("cited_contribution_type", ""),
                                r.get("citation_function", ""),
                                r.get("contribution_realization", ""))][:3]
    if unclear:
        for i, r in enumerate(unclear, 1):
            dim = next(d for d in ("cited_contribution_type", "citation_function",
                                   "contribution_realization") if r.get(d) == "unclear")
            L += [f"### Case {i} — dimension `{dim}`", ""] + case(r, dim)
    else:
        L += ["*You did not use `unclear` on the multidimensional pass. Answer for the "
              "compact pass instead, or skip.*", ""]

    L += [
        "**Pick one per case.** When you wrote `unclear`, did you mean:",
        "",
        "- [ ] **(a)** I could not tell from this citance — there was not enough evidence.",
        "- [ ] **(b)** The evidence was clear enough, but none of the categories fitted.",
        "",
        "These are different problems and I currently have one token for both.",
        "",
        "---",
        "",
        "## 4. One citance, several cited papers",
        "",
        "Sometimes a single sentence cites several papers at once — *\"prior work has shown",
        "X [12, 13, 14, 15]\"* — and you are asked about only one of them.",
        "",
        "- Roughly how many of your 40 rows were like this? ____",
        "- What did you do when it happened?",
        "",
        "- Did you ever get a citation that was **just a reference tag with no sentence",
        "  around it**? Roughly how many? ____",
        "",
        "---",
        "",
        "## 5. Keyword cues",
        "",
        "Some coders decide partly from specific phrases — *\"as previously described\"*,",
        "*\"also been linked\"*, *\"we observe\"*, *\"others found\"*.",
        "",
        "- [ ] I do this deliberately",
        "- [ ] I do it sometimes without thinking about it",
        "- [ ] I do not do this",
        "",
        "If you do: which phrases, and what do they tell you?",
        "",
        "---",
        "",
        "## 6. What the brief did not tell you",
        "",
        "What did you have to guess at? What would you have wanted to know before you",
        "started? Anything that slowed you down or that you are still unsure about.",
        "",
        "",
        "",
        "---",
        "",
        "*Send this back to me only — not to the other coder. There will be a second",
        "round on 40 new edges once I have rewritten the codebook, and it has to stay",
        "independent to be worth anything.*",
        "",
    ]
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--coders", nargs="+", default=["annotator_a", "annotator_b"])
    ap.add_argument("--names", nargs="+", default=None,
                    help="display names, same order as --coders")
    args = ap.parse_args()

    names = args.names or args.coders
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for coder, name in zip(args.coders, names):
        multi = read(pick(args.repo, coder, MULTI))
        compact = read(pick(args.repo, coder, COMPACT))
        out = args.out_dir / f"calibration_followup_{coder}.md"
        out.write_text(build(name, multi, compact), encoding="utf-8")
        print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
