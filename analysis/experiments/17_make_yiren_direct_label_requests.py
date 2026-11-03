#!/usr/bin/env python3
"""Create non-leaky direct-label requests from Yiren's canonical HCI sheet."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "yiren_artifacts/HCI contribution coding - hci coding (n=1040).csv"
OUT = ROOT / "analysis/results/experiments"
MODEL = "openai:openai/gpt-5.6-sol"
FUNCTIONS = ["motivates_background", "builds_on_finding", "uses_method", "compares_against", "applies_framework", "reuses_artifact", "other"]
REALIZATION = ["yes", "partial", "no_background"]


def load(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def required(value: str | None, name: str, pair: tuple[str, str]) -> str:
    value = (value or "").strip()
    if not value:
        raise ValueError(f"Blank {name} for citation pair {pair!r}")
    return value


def make_prompt(row: dict[str, str]) -> str:
    """Use only the four permitted evidence fields and task instructions."""
    return "\n".join([
        "Classify this one cited–citing citation record using only the supplied evidence.",
        "Return exactly one JSON object and no Markdown or explanation.",
        "Required JSON keys: citance_function, realizes_contribution.",
        f"citance_function must be one of: {', '.join(FUNCTIONS)}.",
        f"realizes_contribution must be one of: {', '.join(REALIZATION)}.",
        "Definitions: motivates_background = cites for motivation/context; builds_on_finding = extends or relies on a prior finding; uses_method = uses a prior method; compares_against = contrasts or evaluates against prior work; applies_framework = uses a prior conceptual/theoretical framework; reuses_artifact = reuses a prior system, tool, dataset, or other artifact; other = none of these.",
        "realizes_contribution: yes = the citation substantively uses or realizes the cited contribution; partial = limited/ambiguous realization; no_background = background-only or does not realize it.",
        "",
        f"Cited-paper title: {row['cited_title']}",
        f"Citing-paper title: {row['citing_title']}",
        f"Section: {row['section']}",
        f"Citing sentence/context: {row['citing_sentence(s)']}",
    ])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, default=OUT / "yiren_direct_label_llm_requests.jsonl")
    parser.add_argument("--manifest", type=Path, default=OUT / "yiren_direct_label_llm_manifest.csv")
    parser.add_argument("--model-id", default=MODEL)
    args = parser.parse_args()
    if args.source.resolve() != SOURCE.resolve():
        raise SystemExit(f"Refusing non-canonical source: expected {SOURCE}")
    rows = load(args.source)
    pairs = [(required(r.get("cited_id"), "cited_id", ("?", "?")), required(r.get("citing_id"), "citing_id", ("?", "?"))) for r in rows]
    if len(rows) != 1040 or len(set(pairs)) != 1040:
        raise SystemExit(f"Expected exactly 1,040 unique citation pairs; found rows={len(rows)}, pairs={len(set(pairs))}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    created = datetime.now(timezone.utc).isoformat()
    manifest: list[dict[str, object]] = []
    with args.output.open("w", encoding="utf-8") as handle:
        for index, (row, (cited_id, citing_id)) in enumerate(zip(rows, pairs), start=1):
            for field in ("cited_title", "citing_title", "section", "citing_sentence(s)"):
                required(row.get(field), field, (cited_id, citing_id))
            prompt = make_prompt(row)
            request = {
                "request_id": f"yiren_direct_label_llm_{index:04d}",
                "cited_id": cited_id,
                "citing_id": citing_id,
                "model_id": args.model_id,
                "temperature": 0,
                "seed": None,
                "prompt": prompt,
                "prompt_hash": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                "response_schema": {"citance_function": FUNCTIONS, "realizes_contribution": REALIZATION},
            }
            handle.write(json.dumps(request, ensure_ascii=False) + "\n")
            manifest.append({k: request[k] for k in ("request_id", "cited_id", "citing_id", "model_id", "temperature", "seed", "prompt_hash")} | {"prepared_at": created, "generation_status": "prepared_dry_run_no_provider_call"})
    with args.manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest[0]))
        writer.writeheader(); writer.writerows(manifest)
    print(f"Dry run complete: wrote {len(rows)} non-leaky requests from Yiren's canonical source.")


if __name__ == "__main__":
    main()
