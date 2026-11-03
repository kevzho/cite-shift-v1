#!/usr/bin/env python3
"""Strictly evaluate a complete direct-label model run against Yiren's coding."""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "yiren_artifacts/HCI contribution coding - hci coding (n=1040).csv"
OUT = ROOT / "analysis/results/experiments"
FUNCTIONS = ["motivates_background", "builds_on_finding", "uses_method", "compares_against", "applies_framework", "reuses_artifact", "other"]
REALIZATION = ["yes", "partial", "no_background"]
REQUIRED = {"request_id", "cited_id", "citing_id", "raw_response", "model_id", "response_date", "temperature", "prompt_hash", "parse_status", "citance_function", "realizes_contribution"}


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def load_jsonl(path: Path) -> list[dict[str, object]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def text(value: object) -> str:
    return " ".join(str(value or "").strip().lower().split())


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)


def metrics(labels: list[str], pairs: list[tuple[str, str]], prediction_name: str, human_name: str) -> list[dict[str, object]]:
    result = []
    for label in labels:
        tp = sum(p == label and h == label for p, h in pairs)
        fp = sum(p == label and h != label for p, h in pairs)
        fn = sum(p != label and h == label for p, h in pairs)
        precision = tp / (tp + fp) if tp + fp else None
        recall = tp / (tp + fn) if tp + fn else None
        f1 = 2 * precision * recall / (precision + recall) if precision is not None and recall is not None and precision + recall else None
        result.append({"label": label, "support_human": sum(h == label for _, h in pairs), "predicted_n": sum(p == label for p, _ in pairs), "true_positive": tp, "precision": round(precision, 6) if precision is not None else "", "recall": round(recall, 6) if recall is not None else "", "f1": round(f1, 6) if f1 is not None else "", "prediction_field": prediction_name, "human_field": human_name})
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--responses", type=Path, required=True)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output-dir", type=Path, default=OUT)
    args = parser.parse_args()
    if args.source.resolve() != SOURCE.resolve(): raise SystemExit("Refusing non-canonical human source.")
    human_rows = load_csv(args.source); responses = load_jsonl(args.responses)
    human = {(text(r["cited_id"]), text(r["citing_id"])): r for r in human_rows}
    pairs = [(text(r.get("cited_id")), text(r.get("citing_id"))) for r in responses]
    errors = []
    if len(human) != 1040: errors.append(f"canonical source contains {len(human)} unique pairs, not 1040")
    if len(responses) != 1040: errors.append(f"run contains {len(responses)} responses, not 1040")
    if len(set(pairs)) != 1040: errors.append(f"run contains {len(set(pairs))} unique citation pairs, not 1040")
    if set(pairs) != set(human): errors.append("response citation pairs do not exactly equal canonical source pairs")
    for index, row in enumerate(responses, 1):
        missing = REQUIRED - set(row)
        if missing: errors.append(f"response {index} missing fields {sorted(missing)}"); continue
        if not str(row["raw_response"]).strip(): errors.append(f"response {index} has blank raw_response")
        if not str(row["model_id"]).strip() or not str(row["response_date"]).strip() or not str(row["prompt_hash"]).strip(): errors.append(f"response {index} lacks required provenance")
        if row["parse_status"] != "parsed": errors.append(f"response {index} parse_status is {row['parse_status']!r}")
        if text(row["citance_function"]) not in FUNCTIONS or text(row["realizes_contribution"]) not in REALIZATION: errors.append(f"response {index} has an invalid direct label")
    if errors: raise SystemExit("Evaluation refused:\n- " + "\n- ".join(errors[:30]))
    function_pairs = [(text(r["citance_function"]), text(human[(text(r["cited_id"]), text(r["citing_id"]))]["citance_function"])) for r in responses]
    realization_pairs = [(text(r["realizes_contribution"]), text(human[(text(r["cited_id"]), text(r["citing_id"]))]["realizes_contribution"])) for r in responses]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = [
        {"task": "citance_function", "n": 1040, "exact_agreement": round(sum(p == h for p, h in function_pairs) / 1040, 6)},
        {"task": "realizes_contribution", "n": 1040, "exact_agreement": round(sum(p == h for p, h in realization_pairs) / 1040, 6)},
    ]
    write_csv(args.output_dir / "yiren_direct_label_llm_evaluation_summary.csv", summary, list(summary[0]))
    for name, labels, pairs, pred, hum in [("function", FUNCTIONS, function_pairs, "citance_function", "citance_function"), ("realization", REALIZATION, realization_pairs, "realizes_contribution", "realizes_contribution")]:
        counts = Counter(pairs)
        write_csv(args.output_dir / f"yiren_direct_label_llm_{name}_confusion.csv", [{"predicted": p, "human_yiren": h, "n": counts[(p, h)]} for p in labels for h in labels], ["predicted", "human_yiren", "n"])
        write_csv(args.output_dir / f"yiren_direct_label_llm_{name}_metrics.csv", metrics(labels, pairs, pred, hum), ["label", "support_human", "predicted_n", "true_positive", "precision", "recall", "f1", "prediction_field", "human_field"])
    write_csv(args.output_dir / "yiren_direct_label_llm_coverage_failures.csv", [{"status": "complete_valid_run", "response_records": 1040, "unique_pairs": 1040, "blank_raw_responses": 0, "parse_failures": 0}], ["status", "response_records", "unique_pairs", "blank_raw_responses", "parse_failures"])
    report = "# Yiren Direct-Label LLM Evaluation\n\nThis report measures agreement between the model and Yiren's original HCI coding scheme across 1,040 cited–citing citation records. It does not establish universal correctness of either coding decision.\n\n" + "\n".join(f"- {r['task']}: exact agreement {r['exact_agreement']:.1%} ({r['n']} records)" for r in summary) + "\n\nAll prompts excluded human labels, rationale, Tier-2 subtype, historical LLM labels, S2 labels, and prior prediction fields. See the CSV tables for confusion matrices, per-label precision/recall/F1, and coverage.\n"
    (args.output_dir / "yiren_direct_label_llm_report.md").write_text(report, encoding="utf-8")
    print("Evaluated complete direct-label run: 1,040 records.")


if __name__ == "__main__":
    main()
