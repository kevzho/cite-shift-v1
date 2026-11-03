"""Determinism check between two independent fetches (run tags A and B) and two extractions of one snapshot.

Compares system outputs with each other only; never with any human label.
Usage: python 03_compare_runs.py runA runB
"""
import csv
import hashlib
import importlib
import json
import sys

from s2_client import OUT

IGNORE = {"retrieved_utc"}


def load(tag):
    return {r["edge_id"]: r for r in csv.DictReader(open(OUT / "outputs" / tag / "s2_edge_labels.csv", encoding="utf-8"))}


def canon(d):
    """S2 serialises the `intents` set in arbitrary order; sort it (and each context's intents)."""
    d = dict(d)
    if d.get("intents") is not None:
        d["intents"] = sorted(d["intents"])
    if d.get("contextsWithIntent") is not None:
        d["contextsWithIntent"] = [dict(c, intents=sorted(c.get("intents") or [])) for c in d["contextsWithIntent"]]
    return d


def data_digest(tag, canonical):
    """sha256 over every returned citation record (sorted), excluding request metadata."""
    snap = OUT / "snapshots" / tag
    recs = []
    for p in sorted((snap / "references").glob("*.json")) + sorted((snap / "citations").glob("*.json")):
        for page in json.load(open(p))["pages"]:
            recs += [json.dumps(canon(d) if canonical else d, sort_keys=True) for d in (page["body"].get("data") or [])]
    return hashlib.sha256("\n".join(sorted(recs)).encode()).hexdigest(), len(recs)


def main(a, b):
    A, B = load(a), load(b)
    diffs = [(e, k, A[e][k], B[e][k]) for e in A for k in A[e] if k not in IGNORE and A[e][k] != B[e][k]]
    da, db = data_digest(a, False), data_digest(b, False)
    ca, cb = data_digest(a, True), data_digest(b, True)
    # re-extract snapshot A a second time and compare byte-for-byte
    before = (OUT / "outputs" / a / "s2_edge_labels.csv").read_bytes()
    importlib.import_module("02_extract").main(a)
    after = (OUT / "outputs" / a / "s2_edge_labels.csv").read_bytes()
    res = {"runs": [a, b], "edges": len(A), "field_diffs_between_fetches": diffs,
           "identical_outputs_between_fetches": not diffs,
           "raw_record_digest": {a: da, b: db}, "raw_records_identical": da == db,
           "canonical_record_digest": {a: ca, b: cb},
           "canonical_records_identical_(intent_order_normalised)": ca == cb,
           "re_extraction_byte_identical": before == after}
    (OUT / "outputs" / "determinism.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
