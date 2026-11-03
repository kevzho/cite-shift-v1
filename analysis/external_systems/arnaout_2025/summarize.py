#!/usr/bin/env python3
"""Tabulate per-edge outputs of run_arnaout.py against the project's canonical labels."""
import json
import sys
from collections import Counter

path = sys.argv[1] if len(sys.argv) > 1 else "outputs/preds_ollama_qwen2.5_7b-instruct.jsonl"
rows = [json.loads(l) for l in open(path)]
print(f"file: {path}\nsystem_variant: {rows[0]['system_variant']}  model: {rows[0]['model']}\n")
print(f"{'edge_id':22} {'corpus':9} {'canon_function':26} {'realiz':8} {'b/c':5} {'runs':38} final")
for r in rows:
    runs = ",".join((x or "None")[:6] for x in r["run_labels"])
    print(f"{r['edge_id']:22} {r['corpus']:9} {r['canonical_function']:26} "
          f"{r['canonical_realization']:8} {r['bulk_or_core']:5} {runs:38} {r['impact_revealing']}")
print()
print("final label counts:", Counter(r["impact_revealing"] for r in rows))
print("unanimous 3/3:", sum(len(set(r["run_labels"])) == 1 for r in rows), "/", len(rows))
print("\nfinal x canonical_realization:")
ct = Counter((r["canonical_realization"], r["impact_revealing"]) for r in rows)
for k in sorted(ct):
    print("  ", k, ct[k])
print("\nfinal x corpus:")
ct = Counter((r["corpus"], r["impact_revealing"]) for r in rows)
for k in sorted(ct):
    print("  ", k, ct[k])
