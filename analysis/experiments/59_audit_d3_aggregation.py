#!/usr/bin/env python3
"""Verify D3 aggregation and paired intervals from saved OOF predictions.

Loads only data/metric functions from the original AST: no torch imports,
training, model calls, or changes to historical results.
"""
import ast
import csv
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "analysis/ml/11_d3_soft_labels.py"
SAVED = ROOT / "analysis/results/ml/d3_2026-10/main"
OUT = ROOT / "analysis/results/audits/2026-10-26_d3_aggregation.json"


def main():
    if OUT.exists():
        raise SystemExit(f"refusing to overwrite {OUT}")
    tree = ast.parse(SOURCE.read_text())
    functions = {"read", "crosswalk", "load_edges", "target", "ece", "brier"}
    assignments = {"ROOT", "D", "H", "CLASSES", "ARMS"}
    nodes = [n for n in tree.body if (
        isinstance(n, ast.FunctionDef) and n.name in functions
    ) or (
        isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id in assignments for t in n.targets)
    )]
    namespace = dict(__file__=str(SOURCE), Path=Path, np=np, csv=csv, Counter=Counter)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), "exec"), namespace)
    edges, excluded = namespace["load_edges"]()
    targets = np.stack([namespace["target"](e, "soft") for e in edges]).astype(np.float64)
    original = json.loads((SAVED / "d3_results.json").read_text())
    assert len(edges) == original["n_edges"] == 294
    assert dict(excluded) == original["excluded"]
    metrics = {"ece": namespace["ece"], "brier": namespace["brier"]}
    means, arms, hashes = {}, {}, {}
    for arm in namespace["ARMS"]:
        matrices = []
        for seed in original["seeds"]:
            path = SAVED / f"oof_{arm}_seed{seed}.npy"
            hashes[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
            matrix = np.load(path)
            assert matrix.shape == targets.shape
            matrices.append(matrix)
        means[arm] = np.mean(matrices, axis=0)
        arms[arm] = {}
        for name, metric in metrics.items():
            per_seed = float(np.mean([metric(matrix, targets) for matrix in matrices]))
            ensemble = metric(means[arm], targets)
            assert round(per_seed, 4) == original["arms"][arm][name + "_seed_mean"]
            assert round(ensemble, 4) == original["arms"][arm][name + "_seed_avg_probs"]
            arms[arm][name] = {"mean_of_seed_scores": per_seed, "score_of_seed_averaged_probabilities": ensemble}
    groups = defaultdict(list)
    for i, edge in enumerate(edges):
        groups[edge["comp"]].append(i)
    keys = sorted(groups)
    rng = random.Random(20261003)
    draws = {hard: {metric: [] for metric in metrics} for hard in ("hard_kevin", "hard_canonical")}
    for _ in range(10000):
        idx = [i for component in rng.choices(keys, k=len(keys)) for i in groups[component]]
        for name, metric in metrics.items():
            soft = metric(means["soft"][idx], targets[idx])
            for hard in draws:
                draws[hard][name].append(soft - metric(means[hard][idx], targets[idx]))
    contrasts = {}
    for hard in draws:
        contrasts[hard] = {}
        for name in metrics:
            difference = arms["soft"][name]["score_of_seed_averaged_probabilities"] - arms[hard][name]["score_of_seed_averaged_probabilities"]
            values = sorted(draws[hard][name])
            ci = [round(values[250], 4), round(values[9749], 4)]
            assert round(difference, 4) == original["soft_minus_hard"][hard][name]["diff"]
            assert ci == original["soft_minus_hard"][hard][name]["ci95"]
            contrasts[hard][name] = {"difference": difference, "ci95": ci}
    result = {
        "status": "verified: aggregation distinction, not an erroneous paired subtraction",
        "n": len(edges), "arms": arms, "paired_ensemble_contrasts": contrasts,
        "bootstrap": {"unit": "citation graph component", "resamples": 10000, "seed": 20261003},
        "verdict": original["verdict_vs_registered_hard_arm"],
        "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "prediction_sha256": hashes,
        "interpretation": "Mean per-seed scores are descriptive. Paired differences and intervals score seed-averaged probabilities. Joint improvement remains unestablished.",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "contrasts": contrasts, "verdict": result["verdict"]}, indent=2))


if __name__ == "__main__":
    main()
