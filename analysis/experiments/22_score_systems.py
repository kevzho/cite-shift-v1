#!/usr/bin/env python3
"""Score system predictions against every human reference, separately.

References (never pooled)
-------------------------
  canonical   analysis/derived/*_human_reviewed.csv (2,937 edges; Yiren Liu)
  v1          annotations/coders/expansion_2026-09-26/annotator_kz_expansion_multidimensional.csv
              (Kevin, codebook v1, 500 edges; locked 26 Sep, sha256 a1234886...)
  v2          annotations/generated/v2_coding_2026-10-02/ (Kevin, codebook v2):
              B Pass 1 + Pass 2 (300 edges) and A Pass 2 (the v1 500). Coding
              files are joined to their *_key.csv on `item`. Only coded rows
              count. A's citation function is v1 data and is therefore scored
              under `v1`, never under `v2`.

Field eligibility (which reference can score which field)
----------------------------------------------------------
  citation_function        canonical, v1, v2   all mapped to v2 names
  cited_contribution_type  canonical, v1, v2   label_crosswalk.csv renames
  realization              canonical, v1       yes / partial / no (v1 no_background -> no)
  credits_contribution     v2
  uptake_mode              v2
  uptake_degree            v2
  claim_dependent          v2
No v2 realization field is mapped onto canonical or v1 realization: codebook v2
§10 makes that relationship an empirical question, not a crosswalk.

Predictions
-----------
Either a long CSV with columns `edge_id, field, label` (optional `system`), or
a frozen-prediction file written by `06_train_encoder.py --predict-frozen`
(columns `edge_id_v1, task, y_pred`), detected automatically. Encoder tasks map
to fields as function -> citation_function, realization -> realization;
`contribution` predicts canonical `cited_contribution`, which has no crosswalk
to `cited_contribution_type`, and is refused as ineligible. edge ids may use
either hashing scheme (02_make_splits.py); both resolve.

Metrics, per field x system x reference
---------------------------------------
macro-F1 over classes with reference support >= 5 (withheld classes named),
accuracy, majority-class rate, Cohen's kappa. 95% percentile intervals from a
cluster bootstrap over citation-graph connected components (`component_key` in
analysis/ml/splits/), 10,000 resamples by default; the reportable class set is
fixed on the full sample.

Guard. Scoring against the real v2 record requires
private/preregistration/REGISTERED.txt (pre-register before any system meets a
v2 label). A `--v2-dir` outside annotations/ (synthetic test data) is exempt.

Usage:
    python3 analysis/experiments/22_score_systems.py PRED.csv [PRED2.csv ...] \\
        --refs canonical v1 --out analysis/results/scoring/run_name.json
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DERIVED = {c: ROOT / f"analysis/derived/{c}_contribution_coding_human_reviewed.csv"
           for c in ("hci", "nature_oa")}
SPLITS = {c: ROOT / f"analysis/ml/splits/{c}_splits.csv" for c in DERIVED}
V1_FILE = ROOT / "annotations/coders/expansion_2026-09-26/annotator_kz_expansion_multidimensional.csv"
V2_DIR = ROOT / "annotations/generated/v2_coding_2026-10-02"
CROSSWALK = ROOT / "annotations/label_crosswalk.csv"
REGISTERED = ROOT / "private/preregistration/REGISTERED.txt"

MIN_SUPPORT = 5

ELIGIBLE = {
    "citation_function": {"canonical", "v1", "v2"},
    "cited_contribution_type": {"canonical", "v1", "v2"},
    "realization": {"canonical", "v1"},
    "credits_contribution": {"v2"},
    "uptake_mode": {"v2"},
    "uptake_degree": {"v2"},
    "claim_dependent": {"v2"},
}
TASK_FIELD = {"function": "citation_function", "realization": "realization"}
TASK_INELIGIBLE = {"contribution": "predicts canonical `cited_contribution`; no crosswalk "
                                   "to `cited_contribution_type`",
                   "centrality": "Pass 3 dropped; no reference",
                   "polarity": "Pass 3 dropped; no reference"}

V2_PASS1 = ("cited_contribution_type", "citation_function")
V2_PASS2 = ("credits_contribution", "uptake_mode", "uptake_degree", "claim_dependent")


# --------------------------------------------------------------------------
# io and ids

def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def id_v1(corpus: str, row: dict) -> str:
    key = "\x1f".join((corpus, (row.get("cited_id") or "").strip(),
                       (row.get("citing_id") or "").strip()))
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:20]


def id_v2(corpus: str, row: dict) -> str:
    key = f"{corpus}|{(row.get('cited_id') or '').strip()}|{(row.get('citing_id') or '').strip()}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:20]


def components() -> tuple[dict[str, str], dict[str, str]]:
    """(any edge id -> canonical v1 id, v1 id -> cluster key)."""
    alias, cluster = {}, {}
    for corpus, path in SPLITS.items():
        for r in read_csv(path):
            alias[r["edge_id_v1"]] = r["edge_id_v1"]
            alias[r["edge_id_v2"]] = r["edge_id_v1"]
            cluster[r["edge_id_v1"]] = f"{corpus}:{r['component_key']}"
    return alias, cluster


# --------------------------------------------------------------------------
# vocabularies: everything is scored in v2 names

def crosswalk() -> dict[tuple[str, str], str]:
    """(dimension, canonical value) -> codebook label, mechanical rows only."""
    out = {}
    for r in read_csv(CROSSWALK):
        if r["status"] == "mechanical" and r["canonical_value"]:
            out[(r["dimension"], r["canonical_value"])] = r["codebook_label"]
    return out


def norm(field: str, value: str, cw: dict) -> str | None:
    v = (value or "").strip()
    if not v:
        return None
    if field == "citation_function":
        v = cw.get(("citance_function", v), v)
        return "background" if v == "motivates_background" else v
    if field == "cited_contribution_type":
        return cw.get(("research_contribution", v), v)
    if field == "realization":
        return {"no_background": "no"}.get(v, v)
    return v


# --------------------------------------------------------------------------
# references

def ref_canonical(cw) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = defaultdict(dict)
    cols = {"citation_function": "citance_function",
            "cited_contribution_type": "research_contribution",
            "realization": "realizes_contribution"}
    for corpus, path in DERIVED.items():
        for row in read_csv(path):
            eid = id_v1(corpus, row)
            for field, col in cols.items():
                v = norm(field, row.get(col, ""), cw)
                if v:
                    out[field][eid] = v
    return out


def ref_v1(cw) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = defaultdict(dict)
    cols = {"citation_function": "citation_function",
            "cited_contribution_type": "cited_contribution_type",
            "realization": "contribution_realization"}
    for row in read_csv(V1_FILE):
        for field, col in cols.items():
            v = norm(field, row.get(col, ""), cw)
            if v:
                out[field][row["edge_id"].strip()] = v
    return out


def ref_v2(v2_dir: Path, cw) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = defaultdict(dict)

    def coded(stem: str, fields: tuple[str, ...]) -> None:
        sheet, key = v2_dir / f"{stem}.csv", v2_dir / f"{stem}_key.csv"
        if not sheet.exists():
            return
        edge = {k["item"]: k["edge_id"] for k in read_csv(key)}
        for row in read_csv(sheet):
            for field in fields:
                v = norm(field, row.get(field, ""), cw)
                if v:
                    out[field][edge[row["item"]]] = v

    for stem in ("B_stratum", "B_rest"):
        coded(f"{stem}_pass1", V2_PASS1)
        coded(f"{stem}_pass2", V2_PASS2)
    coded("A_pass2", V2_PASS2)
    return out


# --------------------------------------------------------------------------
# predictions

def predictions(paths: list[Path], alias: dict, cw: dict) -> tuple[dict, list[str]]:
    """{(system, field): {v1 edge id: label}}, plus notes on what was refused.

    Prediction labels go through the same norm() as references, so a system that emits canonical
    vocabulary (e.g. the encoder's `background_bulk`) is compared in v2 names."""
    out: dict[tuple[str, str], dict[str, str]] = defaultdict(dict)
    notes = []
    for path in paths:
        rows = read_csv(path)
        if not rows:
            continue
        if "y_pred" in rows[0] and "task" in rows[0]:          # 06 --predict-frozen
            task = rows[0]["task"]
            if task in TASK_INELIGIBLE:
                notes.append(f"{path.name}: refused, {TASK_INELIGIBLE[task]}")
                continue
            field, system = TASK_FIELD[task], path.stem
            for r in rows:
                v = norm(field, r["y_pred"], cw)
                if v:
                    out[(system, field)][alias.get(r["edge_id_v1"], r["edge_id_v1"])] = v
        else:
            for r in rows:
                system = r.get("system") or path.stem
                eid = alias.get(r["edge_id"].strip(), r["edge_id"].strip())
                v = norm(r["field"], r["label"], cw)
                if v:
                    out[(system, r["field"])][eid] = v
    return out, notes


# --------------------------------------------------------------------------
# metrics

def metrics_from_confusion(M: np.ndarray, reportable: np.ndarray) -> tuple[float, float, float]:
    """macro-F1 over reportable classes, accuracy, Cohen's kappa. M[..., true, pred]."""
    n = M.sum(axis=(-2, -1))
    tp = np.diagonal(M, axis1=-2, axis2=-1)
    support, predicted = M.sum(axis=-1), M.sum(axis=-2)
    with np.errstate(divide="ignore", invalid="ignore"):
        f1 = np.where(support + predicted > 0, 2 * tp / (support + predicted), 0.0)
        acc = tp.sum(axis=-1) / n
        pe = (support * predicted).sum(axis=-1) / (n * n)
        kappa = np.where(pe < 1, (acc - pe) / (1 - pe), np.nan)
    macro = f1[..., reportable].mean(axis=-1) if reportable.any() else np.full(n.shape, np.nan)
    return macro, acc, kappa


def score(pred: dict, ref: dict, cluster: dict, cw_field: str, cw, n_boot: int,
          seed: int) -> dict:
    shared = sorted(e for e in pred if e in ref)
    if not shared:
        return {"n": 0, "note": "no overlap between predictions and reference"}
    y_true = [ref[e] for e in shared]
    y_pred = [norm(cw_field, pred[e], cw) or "" for e in shared]
    labels = sorted(set(y_true) | set(y_pred))
    li = {l: i for i, l in enumerate(labels)}
    support = Counter(y_true)
    reportable = np.array([support[l] >= MIN_SUPPORT for l in labels])

    clusters = sorted({cluster.get(e, f"singleton:{e}") for e in shared})
    ci = {c: i for i, c in enumerate(clusters)}
    K = len(labels)
    per = np.zeros((len(clusters), K, K))
    for e, t, p in zip(shared, y_true, y_pred):
        per[ci[cluster.get(e, f"singleton:{e}")], li[t], li[p]] += 1

    full = per.sum(axis=0)
    macro, acc, kappa = metrics_from_confusion(full, reportable)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(clusters), size=(n_boot, len(clusters)))
    weights = np.stack([np.bincount(d, minlength=len(clusters)) for d in draws])
    boot = np.tensordot(weights, per, axes=(1, 0))
    b_macro, b_acc, b_kappa = metrics_from_confusion(boot, reportable)

    def ci95(x):
        x = x[~np.isnan(x)]
        return [round(float(np.percentile(x, 2.5)), 4),
                round(float(np.percentile(x, 97.5)), 4)] if len(x) else None

    majority = max(support, key=support.get)
    return {
        "n": len(shared), "n_clusters": len(clusters),
        "n_pred_without_ref": len(pred) - len(shared),
        "macro_f1": round(float(macro), 4) if not np.isnan(macro) else None,
        "macro_f1_ci95": ci95(b_macro),
        "accuracy": round(float(acc), 4), "accuracy_ci95": ci95(b_acc),
        "kappa": round(float(kappa), 4) if not np.isnan(kappa) else None,
        "kappa_ci95": ci95(b_kappa),
        "majority_class": majority,
        "majority_class_rate": round(support[majority] / len(shared), 4),
        "reportable_classes": [l for l, r in zip(labels, reportable) if r],
        "withheld_classes": [l for l, r in zip(labels, reportable) if not r],
        "withheld_reason": f"reference support < {MIN_SUPPORT}",
        "reference_distribution": dict(sorted(support.items())),
        "prediction_distribution": dict(sorted(Counter(y_pred).items())),
    }


# --------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("predictions", type=Path, nargs="+")
    ap.add_argument("--refs", nargs="+", default=["canonical", "v1"],
                    choices=["canonical", "v1", "v2"])
    ap.add_argument("--v2-dir", type=Path, default=V2_DIR)
    ap.add_argument("--n-boot", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=20261003)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    if args.out.exists():
        raise SystemExit(f"refusing to overwrite {args.out}")
    v2_dir = args.v2_dir.resolve()
    if "v2" in args.refs and (ROOT / "annotations") in v2_dir.parents and not REGISTERED.exists():
        raise SystemExit("v2 reference refused: the pre-registration is not registered "
                         f"({REGISTERED.relative_to(ROOT)} is missing)")

    cw = crosswalk()
    alias, cluster = components()
    loaders = {"canonical": lambda: ref_canonical(cw), "v1": lambda: ref_v1(cw),
               "v2": lambda: ref_v2(v2_dir, cw)}
    refs = {name: loaders[name]() for name in args.refs}
    preds, notes = predictions(args.predictions, alias, cw)

    results = []
    for (system, field), pred in sorted(preds.items()):
        if field not in ELIGIBLE:
            notes.append(f"{system}: field {field!r} unknown; skipped")
            continue
        for ref_name in args.refs:
            if ref_name not in ELIGIBLE[field]:
                results.append({"system": system, "field": field, "reference": ref_name,
                                "ineligible": "no such field in this reference"})
                continue
            res = score(pred, refs[ref_name].get(field, {}), cluster, field, cw,
                        args.n_boot, args.seed)
            results.append({"system": system, "field": field, "reference": ref_name, **res})

    header = f"{'system':<34}{'field':<24}{'ref':<10}{'n':>5}{'macroF1':>9}  {'95% CI':<16}{'kappa':>7}{'maj':>7}"
    print(header)
    print("-" * len(header))
    for r in results:
        if "ineligible" in r or not r.get("n"):
            continue
        ci = r["macro_f1_ci95"]
        mf = r["macro_f1"]
        print(f"{r['system'][:33]:<34}{r['field']:<24}{r['reference']:<10}{r['n']:>5}"
              f"{(f'{mf:.4f}' if mf is not None else 'n/a'):>9}  "
              f"{(f'[{ci[0]:.3f}, {ci[1]:.3f}]' if ci else '-'):<16}"
              f"{(r['kappa'] if r['kappa'] is not None else float('nan')):>7.3f}"
              f"{r['majority_class_rate']:>7.3f}")
    for n in notes:
        print("note:", n)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "script": "analysis/experiments/22_score_systems.py",
        "predictions": [str(p) for p in args.predictions],
        "references": args.refs, "v2_dir": str(args.v2_dir), "n_boot": args.n_boot,
        "seed": args.seed, "min_support": MIN_SUPPORT, "notes": notes,
        "results": results}, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
