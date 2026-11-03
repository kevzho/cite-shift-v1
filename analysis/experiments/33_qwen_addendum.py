#!/usr/bin/env python3
"""Addendum (12 Oct 2026; DEVIATIONS.md #6): the registered local LLM (Qwen2.5-7B-Instruct, ollama, frozen v2 prompts)
added to the lineup after the 8 Oct confirmatory run.

1) Re-runs 29_confirmatory_2026_10_08.py UNCHANGED except that Qwen is added to its systems, into
   analysis/results/confirmatory_2026-10-12_with_qwen/ (the 8 Oct outputs are not touched). RQ1(b), RQ2 and D1 do not
   involve Qwen and must reproduce exactly (checked).
2) Inter-LLM agreement on identical prompts (§7): Sol vs Qwen vs Jev, per field, raw agreement + Cohen's kappa, on
   the 550 v2 edges (B + A250). System-to-system agreement, never reliability.
3) Qwen run-to-run (§7 LLM consistency): two fresh runs of the first 100 of B, per field raw + kappa.
Outputs refuse to overwrite. Note: docs/results/<date>_qwen_addendum.md.
"""
from __future__ import annotations

import csv
import importlib.util
import json
import time
from collections import Counter
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("conf", ROOT / "analysis/experiments/29_confirmatory_2026_10_08.py")
C = importlib.util.module_from_spec(spec)
spec.loader.exec_module(C)
S = C.S
RES = ROOT / "analysis/results"
Q = RES / "external_systems/local_llm_codebook_v2"
OLD, NEW = RES / "confirmatory_2026-10-08", RES / "confirmatory_2026-10-12_with_qwen"
FIELDS = C.PASS1 + C.PASS2
QWEN = "qwen2.5-7b"


def qwen_rows(tag):
    return {r["edge_id"]: r for r in C.read(Q / f"edges_{tag}.csv")}


def kappa(pairs):
    n = len(pairs)
    if not n:
        return None, None
    po = sum(a == b for a, b in pairs) / n
    ca, cb = Counter(a for a, _ in pairs), Counter(b for _, b in pairs)
    pe = sum(ca[k] * cb[k] for k in set(ca) | set(cb)) / n / n
    return round(po, 4), (round((po - pe) / (1 - pe), 4) if pe < 1 else None)


def labels(rows, field, cw, alias):
    out = {}
    for e, r in rows.items():
        v = (r.get(field) or "").strip()
        if v and v != "not_applicable":
            out[alias.get(e, e)] = S.norm(field, v, cw)
    return out


def main():
    orig = C.load_systems

    def with_qwen(cw, alias):
        sysp = orig(cw, alias)
        for f, d in ((f, labels(qwen_rows("v2edges"), f, cw, alias)) for f in FIELDS):
            sysp[QWEN][0][f].update(d)
        return sysp

    C.load_systems = with_qwen
    C.OUT = NEW
    C.main()

    same = {n: json.loads((OLD / n).read_text()) == json.loads((NEW / n).read_text())
            for n in ("rq1b.json", "rq2.json", "d1.json")}
    old3, new3 = json.loads((OLD / "rq3.json").read_text()), json.loads((NEW / "rq3.json").read_text())
    qscores = {k: v for k, v in new3["scores"].items() if f"|{QWEN}|" in k}

    cw = S.crosswalk()
    alias, _ = S.components()
    v2 = qwen_rows("v2edges")
    ids = {alias.get(e, e) for e in v2}
    sysl = {QWEN: {f: labels(v2, f, cw, alias) for f in FIELDS}}
    for name, path in (("gpt-6.1-sol", RES / "external_systems/hosted_llm_codebook_v2/edges_canonical.csv"),
                       ("jev-1.13.0", RES / "external_systems/jev_codebook_v2/edges_canonical.csv")):
        rows = {r["edge_id"]: r for r in C.read(path)}
        sysl[name] = {f: labels(rows, f, cw, alias) for f in FIELDS}
    inter = {}
    for a, b in combinations(sorted(sysl), 2):
        for f in FIELDS:
            pairs = [(sysl[a][f][e], sysl[b][f][e]) for e in sorted(ids) if e in sysl[a][f] and e in sysl[b][f]]
            raw, k = kappa(pairs)
            inter[f"{a} vs {b}|{f}"] = {"n": len(pairs), "raw_agreement": raw, "kappa": k}
    s1, s2 = qwen_rows("s1"), qwen_rows("s2")
    stab = {}
    for f in FIELDS:
        pairs = [((s1[e].get(f) or "").strip(), (s2[e].get(f) or "").strip()) for e in sorted(set(s1) & set(s2))]
        raw, k = kappa(pairs)
        stab[f] = {"n": len(pairs), "raw_agreement": raw, "kappa": k}
    validity = {"v2edges_all_passes_valid": sum(r["all_passes_valid"] == "1" for r in v2.values()),
                "v2edges_degree_rule_ok": sum(r["degree_rule_ok"] == "1" for r in v2.values()), "v2edges_n": len(v2),
                "meta": json.loads((Q / "meta_v2edges.json").read_text())}
    out = {"reproduced_unchanged_8_oct": same, "qwen_scores": qscores,
           "rq3_8_oct": {"verdict": old3["verdict"], "max_flip": old3["max_flip_probability"],
                         "kendall_tau": old3["kendall_tau_canonical_vs_kevin"],
                         "n_flips_gt_0.5_ref": sum(x["p_flip"] > 0.5 for x in old3["flip_probabilities_between_references"]),
                         "n_pairs_ref": len(old3["flip_probabilities_between_references"]),
                         "n_flips_gt_0.5_field": sum(x["p_flip"] > 0.5 for x in old3["flip_probabilities_between_fields"]),
                         "n_pairs_field": len(old3["flip_probabilities_between_fields"])},
           "rq3_with_qwen": {"verdict": new3["verdict"], "max_flip": new3["max_flip_probability"],
                             "kendall_tau": new3["kendall_tau_canonical_vs_kevin"],
                             "n_flips_gt_0.5_ref": sum(x["p_flip"] > 0.5 for x in new3["flip_probabilities_between_references"]),
                             "n_pairs_ref": len(new3["flip_probabilities_between_references"]),
                             "n_flips_gt_0.5_field": sum(x["p_flip"] > 0.5 for x in new3["flip_probabilities_between_fields"]),
                             "n_pairs_field": len(new3["flip_probabilities_between_fields"]),
                             "construct_recovery_qwen": new3["construct_recovery_B"].get(QWEN)},
           "inter_llm_agreement_v2_edges": inter, "qwen_run_to_run_B100": stab, "qwen_validity": validity}
    p = NEW / "qwen_addendum.json"
    if p.exists():
        raise SystemExit(f"refusing to overwrite {p}")
    p.write_text(json.dumps(out, indent=2, default=float) + "\n")

    day = time.strftime("%Y-%m-%d")
    o, n = out["rq3_8_oct"], out["rq3_with_qwen"]
    L = [f"# Qwen2.5-7B added to the lineup — addendum ({day})", "",
         "**Registered system, run late** (DEVIATIONS.md #6: ollama 0.40.0 instead of 0.34.4; same model digest and frozen",
         "prompts as Sol). Script `analysis/experiments/33_qwen_addendum.py`; outputs `analysis/results/confirmatory_2026-10-12_with_qwen/`.",
         "The 8 Oct confirmatory outputs are unchanged; this re-runs the same script with Qwen added.", "",
         f"Reproduced exactly without Qwen involvement: {same}.",
         f"Validity: {validity['v2edges_all_passes_valid']}/{len(v2)} edges schema-valid on both passes; "
         f"degree rule kept on {validity['v2edges_degree_rule_ok']}/{len(v2)}; {validity['meta']['s_per_edge']} s/edge.", "",
         "## Qwen vs human references (primary variant)", "", "| Reference | Field | n | macro-F1 [95% CI] | κ |", "|---|---|---|---|---|"]
    for k, v in sorted(qscores.items()):
        ref, f, _, var = k.split("|")
        if var == "primary":
            L.append(f"| {ref} | {f} | {v['n']} | {v['macro_f1']} [{v['macro_f1_ci95'][0]}, {v['macro_f1_ci95'][1]}] | {v['kappa']} |")
    L += ["", "## RQ3 with Qwen added", "", "| | 8 Oct lineup | + Qwen |", "|---|---|---|",
          f"| Verdict | {o['verdict']} | {n['verdict']} |",
          f"| Max flip probability | {o['max_flip']} | {n['max_flip']} |",
          f"| Flips p > 0.5 between references | {o['n_flips_gt_0.5_ref']}/{o['n_pairs_ref']} | {n['n_flips_gt_0.5_ref']}/{n['n_pairs_ref']} |",
          f"| Flips p > 0.5 between fields | {o['n_flips_gt_0.5_field']}/{o['n_pairs_field']} | {n['n_flips_gt_0.5_field']}/{n['n_pairs_field']} |",
          f"| Kendall τ (canonical vs Kevin) | {json.dumps(o['kendall_tau'])} | {json.dumps(n['kendall_tau'])} |", "",
          f"Construct recovery, Qwen (B): {json.dumps(n['construct_recovery_qwen'])}", "",
          "## Inter-LLM agreement on identical prompts (550 v2 edges; system-to-system, not reliability)", "",
          "| Pair | Field | n | Raw | κ |", "|---|---|---|---|---|"]
    L += [f"| {k.split('|')[0]} | {k.split('|')[1]} | {v['n']} | {v['raw_agreement']} | {v['kappa']} |" for k, v in inter.items()]
    L += ["", "## Qwen run-to-run (two fresh runs, first 100 of B)", "", "| Field | n | Raw | κ |", "|---|---|---|---|"]
    L += [f"| {f} | {v['n']} | {v['raw_agreement']} | {v['kappa']} |" for f, v in stab.items()]
    (ROOT / f"docs/results/{day}_qwen_addendum.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
