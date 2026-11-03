#!/usr/bin/env python3
"""LaTeX tables for the iConference manuscript, generated from the result files (27 Oct 2026).

    python3 analysis/experiments/61_paper_tables.py [--out-dir DIR]

Default output: analysis/results/paper_numbers/tables/ (refuses to overwrite existing files).
Inputs: sensitivity_2026-10-23_B300/{rq1b,rq2,rq3}.json, exploratory/2026-10-11_rq4a_final.json,
paper_numbers/derived_numbers.json (script 60), and the private v2 coding keys.
The deviations table is a dated record of protocol changes, not a computed result; it is not generated here.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "analysis/results"
B300 = RES / "sensitivity_2026-10-23_B300"
V2 = ROOT / "annotations/generated/v2_coding_2026-10-02"

FIELD = {"cited_contribution_type": "Contribution type", "citation_function": "Citation function",
         "credits_contribution": "Crediting", "uptake_mode": "Uptake mode", "uptake_degree": "Uptake degree",
         "claim_dependent": "Claim dependence"}
SYSTEM = {"gpt-6.1-sol": "GPT-6.1 Sol", "jev-1.13.0": "Jev 1.13.0", "d1_distilled": "D1 (distilled)",
          "encoder_scibert": "SciBERT encoder", "scicite": "SciCite", "s2_intents": "S2 intents",
          "multicite": "MultiCite", "s2_influential": "S2 influential", "no_model_baseline": "No-model baseline"}
SOURCE = "\\aisource Author's project data and frozen system outputs.\n"


def load(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def read(p):
    with Path(p).open(newline="", encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def f2(x):
    return f"{x:.2f}"


def f3(x):
    return f"{x:.3f}"


def ci(c, fmt=f2):
    return f"[{fmt(c[0])},{fmt(c[1])}]"


def table(colspec, header, rows, caption, label, size="\\footnotesize"):
    body = "\n".join(" & ".join(r) + " \\\\" for r in rows)
    return (f"\\begin{{table}}[htbp]\n\\centering{size}\n\\setlength{{\\tabcolsep}}{{3pt}}\n\\begin{{tabular}}{{{colspec}}}\n"
            f"\\toprule\n{' & '.join(header)} \\\\\n\\midrule\n{body}\n\\bottomrule\n\\end{{tabular}}\n"
            f"\\caption{{{caption}}}\n\\label{{{label}}}\n{SOURCE}\\end{{table}}\n")


def samples(derived):
    canon = Counter()
    for c in ("hci", "nature_oa"):
        canon[c] = sum(1 for _ in read(ROOT / f"analysis/ml/splits/{c}_splits.csv"))
    a500 = Counter(r["corpus"] for r in read(V2 / "A_pass2_key.csv"))
    b = Counter(r["corpus"] for p in ("B_stratum_pass2_key.csv", "B_rest_pass2_key.csv") for r in read(V2 / p))
    a250 = Counter(r["corpus"] for r in read(V2 / "A250_pass2_key.csv"))
    el = derived["b300_eligible_by_corpus"]["eligible_by_corpus"]
    n = lambda c: f"{sum(c.values()):,}"
    rows = [["Canonical", n(canon), f"{canon['hci']:,}", f"{canon['nature_oa']:,}"],
            ["Earlier v1 coding", n(a500), str(a500["hci"]), str(a500["nature_oa"])],
            ["B300, Pass 1 and Pass 2", n(b), str(b["hci"]), str(b["nature_oa"])],
            ["B300, RQ1 eligible", str(sum(el.values())), str(el.get("hci", 0)), str(el.get("nature_oa", 0))],
            ["A250, Pass 2", n(a250), str(a250["hci"]), str(a250["nature_oa"])]]
    return table("lrrr", ["Record", "Edges", "HCI", "Nature OA"], rows,
                 "Reference records and the B300 presentation. Other evaluations apply field-specific eligibility rules.",
                 "tab:samples")


def confirmatory(derived):
    r1, r2 = load(B300 / "rq1b.json"), load(B300 / "rq2.json")
    b, a = r1["B"]["primary_realized_partial_or_yes"], r1["A250_replication"]["primary_realized_partial_or_yes"]
    cells = list(r2["a_critical_cells_B"]["counts"].values())
    nmi = r2["b_function_realization_nmi"]["A250"]
    fl = derived["rq3_flips"]["b300"]
    rows = [["RQ1(b), B", "Crediting-minus-uptake $\\kappa$", f3(b["kappa_difference_crediting_minus_uptake"]),
             f"${ci(b['difference_ci95'], f3)}$; {'unsupported' if b['difference_ci95'][0] <= 0 else 'supported'}"],
            ["RQ1(b), A", "Replication $\\kappa$ contrast", f3(a["kappa_difference_crediting_minus_uptake"]),
             f"${ci(a['difference_ci95'], f3)}$"],
            ["RQ2(a), B", "Critical-cell counts", "/".join(map(str, cells)),
             "All nonempty" if all(cells) else "Some empty"],
            ["RQ2(b), A", "NMI difference from 0.700", f3(nmi["difference_vs_0.700"]), f"${ci(nmi['difference_ci95'], f3)}$"],
            ["RQ3", "October 8-lineup reference / field flips",
             f"{fl['references']['flips']}/{fl['references']['pairs']}; {fl['fields']['flips']}/{fl['fields']['pairs']}",
             "Supported" if fl["references"]["flips"] + fl["fields"]["flips"] else "Not supported"]]
    return table("p{.12\\textwidth}p{.39\\textwidth}rp{.23\\textwidth}", ["Question", "Estimand", "Estimate", "95\\% CI / decision"],
                 rows, "Preregistered questions evaluated on the B300 reference.", "tab:confirmatory")


def rq3_scores():
    sc = load(B300 / "rq3.json")["scores"]
    rows = []
    for f in FIELD:
        systems = [s for s in SYSTEM if f"kevin_v2@B|{f}|{s}|primary" in sc]
        for s in systems:
            c, v = sc.get(f"canonical@B|{f}|{s}|primary"), sc[f"kevin_v2@B|{f}|{s}|primary"]
            rows.append([FIELD[f], SYSTEM[s],
                         f"{f2(c['macro_f1'])} {ci(c['macro_f1_ci95'])}" if c else "--", f2(c["kappa"]) if c else "--",
                         f"{f2(v['macro_f1'])} {ci(v['macro_f1_ci95'])}", f2(v["kappa"])])
    return table("p{.16\\textwidth}p{.17\\textwidth}p{.21\\textwidth}rp{.21\\textwidth}r",
                 ["Field", "System", "F1 vs canonical", "$\\kappa$", "F1 vs B300", "$\\kappa$"], rows,
                 "Macro-F1 with 95\\% intervals and agreement for the October 8 lineup evaluated on B300. Dashes: the "
                 "canonical record has no labels for the four fields introduced in v2.", "tab:rq3_scores")


def rq3_support():
    sc = load(B300 / "rq3.json")["scores"]
    full = {f: max(v["n"] for k, v in sc.items() if k.startswith(f"kevin_v2@B|{f}|") and k.endswith("|primary"))
            for f in FIELD}
    rows = []
    for k, v in sc.items():
        ref, f, s, variant = k.split("|")
        if ref != "kevin_v2@B" or variant != "primary" or s not in SYSTEM:
            continue
        withheld = [w for w in v["withheld"] if w != "unclear"]
        if v["n"] < full[f] or withheld or f == "uptake_degree":
            rows.append([FIELD[f], SYSTEM[s], str(v["n"]), ", ".join(w.replace("_", " ") for w in withheld) or "none",
                         f2(v["majority_rate"])])
    return table("llrlr", ["Field", "System", "Edges scored", "Classes withheld ($<5$)", "Majority rate"], rows,
                 "Coverage and withheld classes against the v2 B reference, for cells with reduced coverage or withheld "
                 "classes. Uptake degree is scored only where both the reference and the system record uptake.",
                 "tab:rq3_support")


def rq4a():
    res = {(r["source"], r["construct"]): r for r in load(RES / "exploratory/2026-10-11_rq4a_final.json")["results"]}
    order = [("canonical", "realized"), ("gpt-6.1-sol", "credits"), ("gpt-6.1-sol", "uptake"), ("gpt-6.1-sol", "claim"),
             ("jev-1.13.0", "credits"), ("jev-1.13.0", "uptake"), ("jev-1.13.0", "claim"), ("multicite", "uptake"),
             ("s2_influential", "claim")]
    name = {"canonical": "Canonical", **SYSTEM}
    con = {"realized": "realized", "credits": "credited specifically", "uptake": "uptake", "claim": "claim dependent"}
    rows = [[name[s], con[c], f3(res[(s, c)]["rate_hci"]), f3(res[(s, c)]["rate_nature"]),
             f"${f3(res[(s, c)]['delta']).replace('-', '-')}$ $[{f3(res[(s, c)]['ci95'][0])},{f3(res[(s, c)]['ci95'][1])}]$"]
            for s, c in order]
    return table("llrrl", ["Source", "Construct", "HCI", "Nature OA", "HCI$-$Nature [95\\% CI]"], rows,
                 "Positive-label rates for HCI and Nature OA on the canonical labeled sample, with the HCI$-$Nature "
                 "difference and 95\\% confidence interval.", "tab:rq4a", size="\\small")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out-dir", type=Path, default=RES / "paper_numbers/tables")
    ap.add_argument("--derived", type=Path, default=RES / "paper_numbers/derived_numbers.json")
    a = ap.parse_args()
    derived = load(a.derived)
    out = {"samples.tex": samples(derived), "confirmatory.tex": confirmatory(derived), "rq3_scores.tex": rq3_scores(),
           "rq3_support.tex": rq3_support(), "rq4a.tex": rq4a()}
    a.out_dir.mkdir(parents=True, exist_ok=True)
    clash = [n for n in out if (a.out_dir / n).exists()]
    if clash:
        sys.exit(f"refusing to overwrite {clash} in {a.out_dir}")
    for n, s in out.items():
        (a.out_dir / n).write_text(s, encoding="utf-8")
        print("wrote", a.out_dir / n)


if __name__ == "__main__":
    main()
