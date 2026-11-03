#!/usr/bin/env python3
"""Exploratory specification curve for the HCI − Nature gap in 'positive' citation-use labels, assembled ONLY from
outputs already on disk (no model, API or bootstrap is run here). Δ = HCI (CHI) rate − Nature rate; negative means
Nature > HCI. Panel A: the labelled frame (2,937 canonical edges, HCI 1,040 vs Nature OA 1,897, and its registered
subsets), every RQ4(a) system × construct, S2 isInfluential over all flagged edges vs context-only, and the SciBERT
encoder seeds (conditioned, unconditioned). Panel B: the at-scale frame (CHI vs Nature arm, MultiCite uptake among
edges with a claim) under JATS vs GROBID on the 200-paper parity set, the recovered-edge subset, and the repaired
parser with and without de-duplication. Where a specification's CI is not on disk the row keeps its point estimate
only and is flagged ci_missing; if the point estimate is itself not stored, it is the difference of the two stored
rates (point_derived). A specification with no stored number at all is listed under 'unsourced'.

    python3 analysis/experiments/70_specification_curve.py [--out PATH] [--fig PATH] [--png]
Default outputs: analysis/results/spec_curve_2026-11-03.json and paper_iconf/figures/fig_spec_curve.pdf
(refuses to overwrite either).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RQ4A = "analysis/results/exploratory/2026-10-11_rq4a_final.json"
S2C = "analysis/results/exploratory/2026-10-12_s2_consequence.json"
REC = "analysis/results/exploratory/2026-10-12_recovered_nature_edges.json"
ATSCALE = "analysis/results/exploratory/2026-10-07_multicite_at_scale.json"
PBIAS = "external_data/nature_arm_2026-10-04/parser_bias/parser_bias_result.json"
REPAIR = "external_data/nature_repair_v2_2026-10-25/classifier"
FULL_RAW, FULL_DEDUP = f"{REPAIR}/full_v6/summary.json", f"{REPAIR}/full_v6_dedup/summary.json"
DEV_REPAIRED = f"{REPAIR}/development_repaired_v6/summary.json"

SYSTEMS = {"canonical": "Canonical (human)", "canonical@frozen": "Canonical (human)", "gpt-6.1-sol": "GPT-6.1 Sol",
           "jev-1.13.0": "Jev 1.13.0", "multicite": "MultiCite", "s2_influential": "S2 isInfluential",
           "encoder_uncond": "SciBERT enc., uncond.", "encoder_cond": "SciBERT enc., cond."}
CONSTRUCTS = {"realized": "realized", "credits": "credited specifically", "uptake": "uptake",
              "claim": "claim dependent", "influential": "influential"}
F_ALL, F_FROZEN = "Labelled, all 2,937", "Labelled, frozen 1,631"
F_S2ALL, F_S2CTX = "Labelled, S2-flagged 2,917", "Labelled, S2 context 1,316"
F_ARM, F_PARITY, F_RECOV = "At scale, 1,200 Nature papers", "Parity, 200 papers", "Parity, recovered edges"


def load(rel):
    with open(ROOT / rel, encoding="utf-8") as fh:
        return json.load(fh)


def spec(panel, label, system, construct, frame, est, ci, n_hci, n_nat, src, key, *, rate_hci=None, rate_nat=None,
         threshold=None, s2_denominator=None, parser=None, dedup=None, seed=None, registered=False,
         point_derived=False, note=None):
    return {"panel": panel, "label": label, "system": system, "construct": construct, "frame": frame,
            "threshold": threshold, "s2_denominator": s2_denominator, "parser": parser, "dedup": dedup, "seed": seed,
            "estimate": round(est, 4), "ci95": None if ci is None else [round(ci[0], 4), round(ci[1], 4)],
            "ci_missing": ci is None, "ci_excludes_0": None if ci is None else bool(ci[0] > 0 or ci[1] < 0),
            "point_derived": point_derived, "n_hci": n_hci, "n_nature": n_nat,
            "rate_hci": None if rate_hci is None else round(rate_hci, 4),
            "rate_nature": None if rate_nat is None else round(rate_nat, 4),
            "registered_rq4a": registered, "source_file": src, "source_key": key, "note": note}


def panel_a():
    rq = load(RQ4A)["results"]
    out = []
    for i, r in enumerate(rq):
        s = r["source"]
        if s.startswith("d1_"):
            continue  # D1 is exploratory, not a registered RQ4(a) system
        seed = None
        if s.startswith("encoder_"):
            sysk, seed = s.rsplit("_s", 1)
            seed = int(seed)
        else:
            sysk = s
        frame = F_FROZEN if (s == "canonical@frozen" or s.startswith("encoder_")) else F_ALL
        construct = "influential" if s == "s2_influential" else r["construct"]
        thr = "partial or yes" if s.startswith("canonical") else None
        lab = (f"{SYSTEMS[sysk]} · {CONSTRUCTS[construct]}" + (f" · s{seed}" if seed else "")
               + (" · frozen 1,631" if s == "canonical@frozen" else ""))
        out.append(spec("A", lab, SYSTEMS[sysk], CONSTRUCTS[construct], F_S2CTX if s == "s2_influential" else frame,
                        r["delta"], r["ci95"], r["n_hci"], r["n_nature"], RQ4A,
                        f"results[{i}] (source={s!r}, construct={r['construct']!r})",
                        rate_hci=r["rate_hci"], rate_nat=r["rate_nature"], threshold=thr, seed=seed,
                        s2_denominator="context only" if s == "s2_influential" else None, registered=True,
                        note=("registered rule: S2 scored only where it has context; s2_consequence.json re-bootstrap "
                              "of the same contrast gives CI [−0.0617, +0.0197]") if s == "s2_influential" else None))
    s2 = load(S2C)["flag_all_citations_missing_as_false"]
    out.append(spec("A", "S2 isInfluential · influential · all flagged", SYSTEMS["s2_influential"], "influential",
                    F_S2ALL, s2["delta"], s2["ci95"], s2["n_hci"], s2["n_nature"], S2C,
                    "flag_all_citations_missing_as_false", rate_hci=s2["rate_hci"], rate_nat=s2["rate_nature"],
                    s2_denominator="all flagged (no context = 0)",
                    note="published-pattern denominator (Cunningham & Smyth); exploratory, not registered"))
    return out


def panel_b():
    rec, pb, ats = load(REC), load(PBIAS), load(ATSCALE)
    raw, dd, dev = load(FULL_RAW), load(FULL_DEDUP), load(DEV_REPAIRED)
    chi = rec["rates"]["chi_claims_only"]
    chi_rate, chi_n = chi["uptake_share"], chi["claims"]
    assert chi_n == 152062 and abs(chi_rate - 0.1175) < 5e-5, "CHI baseline differs from 11.75% of 152,062"
    mc, out = "MultiCite", []
    # original GROBID extraction, full 1,200-paper arm (dated record of 7 Oct)
    og = raw["branches"]["original_grobid"]
    out.append(spec("B", "GROBID (original) · 1,200-paper arm", mc, "uptake", F_ARM,
                    ats["delta_chi_minus_nature"]["delta"], ats["delta_chi_minus_nature"]["ci95"], chi_n,
                    og["claims"], ATSCALE, "delta_chi_minus_nature", rate_hci=chi_rate,
                    rate_nat=ats["corpora"]["nature_arm"]["uptake_share"], parser="GROBID (original)",
                    note=f"n_nature from {FULL_RAW} branches.original_grobid.claims; the repair summaries "
                         "re-bootstrap the same contrast as −0.0530 [−0.0609, −0.0456]"))
    for summ, path, dedup, lab in [(raw, FULL_RAW, "no", "Repaired v6, raw formats · 1,200-paper arm"),
                                   (dd, FULL_DEDUP, "yes", "Repaired v6, de-duplicated · 1,200-paper arm")]:
        b, c = summ["branches"]["legacy"], summ["chi_minus_nature_descriptive"]["legacy"]
        out.append(spec("B", lab, mc, "uptake", F_ARM, c["difference"], c["ci95"], chi_n, b["claims"], path,
                        "chi_minus_nature_descriptive.legacy", rate_hci=chi_rate,
                        rate_nat=b["uptake_share_among_claims"], parser="repaired v6", dedup=dedup))
    # 200-paper parity set: no CHI-minus-Nature CI stored → point only, from the stored rates
    up = pb["systems"]["multicite"]["edge_level_uptake_among_claims"]
    for ver, lab, parser in [("jats", "JATS · parity 200", "JATS (publisher XML)"),
                             ("grobid", "GROBID (original) · parity 200", "GROBID (original)")]:
        r = up["positive_share"][ver]
        out.append(spec("B", lab, mc, "uptake", F_PARITY, chi_rate - r, None, chi_n, up[f"n_{ver}"], PBIAS,
                        f"systems.multicite.edge_level_uptake_among_claims.positive_share.{ver} (CHI rate from "
                        f"{REC} rates.chi_claims_only)", rate_hci=chi_rate, rate_nat=r, parser=parser,
                        point_derived=True))
    b = dev["branches"]["legacy"]
    out.append(spec("B", "Repaired v6, raw formats · parity 200", mc, "uptake", F_PARITY,
                    chi_rate - b["uptake_share_among_claims"], None, chi_n, b["claims"], DEV_REPAIRED,
                    f"branches.legacy.uptake_share_among_claims (CHI rate from {REC} rates.chi_claims_only)",
                    rate_hci=chi_rate, rate_nat=b["uptake_share_among_claims"], parser="repaired v6", dedup="no",
                    point_derived=True))
    # recovered-edge subset of the parity set
    j = rec["rates"]["nature_jats_recovered"]
    out.append(spec("B", "JATS · recovered edges", mc, "uptake", F_RECOV, chi_rate - j["uptake_share"], None, chi_n,
                    j["claims"], REC, "rates.nature_jats_recovered.uptake_share − rates.chi_claims_only.uptake_share",
                    rate_hci=chi_rate, rate_nat=j["uptake_share"], parser="JATS (publisher XML)", point_derived=True))
    g, c = rec["rates"]["nature_grobid_recovered"], rec["contrasts"]["chi_minus_nature_recovered"]
    out.append(spec("B", "GROBID (original) · recovered edges", mc, "uptake", F_RECOV, c["difference"], c["ci95"],
                    chi_n, g["claims"], REC, "contrasts.chi_minus_nature_recovered", rate_hci=chi_rate,
                    rate_nat=g["uptake_share"], parser="GROBID (original)"))
    return out


def check(specs):
    """Known values from the dated records; abort if any source drifted."""
    by = {s["label"]: s for s in specs}
    d = by["Repaired v6, de-duplicated · 1,200-paper arm"]
    assert d["n_nature"] == 41559 and abs(d["rate_nature"] - 0.1901) < 5e-5
    assert abs(d["estimate"] + 0.0726) < 5e-5 and d["ci95"] == [-0.0814, -0.0642]
    assert abs(by["Repaired v6, raw formats · 1,200-paper arm"]["rate_nature"] - 0.1771) < 5e-5
    g = by["GROBID (original) · recovered edges"]
    assert abs(g["rate_nature"] - 0.1741) < 5e-5 and g["estimate"] == -0.0567 and g["ci95"] == [-0.0751, -0.0384]
    c = by["Canonical (human) · realized"]
    assert (c["n_hci"], c["n_nature"], c["estimate"]) == (1040, 1897, -0.09)


# ---------------------------------------------------------------- figure
INK, INK2, INK3, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8985", "#e6e5e1", "#ffffff"
GRID_ROWS = [  # (group, row label, predicate)
    ("Frame", F_ALL, lambda s: s["frame"] == F_ALL), ("Frame", F_FROZEN, lambda s: s["frame"] == F_FROZEN),
    ("Frame", F_S2ALL, lambda s: s["frame"] == F_S2ALL), ("Frame", F_S2CTX, lambda s: s["frame"] == F_S2CTX),
    ("Frame", F_ARM, lambda s: s["frame"] == F_ARM), ("Frame", F_PARITY, lambda s: s["frame"] == F_PARITY),
    ("Frame", F_RECOV, lambda s: s["frame"] == F_RECOV),
    *[("System", v, (lambda v: lambda s: s["system"] == v)(v)) for v in
      ["Canonical (human)", "GPT-6.1 Sol", "Jev 1.13.0", "MultiCite", "S2 isInfluential", "SciBERT enc., uncond.",
       "SciBERT enc., cond."]],
    *[("Construct", v, (lambda v: lambda s: s["construct"] == v)(v)) for v in
      ["realized", "credited specifically", "uptake", "claim dependent", "influential"]],
    ("Threshold", "realized = partial or yes", lambda s: s["threshold"] == "partial or yes"),
    ("Threshold", "S2: all flagged (no context = 0)", lambda s: s["s2_denominator"] == "all flagged (no context = 0)"),
    ("Threshold", "S2: context only", lambda s: s["s2_denominator"] == "context only"),
    ("Parser", "JATS (publisher XML)", lambda s: s["parser"] == "JATS (publisher XML)"),
    ("Parser", "GROBID (original)", lambda s: s["parser"] == "GROBID (original)"),
    ("Parser", "Repaired v6", lambda s: s["parser"] == "repaired v6"),
    ("De-dup.", "Raw context formats", lambda s: s["dedup"] == "no"),
    ("De-dup.", "De-duplicated", lambda s: s["dedup"] == "yes"),
]


FIG_W = 5.75  # + right-hand group labels ≈ 6.5 in after a tight bbox


def figure(specs, path, png=False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix", "font.size": 8, "axes.edgecolor": INK3, "axes.labelcolor": INK,
        "xtick.color": INK2, "ytick.color": INK, "text.color": INK, "axes.spines.top": False,
        "axes.spines.right": False, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "pdf.fonttype": 42,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.04, "axes.titlesize": 8.5, "xtick.labelsize": 8,
        "ytick.labelsize": 8, "legend.fontsize": 8, "legend.frameon": False})
    panels = {p: sorted([s for s in specs if s["panel"] == p], key=lambda s: s["estimate"]) for p in "AB"}
    rows = [r for r in GRID_ROWS if any(r[2](s) for s in specs)]
    nA, nB = len(panels["A"]), len(panels["B"])
    fig, axes = plt.subplots(2, 2, figsize=(FIG_W, 6.2), sharey="row",
                             gridspec_kw={"width_ratios": [nA + 0.6, nB + 0.6], "height_ratios": [1.0, 1.55],
                                          "wspace": 0.05, "hspace": 0.06})
    lo = min(min(s["ci95"][0] if s["ci95"] else s["estimate"] for s in specs), 0) * 100
    hi = max(max(s["ci95"][1] if s["ci95"] else s["estimate"] for s in specs), 0) * 100
    groups = []
    for k, p in enumerate("AB"):
        ax, gx, ss = axes[0, k], axes[1, k], panels[p]
        ax.axhline(0, color=INK2, linewidth=1.0, zorder=1.8)
        ax.yaxis.grid(True, color=GRID, linewidth=0.6, zorder=0)
        for x, s in enumerate(ss):
            y = s["estimate"] * 100
            if s["ci_missing"]:
                ax.plot(x, y, marker="D", markersize=4.2, markerfacecolor=SURFACE, markeredgecolor=INK3,
                        markeredgewidth=1.1, linestyle="none", zorder=3)
                continue
            l, h = (v * 100 for v in s["ci95"])
            ax.plot([x, x], [l, h], color=INK, linewidth=1.1, solid_capstyle="butt", zorder=2)
            ax.plot(x, y, marker="o", markersize=4.4, markerfacecolor=INK if s["ci_excludes_0"] else SURFACE,
                    markeredgecolor=INK, markeredgewidth=1.0, linestyle="none", zorder=3)
        ax.set_xlim(-0.7, len(ss) - 0.3)
        ax.set_ylim(lo - 2, hi + 2)
        ax.set_xticks([])
        ax.spines["bottom"].set_visible(False)
        ax.set_title(["A  Labelled frame (HCI vs Nature OA)", "B  At scale (CHI vs Nature)"][k], loc="left",
                     fontweight="bold")
        # indicator grid
        for i, (grp, lab, pred) in enumerate(rows):
            if i % 2 == 0:
                gx.axhspan(i - 0.5, i + 0.5, color="#f5f4f1", zorder=0, linewidth=0)
            for x, s in enumerate(ss):
                if pred(s):
                    if grp == "System" and s["seed"]:
                        gx.text(x, i, str(s["seed"]), ha="center", va="center", fontsize=8, color=INK,
                                fontweight="bold", zorder=3)
                    else:
                        gx.plot(x, i, marker="s", markersize=4.0, color=INK, linestyle="none", zorder=3)
        gx.set_xlim(-0.7, len(ss) - 0.3)
        gx.set_ylim(len(rows) - 0.5, -0.5)
        gx.set_xticks([])
        for sp in ("top", "right", "bottom"):
            gx.spines[sp].set_visible(False)
        if k == 1:
            gx.spines["left"].set_visible(False)
            gx.tick_params(axis="y", length=0)
            ax.spines["left"].set_visible(False)
            ax.tick_params(axis="y", length=0)
        gx.set_xlabel(f"{len(ss)} specifications, sorted by Δ", fontsize=8, color=INK2, labelpad=3)
    # group separators and labels on the grid
    prev = None
    for i, (grp, _, _) in enumerate(rows):
        if grp != prev:
            if prev is not None:
                for k in (0, 1):
                    axes[1, k].axhline(i - 0.5, color=INK3, linewidth=0.6, zorder=1)
            groups.append((grp, i))
            prev = grp
    bounds = [g[1] for g in groups] + [len(rows)]
    for (grp, start), end in zip(groups, bounds[1:]):
        axes[1, 1].annotate(grp, xy=(1.0, (start + end - 1) / 2), xycoords=("axes fraction", "data"),
                            xytext=(4, 0), textcoords="offset points", ha="left", va="center", fontsize=8,
                            color=INK2, rotation=0)
    axes[1, 0].set_yticks(range(len(rows)))
    axes[1, 0].set_yticklabels([r[1] for r in rows])
    axes[0, 0].set_ylabel("Δ = HCI − Nature (percentage points)")
    axes[0, 0].yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:+.0f}".replace("-", "−")
                                                                          if v else "0"))
    handles = [Line2D([], [], marker="o", color=INK, markerfacecolor=INK, linestyle="-", linewidth=1.1,
                      markersize=4.4, label="95% CI excludes 0"),
               Line2D([], [], marker="o", color=INK, markerfacecolor=SURFACE, linestyle="-", linewidth=1.1,
                      markersize=4.4, label="95% CI includes 0"),
               Line2D([], [], marker="D", color=INK3, markerfacecolor=SURFACE, linestyle="none", markersize=4.2,
                      label="point only (no CI on disk)")]
    axes[0, 0].legend(handles=handles, loc="lower right", ncol=1, handlelength=1.6, borderaxespad=0.3)
    axes[0, 0].text(0.01, 0.98, "Δ < 0: Nature > HCI", transform=axes[0, 0].transAxes, ha="left", va="top",
                    fontsize=8, color=INK2)
    axes[1, 0].set_xlabel(f"{nA} specifications, sorted by Δ (grid digits: encoder seed)", fontsize=8, color=INK2,
                          labelpad=3)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    if png:
        fig.savefig(path.with_suffix(".png"), dpi=200)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, default=ROOT / "analysis/results/spec_curve_2026-11-03.json")
    ap.add_argument("--fig", type=Path, default=ROOT / "paper_iconf/figures/fig_spec_curve.pdf")
    ap.add_argument("--png", action="store_true", help="also write a 200-dpi PNG beside the PDF")
    a = ap.parse_args()
    for p in (a.out, a.fig) + ((a.fig.with_suffix(".png"),) if a.png else ()):
        if p.exists():
            sys.exit(f"refusing to overwrite {p}")
    specs = panel_a() + panel_b()
    assert len({s["label"] for s in specs}) == len(specs), "specification labels must be unique"
    check(specs)
    summ = {}
    for p in "AB":
        ss = [s for s in specs if s["panel"] == p]
        withci = [s for s in ss if not s["ci_missing"]]
        summ[p] = {"n_specs": len(ss), "min_estimate": min(s["estimate"] for s in ss),
                   "max_estimate": max(s["estimate"] for s in ss), "n_with_ci": len(withci),
                   "n_ci_excludes_0": sum(s["ci_excludes_0"] for s in withci),
                   "n_negative": sum(s["estimate"] < 0 for s in ss), "n_positive": sum(s["estimate"] > 0 for s in ss),
                   "n_ci_missing": len(ss) - len(withci)}
    out = {"status": "EXPLORATORY specification curve; assembled from existing outputs only, nothing re-estimated",
           "estimand": "Δ = HCI (CHI) rate − Nature rate of a positive citation-use label; negative = Nature > HCI",
           "units": "proportions (the figure shows percentage points)",
           "ci_note": "Panel A: cluster bootstrap over citation-graph components (10,000, seed 20261003); Panel B: "
                      "citing-paper bootstrap (2,000; seeds 20261003/20261012 per source). point_derived rows are the "
                      "difference of two stored rates and have no CI.",
           "excluded_on_purpose": ["D1 (exploratory, not a registered RQ4(a) system; in the RQ4(a) JSON as d1_*)",
                                   "Kevin v2 rows on B (235 edges; s2_consequence.json) — different frame",
                                   "50-paper fresh-sample classifier checks (no CHI contrast; different frame)"],
           "unsourced": [{"label": "Canonical (human) · realized, threshold = yes only",
                          "reason": "no HCI−Nature yes-only estimate on disk; the yes-only sensitivity exists only "
                                    "for RQ1(b) κ on B (analysis/results/*/rq1b.json sensitivity_realized_yes_only)"}],
           "summary": summ, "specifications": specs}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n")
    figure(specs, a.fig, a.png)
    print(json.dumps(summ, indent=1))
    print(f"wrote {a.out}\nwrote {a.fig}")


if __name__ == "__main__":
    main()
