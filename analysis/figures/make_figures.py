#!/usr/bin/env python3
"""Figures for the iConference manuscript, drawn from the B300 result files and script 60's derived numbers.

    python3 analysis/figures/make_figures.py [--out-dir DIR] [--derived PATH]


Colour follows the dataviz reference palette (validated light mode):
- constructs, everywhere: crediting = blue, uptake = orange, claim dependence = aqua (slots 1-3,
  all-pairs CVD dE 9.2), canonical realization = neutral ink;
- systems (Figures 3 and 6): the eight categorical slots in fixed order, every series direct-labelled
  because aqua, yellow and magenta sit below 3:1 on the surface.
"""
from __future__ import annotations

import json
import statistics as st
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Rectangle  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
R = ROOT / "analysis/results"
B300 = R / "sensitivity_2026-10-23_B300"
OUT = R / "paper_numbers/figures"  # overridden by --out-dir
DERIVED = R / "paper_numbers/derived_numbers.json"  # overridden by --derived

# palette (dataviz reference, light)
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948")
INK, INK2, INK3, GRID, SURFACE, NEUTRAL_BG = "#0b0b0b", "#52514e", "#8a8985", "#e6e5e1", "#ffffff", "#f0efec"
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
BLUE_TINT = "#e8f1fc"

CONSTRUCT = {"credits": (BLUE, "o", "Crediting"), "uptake": (ORANGE, "s", "Uptake"),
             "claim": (AQUA, "^", "Claim dependence"), "realized": (INK2, "D", "Realization")}
SYSTEM = {"gpt-6.1-sol": ("GPT-6.1 Sol", BLUE), "jev-1.13.0": ("Jev 1.13.0", ORANGE),
          "d1_distilled": ("D1 (distilled)", AQUA), "encoder_scibert": ("SciBERT encoder", YELLOW),
          "scicite": ("SciCite", MAGENTA), "s2_intents": ("S2 intents", GREEN),
          "multicite": ("MultiCite", VIOLET), "s2_influential": ("S2 influential", RED),
          "no_model_baseline": ("No-model baseline", INK3)}
FIELD = {"cited_contribution_type": "Contribution type", "citation_function": "Citation function",
         "credits_contribution": "Crediting", "uptake_mode": "Uptake mode", "uptake_degree": "Uptake degree",
         "claim_dependent": "Claim dependence"}

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 9, "axes.edgecolor": INK3, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK, "text.color": INK, "axes.spines.top": False,
    "axes.spines.right": False, "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0,
    "ytick.major.pad": 4, "legend.frameon": False, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.04, "pdf.fonttype": 42, "axes.titlesize": 9.5,
    "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.titlepad": 8,
})


def load(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png", dpi=200)
    plt.close(fig)
    print("wrote", name)


def xgrid(ax):
    ax.xaxis.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)


def ci_dot(ax, x, y, lo, hi, color, marker="o", size=6.5, hollow=False, lw=1.2, z=3, label=None):
    ax.errorbar(x, y, xerr=[[x - lo], [hi - x]], fmt="none", ecolor=color, elinewidth=lw, capsize=2.8,
                capthick=lw, zorder=z - 1)
    ax.plot(x, y, marker=marker, markersize=size, linestyle="", zorder=z, label=label,
            markerfacecolor=SURFACE if hollow else color, markeredgecolor=color if hollow else SURFACE,
            markeredgewidth=1.4 if hollow else 1.0)


def panel_tag(ax, tag):
    ax.set_title(tag, loc="left")


# ------------------------------------------------------------------ Figure 1: RQ1
def fig_rq1b():
    d = load(B300 / "rq1b.json")
    b240 = load(R / "confirmatory_2026-10-08/rq1b.json")["B"]["primary_realized_partial_or_yes"]
    bp, by = d["B"]["primary_realized_partial_or_yes"], d["B"]["sensitivity_realized_yes_only"]
    ap, ay = d["A250_replication"]["primary_realized_partial_or_yes"], d["A250_replication"]["sensitivity_realized_yes_only"]

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 2.7), gridspec_kw={"width_ratios": [1.05, 1], "wspace": 0.55})

    # (a) dumbbell: kappa with crediting vs uptake
    rows = [(f"B300 (n = {bp['n']})", bp), (f"A250 (n = {ap['n']})", ap), (f"B, registered 240 (n = {b240['n']})", b240)]
    for i, (lab, s) in enumerate(rows):
        c, u = s["kappa_realized_vs_crediting"], s["kappa_realized_vs_uptake"]
        a1.plot([c, u], [i, i], color=GRID, linewidth=5, solid_capstyle="round", zorder=1)
        ci_dot(a1, c, i - 0.14, *s["kappa_crediting_ci95"], BLUE, "o")
        ci_dot(a1, u, i + 0.14, *s["kappa_uptake_ci95"], ORANGE, "s")
        if i == 0:
            a1.annotate("specific crediting", (c, i - 0.14), xytext=(0, 9), textcoords="offset points",
                        ha="center", fontsize=7.5, color=INK2)
            a1.annotate("uptake", (u, i + 0.14), xytext=(0, -13), textcoords="offset points",
                        ha="center", fontsize=7.5, color=INK2)
    a1.set_yticks(range(len(rows)), [r[0] for r in rows])
    a1.invert_yaxis()
    a1.set_xlim(0.15, 0.6)
    a1.set_xlabel("Cohen's $\\kappa$ with canonical realization")
    panel_tag(a1, "(a) Agreement with each v2 construct")
    xgrid(a1)

    # (b) the registered contrast against the predicted direction
    rows = [("B300", bp, False, VIOLET), ("B300, yes-only", by, True, VIOLET),
            ("A250", ap, False, INK2), ("A250, yes-only", ay, True, INK2),
            ("B, registered 240", b240, False, INK2)]
    a2.axvspan(0, 0.45, color=BLUE_TINT, zorder=0)
    a2.text(0.43, -0.75, "predicted:\ncrediting > uptake", ha="right", va="top", fontsize=7.5, color=BLUE)
    a2.axvline(0, color=INK2, linewidth=0.8, zorder=1)
    for i, (lab, s, hollow, col) in enumerate(rows):
        v, (lo, hi) = s["kappa_difference_crediting_minus_uptake"], s["difference_ci95"]
        ci_dot(a2, v, i, lo, hi, col, "D", size=6, hollow=hollow)
        a2.annotate(f"{v:+.3f}".replace("-", "\u2212"), (hi, i), xytext=(4, 0), textcoords="offset points", va="center",
                    fontsize=7.5, color=INK2)
    a2.set_yticks(range(len(rows)), [r[0] for r in rows])
    a2.set_ylim(len(rows) - 0.5, -1.0)
    a2.set_xlim(-0.3, 0.45)
    a2.set_xlabel("$\\kappa$(crediting) $-$ $\\kappa$(uptake), 95% CI")
    panel_tag(a2, "(b) Registered contrast")
    xgrid(a2)
    save(fig, "fig_rq1b_kappa")


# ------------------------------------------------------------------ Figure 2: RQ2
def fig_rq2b():
    d = load(B300 / "rq2.json")
    nmi, cx = d["b_function_realization_nmi"], d["c_within_v2_contrasts_B"]
    v1 = nmi["transparency_same_250_edges_v1"]

    fig = plt.figure(figsize=(7.2, 2.55))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.35, 1, 1], wspace=0.75)
    a1 = fig.add_subplot(gs[0])

    bars = [("v1, A250", v1["nmi_v1_multiclass"], v1["ci95"], INK3),
            ("v2, A250", nmi["A250"]["nmi_geometric_v2"], nmi["A250"]["ci95"], BLUE),
            ("v2, B300", nmi["B_descriptive"]["nmi_geometric_v2"], nmi["B_descriptive"]["ci95"], BLUE_RAMP[2])]
    for i, (lab, v, ci, col) in enumerate(bars):
        a1.add_patch(FancyBboxPatch((0, i - 0.3), max(v, 0.004), 0.6, boxstyle="round,pad=0,rounding_size=0.012",
                                    facecolor=col, edgecolor="none", mutation_aspect=0.5))
        a1.plot(ci, [i, i], color=INK, linewidth=1.0, solid_capstyle="round")
        a1.annotate(f"{v:.2f}", (ci[1], i), xytext=(4, 0), textcoords="offset points", va="center",
                    fontsize=8, color=INK)
    diff = nmi["A250"]["difference_vs_0.700"]
    a1.annotate("", xy=(0.05, 0.72), xytext=(v1["nmi_v1_multiclass"] - 0.03, 0.36),
                arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=1.3, connectionstyle="arc3,rad=0.15"))
    a1.text(0.24, 1.0, f"{diff:+.2f} vs. registered 0.70".replace("-", "\u2212"), fontsize=7.5, color=ORANGE,
            ha="left", va="center")
    a1.set_yticks(range(len(bars)), [b[0] for b in bars])
    a1.set_ylim(2.6, -0.6)
    a1.set_xlim(0, 0.85)
    a1.set_xlabel("NMI, function vs. realized/uptake")
    panel_tag(a1, "(a) Function vs. use (NMI)")
    xgrid(a1)

    cmap = LinearSegmentedColormap.from_list("blue", BLUE_RAMP[:5])

    def crosstab(ax, cells, rows, cols, title, total):
        vmax = max(cells.values())
        for i, r in enumerate(rows):
            for j, c in enumerate(cols):
                n = cells[(r, c)]
                off = (i != j)
                face = "#fbe0d4" if off else NEUTRAL_BG
                ax.add_patch(Rectangle((j - 0.47, i - 0.47), 0.94, 0.94, facecolor=face,
                                       edgecolor=ORANGE if off else SURFACE, linewidth=2.0 if off else 0))
                dark = False
                ax.text(j, i - 0.08, f"{n}", ha="center", va="center", fontsize=10.5, fontweight="bold",
                        color=SURFACE if dark else INK)
                ax.text(j, i + 0.22, f"{100 * n / total:.0f}%", ha="center", va="center", fontsize=7,
                        color=SURFACE if dark else INK2)
        ax.set_xlim(-0.5, 1.5)
        ax.set_ylim(1.5, -0.5)
        ax.set_xticks([0, 1], ["uptake", "no uptake"])
        ax.xaxis.tick_top()
        ax.set_yticks([0, 1], [rows[0][1], rows[1][1]])
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(title, pad=22)

    cr = cx["credits_x_uptake"]
    total = sum(v for k, v in cr.items() if "unclear" not in k)
    a2 = fig.add_subplot(gs[1])
    crosstab(a2, {(("s", "specific"), "u"): cr["specific | uptake"], (("s", "specific"), "n"): cr["specific | none"],
                  (("g", "generic"), "u"): cr["generic | uptake"], (("g", "generic"), "n"): cr["generic | none"]},
             [("s", "specific"), ("g", "generic")], ["u", "n"], "(b) Crediting × uptake", total)
    cd = cx["claim_dependent_x_uptake"]
    a3 = fig.add_subplot(gs[2])
    crosstab(a3, {(("y", "dependent"), "u"): cd["yes | uptake"], (("y", "dependent"), "n"): cd["yes | none"],
                  (("n", "not\ndependent"), "u"): cd["no | uptake"], (("n", "not\ndependent"), "n"): cd["no | none"]},
             [("y", "dependent"), ("n", "not\ndependent")], ["u", "n"], "(c) Claim dependence × uptake", total)
    fig.text(0.985, -0.04, "Orange cells: the two fields disagree (B300, n = 297).", ha="right", fontsize=7.5,
             color=INK2)
    save(fig, "fig_rq2b_nmi")


# ------------------------------------------------------------------ Figure 3: RQ3
ONE_CLASS = {("kevin_v2@B", "uptake_degree", "jev-1.13.0"), ("kevin_v2@B", "citation_function", "s2_intents")}
REF_CANON, REF_V2 = "#a9a8a3", VIOLET  # canonical record = neutral, my v2 labels = violet (as in Figure 1)


def fig_rq3():
    sc = load(B300 / "rq3.json")["scores"]

    def get(ref, f, s):
        return sc.get(f"{ref}|{f}|{s}|primary")

    fig = plt.figure(figsize=(7.2, 7.9))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.0, 1.05], hspace=0.32, wspace=1.3)

    # (a, b) paired bars: each system scored against both references, with its rank under each
    for k, f in enumerate(("cited_contribution_type", "citation_function")):
        ax = fig.add_subplot(gs[0, 2 * k:2 * k + 2])
        systems = [s for s in SYSTEM if get("canonical@B", f, s) and get("kevin_v2@B", f, s)]
        canon = {s: get("canonical@B", f, s)["macro_f1"] for s in systems}
        v2 = {s: get("kevin_v2@B", f, s)["macro_f1"] for s in systems}
        rank_c = {s: i + 1 for i, s in enumerate(sorted(systems, key=lambda s: -canon[s]))}
        rank_v = {s: i + 1 for i, s in enumerate(sorted(systems, key=lambda s: -v2[s]))}
        order = sorted(systems, key=lambda s: rank_c[s])
        h = 0.36
        for i, s in enumerate(order):
            one = ("kevin_v2@B", f, s) in ONE_CLASS
            ax.barh(i - h / 2 - 0.02, canon[s], height=h, color=REF_CANON, edgecolor="none")
            ax.barh(i + h / 2 + 0.02, v2[s], height=h, color=SURFACE if one else REF_V2,
                    edgecolor=REF_V2, linewidth=1.0, hatch="////" if one else None)
            ax.text(canon[s] + 0.012, i - h / 2 - 0.02, f"{canon[s]:.2f}", va="center", fontsize=8, color=INK2)
            ax.text(v2[s] + 0.012, i + h / 2 + 0.02, f"{v2[s]:.2f}" + ("*" if one else ""), va="center",
                    fontsize=8, color=INK)
            moved = rank_c[s] != rank_v[s]
            ax.text(1.17, i, f"{rank_c[s]} \u2192 {rank_v[s]}", ha="center", va="center", fontsize=8.5,
                    color=REF_V2 if moved else INK3, fontweight="bold" if moved else "normal")
        ax.text(1.17, -0.85, "rank", ha="center", va="center", fontsize=8, color=INK2)
        ax.set_yticks(range(len(order)), [SYSTEM[s][0] + ("*" if ("kevin_v2@B", f, s) in ONE_CLASS else "")
                                          for s in order], fontsize=8.5)
        ax.set_ylim(len(order) - 0.45, -1.15)
        ax.set_xlim(0, 1.27)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_xlabel("Macro-F1", fontsize=8.5)
        tau = "; Kendall's $\\tau$ = 0.62" if f == "citation_function" else ""
        ax.set_title(f"({'ab'[k]}) {FIELD[f]}{tau}", fontsize=9.5)
        xgrid(ax)

    # (c) v2-only fields in one wide panel, grouped by field
    ax = fig.add_subplot(gs[1, :])
    y, ticks, labels = 0.0, [], []
    for f in ("credits_contribution", "uptake_mode", "uptake_degree", "claim_dependent"):
        ax.text(-0.012, y - 0.15, FIELD[f], ha="right", va="center", fontsize=8.8, fontweight="bold",
                transform=ax.get_yaxis_transform())
        y += 0.85
        systems = sorted([s for s in SYSTEM if get("kevin_v2@B", f, s)], key=lambda s: -get("kevin_v2@B", f, s)["macro_f1"])
        for s in systems:
            v = get("kevin_v2@B", f, s)
            one = ("kevin_v2@B", f, s) in ONE_CLASS
            ci_dot(ax, v["macro_f1"], y, *v["macro_f1_ci95"], REF_V2, "o", size=6, hollow=one)
            ax.text(v["macro_f1_ci95"][1] + 0.012, y, f"{v['macro_f1']:.2f}" + ("*" if one else ""), va="center",
                    fontsize=8, color=INK2)
            ticks.append(y)
            labels.append(SYSTEM[s][0])
            y += 1
        y += 0.55
    ax.set_yticks(ticks, labels, fontsize=8.5)
    ax.set_ylim(y - 0.9, -0.75)
    ax.set_xlim(0.2, 1.06)
    ax.set_xticks([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax.set_xlabel("Macro-F1 vs. my v2 labels (95% CI)", fontsize=8.5)
    ax.set_title("(c) Fields scored against v2 only", fontsize=9.5)
    xgrid(ax)

    from matplotlib.patches import Patch
    handles = [Patch(facecolor=REF_CANON, label="vs. canonical record"),
               Patch(facecolor=REF_V2, label="vs. my v2 labels (B300)"),
               Patch(facecolor=SURFACE, edgecolor=REF_V2, hatch="////", label="* only one class scorable")]
    fig.legend(handles=handles, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.0), fontsize=8.5,
               handlelength=1.6, columnspacing=1.8)
    fig.text(0.5, 0.0, "Hollow marker: one-class cell. Violet ranks changed between references.",
             ha="center", fontsize=8, color=INK2)
    fig.subplots_adjust(top=0.93, bottom=0.07)
    save(fig, "fig_rq3_scores")


# ------------------------------------------------------------------ Figure 4: RQ4(a)
def fig_rq4a():
    res = {(r["source"], r["construct"]): r for r in load(R / "exploratory/2026-10-11_rq4a_final.json")["results"]}
    groups = [("Canonical record", [("canonical", "realized")]),
              ("GPT-6.1 Sol", [("gpt-6.1-sol", c) for c in ("credits", "uptake", "claim")]),
              ("Jev 1.13.0", [("jev-1.13.0", c) for c in ("credits", "uptake", "claim")]),
              ("MultiCite", [("multicite", "uptake")]),
              ("S2 influential", [("s2_influential", "claim")])]
    seeds = [("Encoder, unconditioned", "encoder_uncond"), ("Encoder, conditioned", "encoder_cond")]

    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    y, ticks, labels = 0, [], []
    lo_x, hi_x = -0.30, 0.06
    for g, items in groups:
        ax.text(-0.012, y - 0.75, g, ha="right", va="center", fontsize=8.3, fontweight="bold", color=INK,
                transform=ax.get_yaxis_transform())
        y += 0.25
        for key in items:
            r = res[key]
            col, m, lab = CONSTRUCT[key[1]]
            ns = r["ci95"][0] <= 0 <= r["ci95"][1]
            ci_dot(ax, r["delta"], y, *r["ci95"], col, m, hollow=ns)
            ticks.append(y)
            labels.append(lab.lower())
            y += 1
        y += 1.2
    # seed strips: each seed is a dot; the bracket spans the seed range
    canon_frozen = res[("canonical@frozen", "realized")]
    ax.text(-0.012, y - 0.75, "SciBERT encoder (5 seeds each)", ha="right", va="center", fontsize=8.3,
            fontweight="bold", color=INK, transform=ax.get_yaxis_transform())
    header_y = y - 0.75
    y += 0.25
    for name, prefix in seeds:
        vals = [res[(f"{prefix}_s{i}", "realized")] for i in range(1, 6)]
        ds = [v["delta"] for v in vals]
        ax.plot([min(ds), max(ds)], [y, y], color=YELLOW, linewidth=6, alpha=0.35, solid_capstyle="round")
        for v in vals:
            ns = v["ci95"][0] <= 0 <= v["ci95"][1]
            ax.plot(v["delta"], y, "o", markersize=6, markerfacecolor=SURFACE if ns else YELLOW,
                    markeredgecolor="#b07800" if ns else SURFACE, markeredgewidth=1.3 if ns else 1)
        ax.annotate(f"range {max(ds) - min(ds):.2f}", (min(ds), y), xytext=(-9, 0), textcoords="offset points",
                    va="center", ha="right", fontsize=7.3, color=INK2)
        ticks.append(y)
        labels.append(name.split(", ")[1])
        y += 1
    cf = canon_frozen["delta"]
    ax.plot([cf, cf], [header_y + 0.3, y - 0.45], color=INK, linewidth=1.0, linestyle=(0, (2, 2)), zorder=1)
    ax.text(cf - 0.004, header_y, f"canonical on the same edges: {cf:.3f}".replace("-", "\u2212"),
            ha="right", va="center", fontsize=7.5, color=INK2)

    ax.axvline(0, color=INK, linewidth=0.9)
    ax.axvspan(lo_x, 0, color=BLUE_TINT, zorder=0)
    ax.text(lo_x + 0.005, -1.9, "← higher positive rate in Nature OA", fontsize=7.5, color=BLUE, va="center")
    ax.text(0.003, -1.9, "HCI higher →", fontsize=7.5, color=INK2, va="center")
    ax.set_yticks(ticks, labels, fontsize=8)
    ax.set_ylim(y - 0.2, -2.5)
    ax.set_xlim(lo_x, hi_x)
    ax.set_xlabel("HCI $-$ Nature OA difference in positive-label rate (95% CI)")
    xgrid(ax)
    handles = [plt.Line2D([], [], marker=CONSTRUCT[c][1], color=CONSTRUCT[c][0], linestyle="", markersize=6,
                          markeredgecolor=SURFACE, label=CONSTRUCT[c][2]) for c in ("realized", "credits", "uptake", "claim")]
    handles.append(plt.Line2D([], [], marker="o", color=INK2, markerfacecolor=SURFACE, linestyle="", markersize=6,
                              label="CI includes 0"))
    ax.legend(handles=handles, loc="upper center", fontsize=7.5, ncol=5, bbox_to_anchor=(0.42, -0.13),
              handletextpad=0.3, columnspacing=1.2)
    save(fig, "fig_rq4a_delta")


# ------------------------------------------------------------------ Figure 6: system consistency
def fig_stability():
    sol = load(R / "external_systems/hosted_llm_codebook_v2/stability_s1_vs_s2.json")["label_agreement"]
    jev = load(R / "external_systems/jev_codebook_v2/compare_s1_s2.json")
    rerun = load(DERIVED)["rerun_consistency"]
    s2 = rerun["s2"]["s2_edge_labels.csv"]
    fields = list(FIELD)
    rows = {"GPT-6.1 Sol": {f: float(sol[f].split("=")[1]) for f in fields},
            "Jev 1.13.0": {f: jev[f]["label_agreement"] for f in fields},
            # SciCite, MultiCite, S2: rerun agreement on the same 100 edges, from script 60; each covers one field.
            "SciCite": {"citation_function": rerun["scicite"]["identical"] / rerun["scicite"]["edges"]},
            "MultiCite": {"uptake_mode": rerun["multicite"]["identical"] / rerun["multicite"]["edges"]},
            "S2 intents / influential": {f: s2["identical"] / s2["edges"] for f in ("citation_function", "claim_dependent")}}
    cmap = LinearSegmentedColormap.from_list("blue", BLUE_RAMP)
    fig, ax = plt.subplots(figsize=(6.6, 2.6))
    for i, (name, vals) in enumerate(rows.items()):
        for j, f in enumerate(fields):
            if f not in vals:
                ax.add_patch(Rectangle((j - 0.47, i - 0.45), 0.94, 0.9, facecolor=NEUTRAL_BG, edgecolor=SURFACE))
                ax.text(j, i, "–", ha="center", va="center", color=INK3, fontsize=8)
                continue
            v = vals[f]
            t = max(0.0, min(1.0, (v - 0.65) / 0.35))
            low = v < 0.85
            ax.add_patch(Rectangle((j - 0.47, i - 0.45), 0.94, 0.9, facecolor=cmap(t),
                                   edgecolor=ORANGE if low else SURFACE, linewidth=2.2 if low else 1.5))
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8.5, fontweight="bold" if low else "normal",
                    color=SURFACE if t > 0.55 else INK)
    ax.set_xlim(-0.5, len(fields) - 0.5)
    ax.set_ylim(len(rows) - 0.5, -0.5)
    ax.set_xticks(range(len(fields)), [FIELD[f] for f in fields], fontsize=8)
    ax.xaxis.tick_top()
    ax.set_yticks(range(len(rows)), list(rows))
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    ax.set_xlabel("Share of identical labels across two runs (100 B edges). Orange outline: below 0.85.",
                  fontsize=7.5, color=INK2)
    save(fig, "fig_stability")


# ------------------------------------------------------------------ Figure 5: D1 fidelity
def fig_d1():
    seeds = [load(R / f"ml/d1_2026-10-07/d1_meta_seed{s}.json")["heldout_fidelity_vs_sol"] for s in (1, 2, 3)]
    sol = load(R / "external_systems/hosted_llm_codebook_v2/stability_s1_vs_s2.json")["label_agreement"]
    rows = []
    for f in FIELD:
        vals = [s[f]["agreement_with_sol"] for s in seeds]
        rows.append((f, st.mean(vals), min(vals), max(vals), float(sol[f].split("=")[1])))
    rows.sort(key=lambda r: r[4] - r[1])  # smallest gap to Sol's own ceiling first
    fig, ax = plt.subplots(figsize=(6.0, 2.7))
    for i, (f, m, lo, hi, ceil) in enumerate(rows):
        ax.add_patch(FancyBboxPatch((0.5, i - 0.32), ceil - 0.5, 0.64, boxstyle="round,pad=0,rounding_size=0.01",
                                    facecolor=BLUE_TINT, edgecolor="none", mutation_aspect=0.4))
        ax.add_patch(FancyBboxPatch((0.5, i - 0.17), m - 0.5, 0.34, boxstyle="round,pad=0,rounding_size=0.01",
                                    facecolor=AQUA, edgecolor="none", mutation_aspect=0.4))
        ax.plot([lo, hi], [i, i], color=INK, linewidth=1.0)
        ax.plot([ceil, ceil], [i - 0.36, i + 0.36], color=BLUE, linewidth=2.2, solid_capstyle="round")
        ax.annotate(f"{m:.2f}", (0.5, i), xytext=(4, 0), textcoords="offset points", va="center", fontsize=7.5,
                    color=SURFACE, fontweight="bold")
        ax.annotate(f"gap {ceil - m:.2f}", (ceil, i), xytext=(6, 0), textcoords="offset points", va="center",
                    fontsize=7.5, color=INK2)
    ax.set_yticks(range(len(rows)), [FIELD[r[0]] for r in rows])
    ax.invert_yaxis()
    ax.set_xlim(0.5, 1.07)
    ax.set_xlabel("Label agreement")
    xgrid(ax)
    handles = [Rectangle((0, 0), 1, 1, facecolor=AQUA, label="D1 vs. Sol, 195 held-out edges (3-seed mean; whisker = seed range)"),
               plt.Line2D([], [], color=BLUE, linewidth=2.2, label="Sol vs. itself, two runs (100 edges)")]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.45, 1.0), ncol=1, fontsize=7.5)
    save(fig, "fig_d1_fidelity")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Figures for the iConference manuscript, from the result files.")
    ap.add_argument("--out-dir", type=Path, default=OUT)
    ap.add_argument("--derived", type=Path, default=DERIVED)
    a = ap.parse_args()
    OUT, DERIVED = a.out_dir, a.derived
    OUT.mkdir(parents=True, exist_ok=True)
    fig_rq1b()
    fig_rq2b()
    fig_rq3()
    fig_rq4a()
    fig_stability()
    fig_d1()
