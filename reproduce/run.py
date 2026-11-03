#!/usr/bin/env python3
"""Regenerate the results reported in the iConference 2027 manuscript and check them against the stored outputs.

    python3 reproduce/run.py list                       # stages, tiers, outputs
    python3 reproduce/run.py check-inputs [--hashes]    # private inputs and frozen model outputs present (and identical)?
    python3 reproduce/run.py verify                     # fast tier: rebuild in a scratch workspace and compare
    python3 reproduce/run.py verify --stage b300 --stage figures
    python3 reproduce/run.py verify --tier expensive --stage d1     # hours; see reproduce/README.md
    python3 reproduce/run.py run --inplace --stage ...  # write into this repository (fresh copy without outputs)
    REPRO_DATA_DIR=/path/to/unzipped/data python3 reproduce/run.py verify   # use a data package unpacked elsewhere

`verify` never writes into the repository. It builds reproduce_out/work/: every directory that will receive an
output is a real directory, code files are copied (so scripts that locate the repository from their own path
resolve to the workspace), and everything else is a symlink to the original inputs. The stage's outputs are absent
there, so each script runs exactly as on a fresh checkout; the regenerated files are then compared with the stored
ones. A final check confirms that nothing under the original repository changed during the run.
"""
from __future__ import annotations

import argparse
import datetime as dt
import filecmp
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTROOT = ROOT / "reproduce_out"
PY = os.environ.get("PYTHON", sys.executable)
X, ML, ES, R = "analysis/experiments", "analysis/ml", "analysis/external_systems", "analysis/results"
ND, NR = "external_data/nature_arm_2026-10-04", "external_data/nature_repair_v2_2026-10-25"

# ------------------------------------------------------------------------------------------------ stages
# outputs: paths (files or directories) the stage creates. `ignore`: outputs that legitimately differ between runs
# (dates, timings, absolute paths) and are not compared. Order = dependency order.
STAGES = [
    # ---- fast tier: offline, CPU, seconds to minutes, deterministic
    dict(name="extract_nature", tier="fast", desc="GROBID TEI -> Nature citation edges (70,400)",
         cmd=[PY, "analysis/extraction/extract_citances_grobid.py", f"{ND}/pdfs_grobid", f"{ND}/citances/nature_arm_citances.jsonl"],
         outputs=[f"{ND}/citances/nature_arm_citances.jsonl"]),
    dict(name="extract_jats_parity", tier="fast", desc="JATS XML -> parity citation edges (200 papers)",
         cmd=[PY, "analysis/extraction/extract_citances_jats.py", f"{ND}/jats", f"{ND}/parity/jats_citances_200.jsonl",
              "--manifest", f"{ND}/frame/manifest.csv"],
         outputs=[f"{ND}/parity/jats_citances_200.jsonl"]),
    dict(name="parity_gate", tier="fast", desc="RQ4(b) parity gate: 86.1% recovery, 82.5% edge match",
         cmd=[PY, "analysis/nature_arm/07_parity_gate.py"],
         outputs=[f"{ND}/parity/parity_gate_result.json", f"{ND}/parity/parity_gate_edges.csv"]),
    dict(name="splits", tier="fast", desc="component-aware splits with all human-coded edges frozen (914/197/195)",
         cmd=[PY, f"{ML}/02_make_splits.py", "--derived-dir", "analysis/derived", "--frozen", "annotations/generated",
              "--frozen", "annotations/generated/expansion_2026-09-16", "--frozen",
              "private/history/citation-annotation-2027/templates", "--out-dir", f"{ML}/splits"],
         outputs=[f"{ML}/splits/hci_splits.csv", f"{ML}/splits/nature_oa_splits.csv", f"{ML}/splits/split_manifest.json"]),
    dict(name="hosted_input", tier="fast", desc="system input: 2,937 canonical edges minus 8 anchors = 2,929",
         cmd=[PY, f"{ES}/hosted_llm_codebook_v2/make_input.py"],
         outputs=[f"{R}/external_systems/hosted_llm_codebook_v2/input_canonical.csv"]),
    dict(name="convert_systems", tier="fast", desc="SciCite/S2/MultiCite raw outputs -> scoring inputs (18, 1,316, 2,187)",
         cmd=[PY, f"{X}/26_convert_system_outputs.py", "--tag", "canonical_20261004"],
         outputs=[f"{R}/external_systems/scoring_inputs_canonical_20261004"],
         csv_ignore=["system"],
         note="the stored SciCite/S2 files were relabelled by hand after generation (system = scicite_all, "
              "s2_intents_coarse, ...); every label is identical and the scorer does not read that column"),
    dict(name="no_model_baseline", tier="fast", desc="majority-class baseline",
         cmd=[PY, f"{X}/25_no_model_baseline.py", "--out", f"{R}/external_systems/no_model_baseline/no_model_baseline.csv"],
         outputs=[f"{R}/external_systems/no_model_baseline/no_model_baseline.csv"]),
    dict(name="llm_judges_vs_canonical", tier="fast", desc="exploratory: Sol +0.163, Jev -0.130 (Δκ)",
         cmd=[PY, f"{X}/23_exploratory_construct_tracking.py", "--out", f"{R}/exploratory/2026-10-04_construct_tracking.json"],
         outputs=[f"{R}/exploratory/2026-10-04_construct_tracking.json"]),
    dict(name="confirmatory_240", tier="fast", desc="registered 240-edge analysis (RQ1-RQ3, D1)",
         cmd=[PY, f"{X}/29_confirmatory_2026_10_08.py"], outputs=[f"{R}/confirmatory_2026-10-08"],
         not_scripted={"rq3_robustness_excluding_degenerate.json": "written by hand on 8 Oct for the 240-edge "
                       "analysis; not used by the paper (the B300 equivalent is regenerated by b300_companions)"}),
    dict(name="b300", tier="fast", desc="primary B300 analysis: RQ1, RQ2, RQ3, D1",
         cmd=[PY, f"{X}/38_sensitivity_B300.py"], outputs=[f"{R}/sensitivity_2026-10-23_B300"]),
    dict(name="calibration_240", tier="fast", desc="calibration on the 240-edge reference (input to companions)",
         cmd=[PY, f"{X}/32_calibration.py"], outputs=[f"{R}/calibration_2026-10-12"]),
    dict(name="qwen_addendum", tier="fast", desc="Qwen addendum on the registered reference (197/550 pairing failures)",
         cmd=[PY, f"{X}/33_qwen_addendum.py"], outputs=[f"{R}/confirmatory_2026-10-12_with_qwen"],
         extra_writes=["docs/results"]),
    dict(name="b300_companions", tier="fast", desc="B300 companions: Qwen, one-class robustness, calibration, JEPA, ACL-ARC",
         cmd=[PY, f"{X}/39_B300_manuscript_companions.py"], outputs=[f"{R}/reanalysis_2026-10-23_B300_companions"]),
    dict(name="model_disagreement", tier="fast", desc="exploratory: Sol/Jev/Qwen non-unanimity (dependence 83.5%, uptake 37.7%)",
         cmd=[PY, f"{X}/58_model_disagreement_B297_primary.py"],
         outputs=[f"{R}/exploratory/2026-10-25_model_disagreement_B297_primary.json"]),
    dict(name="rq4a_inputs", tier="fast", desc="RQ4(a) encoder/D1 inputs",
         cmd=[PY, f"{X}/30_rq4a_final_inputs.py"], outputs=[f"{R}/external_systems/scoring_inputs_rq4a_final_20261011"]),
    dict(name="rq4a", tier="fast", desc="RQ4(a) HCI-Nature differences, final 11 Oct lineup (minutes)",
         cmd=["@rq4a"], outputs=[f"{R}/exploratory/2026-10-11_rq4a_final.json"]),
    dict(name="s2_consequence", tier="fast", desc="S2 influential on 2,917 edges (+0.051)",
         cmd=[PY, f"{X}/34_s2_consequence.py"], outputs=[f"{R}/exploratory/2026-10-12_s2_consequence.json"]),
    dict(name="multicite_at_scale_summary", tier="fast", desc="MultiCite at scale: CHI 11.75%",
         cmd=[PY, f"{X}/28_multicite_at_scale_summary.py", "--out", f"{R}/exploratory/2026-10-07_multicite_at_scale.json"],
         outputs=[f"{R}/exploratory/2026-10-07_multicite_at_scale.json"]),
    dict(name="recovered_nature", tier="fast", desc="recovered Nature subset: 10,138/12,298; 17.41% vs 11.75%",
         cmd=[PY, f"{X}/35_recovered_nature_edges.py"], outputs=[f"{R}/exploratory/2026-10-12_recovered_nature_edges.json"]),
    dict(name="parser_bias", tier="fast", desc="JATS vs GROBID parser diagnostic (needs the system runs in tier expensive)",
         cmd=[PY, "analysis/nature_arm/08_parser_bias.py"], outputs=[f"{ND}/parser_bias/parser_bias_result.json"]),
    dict(name="d3_audit", tier="fast", desc="D3 soft vs hard labels: ECE/Brier from saved probabilities",
         cmd=[PY, f"{X}/59_audit_d3_aggregation.py"], outputs=[f"{R}/audits/2026-10-26_d3_aggregation.json"]),
    dict(name="function_retest_stability", tier="fast", desc="registered function retest, 150 items (kappa 0.22 without unclear)",
         cmd=[PY, f"{X}/63_function_retest_stability.py"], outputs=[f"{R}/stability/function_retest_2026-10-31.json"]),
    dict(name="function_retest_drift", tier="fast", desc="post hoc: function retest vs canonical and own v2 coding (prints only)",
         cmd=[PY, f"{X}/65_function_retest_drift_posthoc.py"], outputs=[]),
    dict(name="stratum_pass1_stability", tier="fast", desc="registered stratum retest, Pass 1 fields (function kappa 0.83)",
         cmd=[PY, f"{X}/64_stratum_pass1_retest_stability.py"], outputs=[f"{R}/stability/stratum_pass1_retest_2026-11-01.json"]),
    dict(name="stratum_pass2_stability", tier="fast", desc="registered stratum retest, Pass 2 fields (uptake mode kappa 0.70)",
         cmd=[PY, f"{X}/66_stratum_pass2_retest_stability.py"], outputs=[f"{R}/stability/stratum_pass2_retest_2026-11-02.json"]),
    dict(name="rq3_retest_noise", tier="fast", desc="exploratory: RQ3 reference flips under measured retest noise (5/27)",
         cmd=[PY, f"{X}/67_rq3_under_retest_noise.py"], outputs=[f"{R}/stability/rq3_under_retest_noise_2026-11-02.json"]),
    dict(name="rq3_like_for_like", tier="fast", desc="post hoc: RQ3 reference reversals on common edges and classes (0/13 without D1)",
         cmd=[PY, f"{X}/68_rq3_like_for_like.py"], outputs=[f"{R}/rq3_like_for_like_2026-11-02.json"]),
    dict(name="rq3_constructs_like_for_like", tier="fast", desc="post hoc: RQ3 construct reversals on common edges (6/12 without D1)",
         cmd=[PY, f"{X}/69_rq3_constructs_like_for_like.py"], outputs=[f"{R}/rq3_constructs_like_for_like_2026-11-03.json"]),
    dict(name="rq1_detectable_difference", tier="fast", desc="post hoc: RQ1 contrast precision (80%-power detectable difference 0.19)",
         cmd=[PY, f"{X}/71_rq1_detectable_difference.py"], outputs=[f"{R}/rq1_detectable_difference_2026-11-03.json"]),
    dict(name="spec_curve", tier="fast", desc="exploratory: specification curve of the HCI-Nature gap (29 specifications)",
         cmd=[PY, f"{X}/70_specification_curve.py", "--fig", f"{R}/paper_numbers/figures/fig_spec_curve.pdf"],
         outputs=[f"{R}/spec_curve_2026-11-03.json"]),
    dict(name="derived_numbers", tier="fast", desc="hand-computed numbers: 3/50, flips, 147/150, 281/61/61, 52/60 ...",
         cmd=[PY, f"{X}/60_paper_derived_numbers.py"], outputs=[f"{R}/paper_numbers/derived_numbers.json"]),
    dict(name="tables", tier="fast", desc="LaTeX tables: samples, confirmatory, RQ3 scores and support, RQ4(a)",
         cmd=[PY, f"{X}/61_paper_tables.py"], outputs=[f"{R}/paper_numbers/tables"]),
    dict(name="figures", tier="fast", desc="manuscript figures 1-6 (PDF and PNG)",
         cmd=[PY, "analysis/figures/make_figures.py"], outputs=[f"{R}/paper_numbers/figures"], compare="png"),

    # ---- expensive tier: deterministic on CPU but hours; separate environments (reproduce/README.md)
    dict(name="encoder", tier="expensive", desc="SciBERT encoder, function and realization, 5 seeds (about 10 h CPU)",
         cmd=["@encoder"], outputs=[f"{ML}/datasets", f"{R}/ml/frozen_predictions"], reconstructed=True),
    dict(name="d1", tier="expensive", desc="D1 distilled encoder, 3 seeds (about 2 h CPU)",
         cmd=["@d1"], outputs=[f"{R}/ml/d1_2026-10-07"]),
    dict(name="d3", tier="expensive", desc="D3 soft-label training, 5 folds x 3 seeds (about 1.8 h CPU)",
         cmd=[PY, f"{ML}/11_d3_soft_labels.py"], outputs=[f"{R}/ml/d3_2026-10"]),
    dict(name="multicite_at_scale", tier="expensive", desc="MultiCite on 226,793 CHI + 70,400 Nature edges (about 6.6 h; aclarc_multicite venv)",
         cmd=["bash", f"{X}/27_multicite_at_scale.sh"], outputs=[f"{R}/external_systems/aclarc_multicite/at_scale"]),
    dict(name="parser_bias_systems", tier="expensive", desc="MultiCite + SciCite on JATS and GROBID parity text (about 3.5 h; two venvs)",
         cmd=["bash", "analysis/nature_arm/run_parser_bias_systems.sh"], outputs=[f"{ND}/parser_bias/run_log.txt"]),
    dict(name="nature_repair_extract", tier="expensive", desc="repaired Nature extraction of 1,200 PDFs (about 8 min; PyMuPDF 1.26.7)",
         cmd=[PY, f"{X}/46_nature_parallel_extract.py", "--parser", f"{X}/42d_nature_pdf_reconstruction_v6.py",
              "--frozen-hash", "9120b0f26d325eaf6ac7bb7bf14570e394198e97b5082002dfa79878a3129fbe",
              "--papers", f"{NR}/full/papers_1200.csv", "--out", "reproduce_out/nature_repair_v6.jsonl", "--workers", "4"],
         outputs=["reproduce_out/nature_repair_v6.jsonl"]),

    # ---- provider tier: network and credentials; outputs are not bit-reproducible, the frozen files are the record
    dict(name="sol", tier="provider", desc="GPT-6.1 Sol on Azure (AZURE_FOUNDRY_* in .env)",
         cmd=[PY, f"{ES}/hosted_llm_codebook_v2/run_hosted.py", "edges", "--input",
              f"{R}/external_systems/hosted_llm_codebook_v2/input_canonical.csv", "--tag", "canonical", "--profile",
              "reasoning", "--reasoning-effort", "low", "--workers", "8"], outputs=[]),
    dict(name="jev", tier="provider", desc="Jev 1.13.0 via TypeSafe (JEV_API_URL / JEV_API_KEY)",
         cmd=[PY, f"{ES}/jev_codebook_v2/run_jev.py", "edges", "--tag", "canonical"], outputs=[]),
    dict(name="s2", tier="provider", desc="Semantic Scholar snapshot (S2_API_KEY)", cmd=["bash", f"{ES}/s2_influential/run_all.sh"], outputs=[]),
    dict(name="qwen", tier="provider", desc="Qwen2.5-7B via Ollama 0.40.0, digest 845dbda0", cmd=["bash", f"{ES}/local_llm_codebook_v2/run_v2_registered.sh"], outputs=[]),
    dict(name="nature_download", tier="provider", desc="Nature frame and PDFs from Europe PMC / PMC S3",
         cmd=["bash", "-c", f"{PY} analysis/nature_arm/01_build_frame.py && {PY} analysis/nature_arm/02_download.py"], outputs=[]),
]

# Inputs that must exist before the fast tier can run: private data and frozen upstream outputs.
INPUTS = [
    "analysis/derived/hci_contribution_coding_human_reviewed.csv", "analysis/derived/nature_oa_contribution_coding_human_reviewed.csv",
    "yiren_artifacts", "annotations/label_crosswalk.csv", "annotations/coders/expansion_2026-09-26",
    "annotations/generated/v2_coding_2026-10-02", "annotations/generated/v2_prep/rationale_check_50.csv",
    "annotations/generated/expansion_2026-09-16", "annotations/generated/codebook_v2_anchors.json",
    "annotations/generated/retest_2026-10-24", "annotations/generated/retest_2026-10-31",
    "private/history/citation-annotation-2027/templates", f"{ML}/splits_pre_expansion_2026-09-29",
    f"{R}/external_systems/hosted_llm_codebook_v2/edges_canonical.csv", f"{R}/external_systems/jev_codebook_v2/edges_canonical.csv",
    f"{R}/external_systems/scicite_local/outputs", f"{R}/external_systems/aclarc_multicite/outputs",
    f"{R}/external_systems/s2_influential/outputs", f"{R}/external_systems/local_llm_codebook_v2",
    f"{R}/ml/frozen_predictions", f"{R}/ml/d1_2026-10-07", f"{R}/ml/d3_2026-10", f"{R}/ml/d2_jepa",
    f"{ND}/pdfs_grobid", f"{ND}/jats", f"{ND}/frame/manifest.csv", NR,
]

# Directories where scripts may create files beyond their declared outputs; mirrored so writes stay in the workspace.
ALWAYS_MIRROR = ["docs/results", f"{R}/exploratory", f"{R}/paper_numbers", f"{ND}/parity", f"{ND}/parser_bias"]
CODE_SUFFIXES = {".py", ".sh"}
SKIP_TOP = {".git", "reproduce_out", "archive", "local", "paper_sts", "paper_iconf", "citation_corpus_fulltext"}


def special(cmd0: str) -> list[list[str]]:
    if cmd0 == "@rq4a":
        extras = [f"{R}/external_systems/scoring_inputs_canonical_20261004/multicite_uptake.csv",
                  f"{R}/external_systems/scoring_inputs_canonical_20261004/s2_influential_claim.csv"]
        extras += sorted(str(p.relative_to(WORK_ROOT)) for p in
                         (WORK_ROOT / f"{R}/external_systems/scoring_inputs_rq4a_final_20261011").glob("*.csv"))
        return [[PY, f"{X}/24_rq4a_hci_vs_nature.py", "--out", f"{R}/exploratory/2026-10-11_rq4a_final.json",
                 "--n-boot", "10000", "--seed", "20261003", "--extra", *extras]]
    if cmd0 == "@encoder":
        cmds = [[PY, f"{ML}/03_build_datasets.py", "--include-frozen"]]
        for task in ("function", "realization"):
            for seed in range(1, 6):
                cmds.append([PY, f"{ML}/06_train_encoder.py", "--train-corpus", "pooled", "--encoder", "scibert",
                             "--task", task, "--seed", str(seed), "--predict-frozen", "--tag", "sweep1004"])
        return cmds
    if cmd0 == "@d1":
        return [[PY, f"{ML}/10_train_distilled.py", "--seed", str(s), "--out", f"{R}/ml/d1_2026-10-07"] for s in (1, 2, 3)]
    raise SystemExit(f"unknown special command {cmd0}")


WORK_ROOT = ROOT  # set by build_workspace for verify


# ------------------------------------------------------------------------------------------------ workspace
def build_workspace(work: Path, exclude: set[str], mirror: set[str]) -> None:
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    needs_real = {Path(p) for p in exclude | mirror}
    ancestors = {a for p in needs_real for a in p.parents} | {Path(m) for m in mirror}
    # Optional data overlay (REPRO_DATA_DIR): files there replace the repository's copies at the same relative path,
    # e.g. to verify against a data package unpacked elsewhere.
    overlay = Path(os.environ["REPRO_DATA_DIR"]).resolve() if os.environ.get("REPRO_DATA_DIR") else None
    if overlay:
        for d in overlay.rglob("*"):
            if d.is_dir():
                rel_d = d.relative_to(overlay)
                ancestors |= {rel_d, *rel_d.parents}

    def place(rel: Path):
        src, dst = ROOT / rel, work / rel
        if str(rel) in exclude:
            return
        if overlay and (overlay / rel).is_file():
            dst.symlink_to(overlay / rel)
            return
        if src.is_dir() and not src.is_symlink() and (rel in ancestors or has_code(rel)):
            dst.mkdir(exist_ok=True)
            for child in sorted(src.iterdir()):
                if child.name == "__pycache__":
                    continue
                place(rel / child.name)
        elif src.is_file() and src.suffix in CODE_SUFFIXES:
            shutil.copy2(src, dst)
        else:
            dst.symlink_to(src)

    for top in sorted(ROOT.iterdir()):
        if top.name in SKIP_TOP or top.name.startswith(".") and top.name != ".env":
            continue
        place(Path(top.name))


_code_cache: dict[Path, bool] = {}


def has_code(rel: Path) -> bool:
    """Directories under analysis/ and reproduce/ are mirrored so that copied scripts find a real tree."""
    return rel.parts[0] in ("analysis", "reproduce") and not (len(rel.parts) >= 2 and rel.parts[1] == "results")


# ------------------------------------------------------------------------------------------------ compare
def compare(stage: dict, work: Path) -> tuple[list[str], list[str]]:
    problems, notes = [], []
    for out in stage["outputs"]:
        a, b = ROOT / out, work / out
        if not b.exists():
            problems.append(f"missing: {out}")
            continue
        if not a.exists():
            problems.append(f"no stored copy to compare: {out}")
            continue
        files = [Path(out)] if b.is_file() else [p.relative_to(work) for p in sorted(b.rglob("*")) if p.is_file()]
        for rel in files:
            fa, fb = ROOT / rel, work / rel
            if stage.get("csv_ignore") and rel.suffix == ".csv" and fa.exists():
                if csv_equal(fa, fb, stage["csv_ignore"]):
                    notes.append(f"equivalent except column(s) {stage['csv_ignore']}: {rel}")
                    continue
            if stage.get("compare") == "png" and rel.suffix == ".pdf":
                continue  # PDF metadata carries a timestamp; the PNG renders are compared instead
            if not fa.exists():
                problems.append(f"extra file not in stored outputs: {rel}")
            elif not filecmp.cmp(fa, fb, shallow=False):
                problems.append(f"differs: {rel}{json_diff(fa, fb)}")
        if b.is_dir():
            for p in sorted(a.rglob("*")):
                if p.is_file() and not (work / p.relative_to(ROOT)).exists():
                    why = stage.get("not_scripted", {}).get(p.name)
                    (notes if why else problems).append(f"stored file not regenerated: {p.relative_to(ROOT)}"
                                                        + (f" ({why})" if why else ""))
    return problems, notes


def csv_equal(a: Path, b: Path, ignore: list[str]) -> bool:
    import csv
    def rows(p):
        with p.open(newline="", encoding="utf-8") as fh:
            return [{k: v for k, v in r.items() if k not in ignore} for r in csv.DictReader(fh)]
    return rows(a) == rows(b)


def json_diff(a: Path, b: Path) -> str:
    if a.suffix != ".json":
        return ""
    try:
        ja, jb = json.loads(a.read_text()), json.loads(b.read_text())
    except Exception:
        return ""
    diffs = []

    def walk(x, y, path):
        if len(diffs) >= 5:
            return
        if isinstance(x, dict) and isinstance(y, dict):
            for k in sorted(set(x) | set(y)):
                walk(x.get(k), y.get(k), f"{path}.{k}")
        elif isinstance(x, list) and isinstance(y, list) and len(x) == len(y):
            for i, (u, v) in enumerate(zip(x, y)):
                walk(u, v, f"{path}[{i}]")
        elif x != y:
            diffs.append(f"{path}: {str(x)[:60]} -> {str(y)[:60]}")
    walk(ja, jb, "")
    return (" (" + "; ".join(diffs) + ")") if diffs else " (same JSON content, different bytes)"


# ------------------------------------------------------------------------------------------------ guard
GUARD_DIRS = ["analysis", "annotations", "docs", "private", "yiren_artifacts", f"{ND}/parity", f"{ND}/parser_bias",
              f"{ND}/citances", NR]


def changed_since(t0: float) -> list[str]:
    hits = []
    for d in GUARD_DIRS:
        for dirpath, dirnames, filenames in os.walk(ROOT / d):
            dirnames[:] = [x for x in dirnames if x != "__pycache__"]
            for f in filenames:
                p = Path(dirpath) / f
                try:
                    if p.stat().st_mtime > t0:
                        hits.append(str(p.relative_to(ROOT)))
                except FileNotFoundError:
                    pass
    return hits


# ------------------------------------------------------------------------------------------------ commands
def select(args) -> list[dict]:
    st = [s for s in STAGES if (args.stage and s["name"] in args.stage) or (not args.stage and s["tier"] == args.tier)]
    unknown = set(args.stage or []) - {s["name"] for s in STAGES}
    if unknown:
        raise SystemExit(f"unknown stage(s): {sorted(unknown)}; see `reproduce/run.py list`")
    return st


def run_stage(stage: dict, cwd: Path, env: dict, log) -> int:
    cmds = special(stage["cmd"][0]) if stage["cmd"][0].startswith("@") else [stage["cmd"]]
    for cmd in cmds:
        log.write(f"$ {' '.join(cmd)}\n")
        log.flush()
        t = time.time()
        rc = subprocess.call(cmd, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT)
        log.write(f"[rc={rc} {time.time() - t:.1f}s]\n")
        if rc:
            return rc
    return 0


def cmd_list(_):
    for s in STAGES:
        flag = " (command reconstructed; not re-run since)" if s.get("reconstructed") else ""
        print(f"{s['tier']:9s} {s['name']:28s} {s['desc']}{flag}")
        for o in s["outputs"]:
            print(f"{'':39s}-> {o}")


MANIFEST = ROOT / "reproduce/inputs_sha256.json"


def input_files():
    for p in INPUTS:
        q = ROOT / p
        if q.is_file():
            yield q
        elif q.is_dir():
            yield from sorted(x for x in q.rglob("*") if x.is_file() and x.name != ".DS_Store" and "__pycache__" not in x.parts)


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def cmd_check_inputs(args):
    missing = [p for p in INPUTS if not (ROOT / p).exists()]
    for p in INPUTS:
        print(("ok      " if (ROOT / p).exists() else "MISSING ") + p)
    print(f"\n{len(INPUTS) - len(missing)}/{len(INPUTS)} inputs present")
    if args.write_hashes:
        man = {str(f.relative_to(ROOT)): sha256(f) for f in input_files()}
        MANIFEST.write_text(json.dumps(man, indent=0, sort_keys=True) + "\n")
        print(f"wrote {len(man)} hashes to {MANIFEST.relative_to(ROOT)}")
    elif args.hashes:
        man = json.loads(MANIFEST.read_text())
        bad = [k for k, v in man.items() if not (ROOT / k).is_file() or sha256(ROOT / k) != v]
        print(f"{len(man) - len(bad)}/{len(man)} input files match {MANIFEST.relative_to(ROOT)}")
        for k in bad[:20]:
            print("  mismatch or missing:", k)
        return 1 if bad or missing else 0
    return 1 if missing else 0


def cmd_verify(args):
    global WORK_ROOT
    stages = select(args)
    work = OUTROOT / "work"
    exclude = {o for s in stages for o in s["outputs"]}
    mirror = set(ALWAYS_MIRROR) | {w for s in stages for w in s.get("extra_writes", [])}
    print(f"building workspace {work} ...")
    build_workspace(work, exclude, mirror)
    WORK_ROOT = work
    env = dict(os.environ, CITATION_ROOT=str(work), NATURE_ARM_DIR=str(work / ND), PYTHONDONTWRITEBYTECODE="1")
    stamp = time.time()
    time.sleep(1.1)
    OUTROOT.mkdir(exist_ok=True)
    report = {"started": dt.datetime.now().isoformat(timespec="seconds"), "stages": {}}
    logdir = OUTROOT / "logs"
    logdir.mkdir(exist_ok=True)
    failed = 0
    for s in stages:
        t = time.time()
        with (logdir / f"{s['name']}.log").open("w") as log:
            rc = run_stage(s, work, env, log)
        probs, notes = compare(s, work) if rc == 0 else ([f"exit code {rc}; see reproduce_out/logs/{s['name']}.log"], [])
        status = ("IDENTICAL" if not notes else "EQUIVALENT") if not probs else ("FAILED" if rc else "DIFFERS")
        failed += bool(probs)
        report["stages"][s["name"]] = {"status": status, "seconds": round(time.time() - t, 1), "problems": probs,
                                       "notes": notes, "explanation": s.get("note")}
        print(f"{status:10s} {s['name']:28s} {time.time() - t:7.1f}s")
        for p in probs[:8]:
            print(f"           {p}")
        if notes:
            print(f"           {len(notes)} documented difference(s): {s.get('note') or notes[0]}")
    leaked = changed_since(stamp)
    report["original_repository_modified"] = leaked
    print("\noriginal repository untouched" if not leaked else f"\nWARNING: original files changed during the run: {leaked[:10]}")
    (OUTROOT / "verify_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"report: {OUTROOT / 'verify_report.json'}")
    return 1 if failed or leaked else 0


def cmd_run(args):
    if not args.inplace:
        raise SystemExit("`run` writes into this repository; pass --inplace to confirm (use `verify` to rebuild in a scratch copy)")
    env = dict(os.environ, CITATION_ROOT=str(ROOT))
    OUTROOT.mkdir(exist_ok=True)
    for s in select(args):
        with (OUTROOT / f"run_{s['name']}.log").open("w") as log:
            rc = run_stage(s, ROOT, env, log)
        print(f"{'ok' if rc == 0 else 'FAILED'} {s['name']}")
        if rc:
            return rc
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    ci = sub.add_parser("check-inputs")
    ci.add_argument("--hashes", action="store_true", help="also compare every input file with reproduce/inputs_sha256.json")
    ci.add_argument("--write-hashes", action="store_true", help="(maintainer) record the current input hashes")
    for name in ("verify", "run"):
        p = sub.add_parser(name)
        p.add_argument("--tier", default="fast", choices=["fast", "expensive", "provider"])
        p.add_argument("--stage", action="append", help="run only these stages (repeatable)")
        if name == "run":
            p.add_argument("--inplace", action="store_true")
    a = ap.parse_args()
    return {"list": cmd_list, "check-inputs": cmd_check_inputs, "verify": cmd_verify, "run": cmd_run}[a.cmd](a) or 0


if __name__ == "__main__":
    sys.exit(main())
