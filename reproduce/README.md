# Reproducing the iConference 2027 results

Every number, table and figure in the manuscript comes from a script in this repository, run from the command line.
`reproduce/run.py` runs those scripts in dependency order and checks the regenerated files against the stored ones.

```bash
pip install -r reproduce/requirements-fast.txt      # numpy, matplotlib; the rest is the standard library
python3 reproduce/run.py check-inputs                # private inputs and frozen model outputs present?
python3 reproduce/run.py verify                      # fast tier, ~4 min: rebuild everything and compare
python3 reproduce/run.py list                        # all stages, tiers and output paths
```

`verify` never writes into the repository. It builds `reproduce_out/work/`, a copy in which the stage outputs are
absent, code is copied and inputs are symlinked, runs each script exactly as on a fresh checkout, and compares the
regenerated files byte for byte with the stored ones (figures: PNG renders; PDFs carry a timestamp). It then checks
that nothing in the original repository changed. The report is `reproduce_out/verify_report.json`, with one log per
stage in `reproduce_out/logs/`.

To regenerate into a copy of the repository that does not contain the outputs yet, use
`python3 reproduce/run.py run --inplace [--stage NAME ...]`.

## Tiers

| Tier | What | Time | Needs |
|---|---|---|---|
| `fast` | all statistics, tables and figures, from the frozen model outputs and private inputs | about 3 min | `requirements-fast.txt` |
| `expensive` | model training and long CPU inference: encoder, D1, D3, MultiCite at scale, parser-bias system runs, repaired Nature extraction | hours each | `requirements-ml.txt`, `requirements-nature.txt`, `setup_envs.sh` |
| `provider` | hosted and remote systems: GPT-6.1 Sol (Azure), Jev (TypeSafe), Semantic Scholar, Qwen (Ollama), Nature downloads | hours; network | API keys in `.env` |

Run a tier with `python3 reproduce/run.py verify --tier expensive --stage d1` (always name the stages; several take
hours). CPU training is bit-for-bit reproducible (`--device cpu`, the default). Provider outputs are not: hosted
models change and sample, so their frozen responses in `analysis/results/external_systems/` are the record that the
fast tier re-scores. The provider stages re-query; `run_hosted.py` and `run_jev.py` replay from their response caches
when the same inputs are sent, but still require the credentials to be set.

The JEPA pretraining (`analysis/ml/d2_jepa/`, several days on an Apple-silicon GPU) and GROBID
(`analysis/nature_arm/grobid_supervised.sh`, Docker image `grobid/grobid:0.9.0-crf`) are documented in their own
folders and are not wrapped as stages.

## Configuration

Scripts locate the repository from their own path. Runners that call other environments read:

| Variable | Default | Used by |
|---|---|---|
| `CITATION_ROOT` | the repository root | all `.sh` runners |
| `CITATION_VENVS` | `~/.cache/citation_corpus/venvs` | SciCite, MultiCite/ACL-ARC, S2, Qwen runners |
| `NATURE_ARM_DIR` | `external_data/nature_arm_2026-10-04` | `analysis/nature_arm/` |
| `PYTHON`, `UV` | `python3`, `uv` on `PATH` | runners |

`bash reproduce/setup_envs.sh` creates the four system environments from the frozen package lists in
`reproduce/envs/` (Python 3.11.15 for SciCite, MultiCite and S2; 3.12.13 for the Qwen client). The main analyses used
Python 3.13.9.

## Inputs

These are not distributed with the code; `check-inputs` lists them.

- **Canonical record** (Yiren Liu): `yiren_artifacts/` and its compatibility copies in `analysis/derived/`.
- **Human coding**: `annotations/generated/` (v2 coding files and keys, including the excluded submissions kept as a
  record in `v2_coding_2026-10-02/received/`, the rationale check, the expansion sample) and
  `annotations/coders/expansion_2026-09-26/`. These files identify the coder; handle them as private.
- **Frozen model outputs**: `analysis/results/external_systems/` (Sol, Jev, SciCite, MultiCite, S2, Qwen, ACL-ARC) and
  `analysis/results/ml/` (encoder, D1, D3 and JEPA predictions and probabilities).
- **Full text and Nature arm**: `external_data/nature_arm_2026-10-04/` (frame, PDFs, GROBID TEI, JATS) and
  `external_data/nature_repair_v2_2026-10-25/`.
- **Credentials**: `.env` (provider tier only). Never share it.

## Where each result comes from

Stage names refer to `reproduce/run.py list`. All paths are under `analysis/results/` unless they start with
`external_data/`.

| Paper result | Stage | File |
|---|---|---|
| Table 1 samples; 297 eligible = 147 HCI + 150 Nature OA | `derived_numbers`, `tables` | `paper_numbers/derived_numbers.json` (`b300_eligible_by_corpus`), `paper_numbers/tables/samples.tex` |
| 2,937 edges; component-aware splits; 914/197/195 | `splits` | `analysis/ml/splits/split_manifest.json` |
| Split overlap found 29 Sep: 281/61/61 (97 frozen) | `derived_numbers` | `derived_numbers.json` (`split_overlap_2026-09-29`) |
| A250 = seeded 250 (seed 20261005) | `derived_numbers` | `derived_numbers.json` (`a250_draw`: the seed reproduces the selected set) |
| Calibration critical cells 5/8/11 | `derived_numbers` | `derived_numbers.json` (`calibration_cells`) |
| Stratum provenance overlap 52/60 vs 16–18/60 | `derived_numbers` | `derived_numbers.json` (`stratum_pass2_overlap`) |
| System counts: 2,929 edges; 18 / 1,316 / 2,187 | `hosted_input`, `convert_systems` | `external_systems/hosted_llm_codebook_v2/input_canonical.csv`, `external_systems/scoring_inputs_canonical_20261004/conversion_notes.json` |
| RQ1 (κ 0.359 / 0.417, −0.058; yes-only +0.099; A250 −0.039, +0.143) | `b300` | `sensitivity_2026-10-23_B300/rq1b.json` |
| RQ1 registered 240-edge analysis (−0.079, n = 237) | `confirmatory_240` | `confirmatory_2026-10-08/rq1b.json` |
| Rationale audit 3/50, Wilson 2.1–16.2% | `derived_numbers` | `derived_numbers.json` (`rationale_audit`) |
| RQ2 (58/46/89; NMI 0.023, −0.677; 132/297, 117/297; 0.208) | `b300` | `sensitivity_2026-10-23_B300/rq2.json` |
| RQ3 scores, Table 4; κ uptake vs crediting | `b300`, `tables` | `sensitivity_2026-10-23_B300/rq3.json`, `paper_numbers/tables/rq3_scores.tex` |
| RQ3 flips 5/27, 14/48; without D1 2/18, 9/17 | `derived_numbers` | `derived_numbers.json` (`rq3_flips`) |
| RQ3 flips without one-class cells 4/21, 11/38 | `b300_companions` | `reanalysis_2026-10-23_B300_companions/robustness.json` |
| Table 5 coverage | `tables` | `paper_numbers/tables/rq3_support.tex` |
| Qwen addendum (5/38, 27/94; τ; F1; κ) | `b300_companions` | `reanalysis_2026-10-23_B300_companions/with_qwen/rq3.json` |
| Qwen pairing-rule failures 197/550 | `qwen_addendum`, `derived_numbers` | `confirmatory_2026-10-12_with_qwen/qwen_addendum.json`; `derived_numbers.json` |
| RQ4(a) table and figure; encoder and D1 seed ranges | `rq4a_inputs`, `rq4a`, `tables` | `exploratory/2026-10-11_rq4a_final.json`, `paper_numbers/tables/rq4a.tex` |
| RQ4(b) parity gate 86.1% / 82.5% | `extract_nature`, `extract_jats_parity`, `parity_gate` | `external_data/nature_arm_2026-10-04/parity/parity_gate_result.json` |
| D1 scores on B300 | `b300` | `sensitivity_2026-10-23_B300/d1.json` |
| D1 4.5 CPU-hours; hosted estimate US$1,394 | `derived_numbers` | `derived_numbers.json` (`costs`) |
| System consistency (Sol, Jev, SciCite, MultiCite, S2) | `derived_numbers`; frozen | `external_systems/hosted_llm_codebook_v2/stability_s1_vs_s2.json`, `external_systems/jev_codebook_v2/compare_s1_s2.json`, `derived_numbers.json` (`rerun_consistency`) |
| D3 (ECE 0.082/0.111, Brier 0.350/0.384, paired CIs) | `d3_audit` (from saved probabilities); `d3` retrains | `audits/2026-10-26_d3_aggregation.json`; `ml/d3_2026-10/main/d3_results.json` |
| Descriptive calibration (Jev, D1 ECE) | `b300_companions` | `reanalysis_2026-10-23_B300_companions/calibration.json` |
| Disagreement among the LLM judges (dependence 83.5%, crediting 64.7%, uptake 37.7%; human vs majority 36.4%, 26.3%) | `model_disagreement` | `exploratory/2026-10-25_model_disagreement_B297_primary.json` |
| LLM judges vs canonical (+0.163, −0.130) | `llm_judges_vs_canonical` | `exploratory/2026-10-04_construct_tracking.json` |
| JATS vs GROBID diagnostic | `parser_bias` (needs `parser_bias_systems`) | `external_data/nature_arm_2026-10-04/parser_bias/parser_bias_result.json` |
| MultiCite at scale, CHI 11.75% | `multicite_at_scale_summary` (needs `multicite_at_scale`) | `exploratory/2026-10-07_multicite_at_scale.json` |
| Recovered subset 10,138/12,298; 10 ambiguous / 5 targets; 17.41% | `recovered_nature`, `derived_numbers` | `exploratory/2026-10-12_recovered_nature_edges.json`; `derived_numbers.json` |
| Nature repair 95.58%; 74,089; 41,559; 19.01%; −7.26 | `nature_repair_extract` and the chain below | `external_data/nature_repair_v2_2026-10-25/` (see its `REPAIR_REPORT.md`) |
| JEPA vs MLM on B300 | `b300_companions` | `reanalysis_2026-10-23_B300_companions/d2_scores.json` |
| S2 influential on 2,917 edges | `s2_consequence` | `exploratory/2026-10-12_s2_consequence.json` |
| ACL-ARC (complete 2,929-edge run, exploratory) | `b300_companions`, `derived_numbers` | `reanalysis_2026-10-23_B300_companions/aclarc.json`; `derived_numbers.json` (`aclarc_runs`) |
| Figures 1–6 | `figures` | `paper_numbers/figures/` |

## Nature repair chain

The repaired extraction (`nature_repair_extract`) feeds this sequence, documented in
`external_data/nature_repair_v2_2026-10-25/REPAIR_REPORT.md`: `54` deduplicate → `49` fetch the fresh-50 publisher
references (network) → `47` evaluate → `52` paired validation → `51` source review → `44c` MultiCite stream (fills an
append-only cache) → `48b`, `55` classifier summaries → `56` artifact checks → `57` report. Scripts 41–43j, 44b, 48
and 50 are earlier development rounds kept as a record.

## Known limits

- The deviations table in the manuscript is a dated record of protocol changes, not a computed result.
- The hosted-model cost estimate (US$1,394) used per-token prices passed on the command line that were not recorded;
  the token counts are in `estimate_canonical.txt`.
- The encoder stage's command line was reconstructed from its output names and logs (tag `sweep1004`); it has not been
  re-run since.
- The fresh-50 Nature validation papers and the 30-context source review were selected and judged with AI assistance;
  the independent human review sheet (`fresh_round4/independent_human_source_review.csv`) is not yet filled in.
