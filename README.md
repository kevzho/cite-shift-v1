# CITE-SHIFT v1: reproduction code

Code for the measurement study "CITE-SHIFT: Same Citations, Different Conclusions — How Citation Constructs, Human
References, and Text Extraction Shape Classifier Findings" (Kevin Zhou, SALT Lab, UIUC iSchool). It produces every
number, table and figure in the manuscript. It does not contain the manuscript, the data, or the results: the human
labels, citation text and system outputs are private or licence-restricted (see [Data you need](#data-you-need)).
The study was pre-registered on OSF (3 Oct 2026); this code corresponds to the frozen record of 3 Nov 2026.

**Codebook.** `annotations/CODEBOOK_v2_PUBLIC.md` is the v2 codebook frozen on 3 Oct 2026 with quotations of the
canonical record's rationales withheld. Scripts that build LLM prompts read the full frozen text
(`annotations/CODEBOOK_v2_DRAFT.md`, sha256 `2acafc2e…7639`), which is available on request.

## Reproduce the paper in five steps

Run everything from the repository root. Python 3.13 was used; any Python ≥ 3.11 should work for step 4.

```bash
# 1. Install the two packages the analyses need (the rest is the Python standard library)
python3 -m pip install -r reproduce/requirements-fast.txt

# 2. Put the project data in place (see "Data you need"), then check it is all there
python3 reproduce/run.py check-inputs

# 3. Optional: confirm your data copy is byte-identical to the one the paper used (3,532 files, ~3 min)
python3 reproduce/run.py check-inputs --hashes

# 4. Regenerate every result and compare it with the stored one (~4 min)
python3 reproduce/run.py verify

# 5. Read the report
cat reproduce_out/verify_report.json
```

Step 4 prints one line per stage. `IDENTICAL` means the regenerated file matches the stored one byte for byte.
`EQUIVALENT` is used for two stages whose stored files were edited by hand after they were generated; the line says
exactly what differs and why. Anything else (`DIFFERS`, `FAILED`) is a real discrepancy; the stage's log is in
`reproduce_out/logs/`.

`verify` never modifies the repository. It builds a scratch copy in `reproduce_out/work/` in which the outputs are
absent, runs every script there in dependency order exactly as on a fresh checkout, and compares. The regenerated
tables and figures end up in `reproduce_out/work/analysis/results/paper_numbers/`.

Other commands:

```bash
python3 reproduce/run.py list                                 # every stage, its tier and its outputs
python3 reproduce/run.py verify --stage b300 --stage figures  # selected stages only
python3 reproduce/run.py run --inplace --stage figures        # write outputs into the repository itself
```

**Where each number in the paper comes from** (stage, script and output file) is listed in
[`reproduce/README.md`](reproduce/README.md#where-each-result-comes-from).

## What is regenerated, and how

| Tier | Covers | Time | Command |
|---|---|---|---|
| fast | every statistic, table and figure in the paper, from the frozen model predictions | ~4 min | `python3 reproduce/run.py verify` |
| expensive | retraining the encoders, D1 and D3; MultiCite over 300k edges; the Nature PDF re-extraction | hours per stage | `python3 reproduce/run.py verify --tier expensive --stage <name>` |
| provider | re-querying GPT-6.1 Sol, Jev, Semantic Scholar and Qwen; downloading the Nature papers | hours; network and API keys | `python3 reproduce/run.py run --inplace --tier provider --stage <name>` |

The fast tier rescores the frozen outputs of the trained and hosted models. The expensive tier is deterministic on
CPU; set it up with `reproduce/requirements-ml.txt`, `reproduce/requirements-nature.txt` and
`bash reproduce/setup_envs.sh`. Hosted models are not bit-reproducible, so their frozen responses are the record.
Details: [`reproduce/README.md`](reproduce/README.md).

## Data you need

Not in this repository (private or licence-restricted). Place them at these paths, relative to the repository root:

| Path | Contents |
|---|---|
| `yiren_artifacts/`, `analysis/derived/` | the canonical human record (2,937 edges) |
| `annotations/generated/`, `annotations/coders/expansion_2026-09-26/` | the v2 coding files and keys, the v1 expansion sample, the rationale check |
| `private/history/citation-annotation-2027/templates/` | earlier annotation templates (held out of the splits) |
| `analysis/results/` | frozen model outputs (Sol, Jev, SciCite, MultiCite, S2, Qwen, ACL-ARC, encoder, D1, D3, JEPA) and the stored results that `verify` compares against |
| `analysis/ml/splits_pre_expansion_2026-09-29/` | the splits before the 29 Sep rebuild |
| `external_data/nature_arm_2026-10-04/`, `external_data/nature_repair_v2_2026-10-25/` | Nature frame, PDFs, GROBID and JATS files, repaired extraction |
| `.env` | API credentials; provider tier only; never share |

`python3 reproduce/run.py check-inputs` lists exactly which of these it finds.

## Configuration

Scripts find the repository from their own location. The shell runners also read `CITATION_ROOT` (default: the
repository root), `CITATION_VENVS` (default `~/.cache/citation_corpus/venvs`), `NATURE_ARM_DIR`, `PYTHON` and `UV`.

## Repository map

| Path | Contents |
|---|---|
| `reproduce/` | runner, requirements, environment setup, input fingerprints, detailed guide |
| `analysis/experiments/` | analyses, numbered in the order they were written; 29/38 are the registered and B300 analyses, 60/61 the derived numbers and tables |
| `analysis/figures/` | the six manuscript figures |
| `analysis/ml/` | splits, datasets, encoder, D1, D3, JEPA |
| `analysis/external_systems/` | harnesses for Sol, Jev, Qwen, SciCite, MultiCite/ACL-ARC, S2 |
| `analysis/nature_arm/`, `analysis/extraction/` | Nature frame, download, parity gate, parser check; citation extractors and the GROBID runner |
| `annotations/` | the v1 and v2 codebooks, their freeze records, the label crosswalk |

## Earlier corpus-analysis package (`run_core_analysis.sh`)

### Data

- The immutable canonical sheets for 1,040 HCI and 1,897 Nature OA citation
  edges are private inputs expected under `yiren_artifacts/`; they are not
  distributed in this repository.
- `annotations/` contains the frozen codebooks and label crosswalk.
- `analysis/derived/` is a private compatibility cache and is not versioned.
- Raw blinded samples, individual coder submissions, questionnaires, and
  adjudication workspaces remain private. Only aggregate reliability results
  and their verification code are included.

### Reproduce the deterministic corpus analysis

The active runner uses only the Python standard library but requires the two
private canonical input sheets in `yiren_artifacts/`:

```bash
bash run_core_analysis.sh
```

It regenerates the cross-corpus distributions and compression measurements,
then checks input hashes, denominators, output structure, and scope exclusions.

### Verify the admitted manuscript quantities

The independent specifications, implementations, and machine-readable results
for the recalibration, validation, compression, and bootstrap derivations are
under `analysis/verification/`. Private governance ledgers and coder workspaces
are not part of the public artifact.

Rows 33 and 35 use private compatibility CSVs derived from the canonical
sheets. With those inputs installed, rerun them with:

```bash
python3 analysis/verification/row33_kz.py
python3.13 analysis/verification/row35_kz.py
```

Row 35 requires Python 3.13 because its implementation uses
`zip(..., strict=True)`. Rows 32 and 34 also require private coder files; their
frozen specifications, code, input hashes, and aggregate outputs are retained
for audit.

### Artifact map

| Path | Contents |
|---|---|
| `annotations/` | Frozen instruments and crosswalk |
| `analysis/data_processing/` | Canonical private-input loader |
| `analysis/measurement/` | Paper-facing measurements and annotation workflow |
| `analysis/quality/` | Integrity and scope checks |
| `analysis/verification/` | Admitted derivations for rows 32–35 |
| `analysis/results/` | Retained aggregate outputs |

### Interpretation boundaries

- The corpora are observational samples, not matched field populations.
- Nature OA is not representative of all science.
- The contribution-sensitive labels are not a fully validated gold standard.
- The compact scheme was not independently applied to the full corpus; the
  compression analysis measures distinctions a compact label cannot express by
  construction.
- No causal, field-invariance, complete-citation-graph, or public-full-text
  release claim is made.

## License

See `LICENSE` for repository licensing.
