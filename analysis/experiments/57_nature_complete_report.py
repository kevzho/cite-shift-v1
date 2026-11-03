#!/usr/bin/env python3
"""Consolidate only completed final repair/validation/classifier artifacts."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT/'external_data/nature_repair_v2_2026-10-25'


def read(path):
    return json.loads((RUN/path).read_text())


def pct(value):
    return f'{100*value:.2f}%'


def main():
    assert (RUN/'classifier/all_final_runs_complete.json').exists(), 'Classifier runs still incomplete'
    target = RUN/'COMPLETION.json'
    if target.exists():
        raise SystemExit('Completion record exists; refusing overwrite')
    validation = read('fresh_round4/evaluation_revised_source_label.json')
    dedup_validation = read('fresh_round4/evaluation_revised_dedup_source_label.json')
    assert validation['citance_recovery'] == dedup_validation['citance_recovery']
    raw = read('classifier/full_v6/summary.json')
    full = read('classifier/full_v6_dedup/summary.json')
    fresh = read('classifier/round4_repaired_v6_dedup/summary.json')
    reference = read('classifier/round4_reference50/summary.json')
    for s in [raw, full, fresh, reference]:
        assert s['revision'] == '7696929cf8ba26c21cc1da0daeb76010575dff80'
    assert full['input_sha256'] == hashlib.sha256((RUN/'full/reconstructed_pdf_1200_v6_dedup.jsonl').read_bytes()).hexdigest()
    assert full['branches']['legacy']['papers'] == 1200
    checks = read('artifact_checks.json')
    assert checks['original_inputs_unchanged']
    result = {
        'status': 'complete revised extraction, fresh validation, source audit, and affected MultiCite reruns',
        'parser': 'analysis/experiments/42d_nature_pdf_reconstruction_v6.py',
        'validation': {k: validation[k] for k in [
            'papers', 'reference_edges', 'reference_contexts', 'citance_recovery',
            'citance_recovery_ci95', 'complete_edge_match', 'strict_unique_citance_recovery']},
        'source_audit': read('fresh_round4/ai_assisted_source_review_summary.json'),
        'full_extraction': checks['full_corpus'],
        'classifier_full_dedup': full,
        'classifier_raw_format_sensitivity': raw,
        'classifier_fresh_dedup': fresh,
        'classifier_fresh_reference': reference,
        'historical_status': 'Original failed RQ4(b) remains not run; new analyses are exploratory',
        'limitations': ['95.6% is measured recovery on 50 fresh papers, not a guarantee of 95% overall accuracy.',
                        'Independent human precision remains unestablished.',
                        'Earlier failed parser attempts and source-audit failures are retained.',
                        'Nature/CHI frames are unmatched and rates are automated-label estimates.',
                        'Only affected MultiCite at-scale results are revised; original SciCite diagnostic is unchanged.'],
    }
    target.write_text(json.dumps(result, indent=2)+'\n')
    rows = [
        ('Original Nature extraction, same 1,200 papers', full['branches']['original_grobid']),
        ('Repaired Nature, raw-format sensitivity', raw['branches']['legacy']),
        ('Repaired Nature, conservative deduplication', full['branches']['legacy']),
    ]
    block = '\n\n## Completed classifier results\n\n'
    block += '| Frame / preprocessing | Edges | Claim-containing edges | Uptake rate among claims |\n|---|---:|---:|---:|\n'
    for label, s in rows:
        block += f"| {label} | {s['edges']:,} | {s['claims']:,} | {pct(s['uptake_share_among_claims'])} |\n"
    contrast = full['chi_minus_nature_descriptive']['legacy']
    block += f"\nFrozen CHI uptake rate: **{pct(full['chi_frozen_baseline']['uptake_share'])}**. CHI-minus-repaired-Nature difference: **{100*contrast['difference']:.2f} percentage points**, 95% citing-paper bootstrap interval **[{100*contrast['ci95'][0]:.2f}, {100*contrast['ci95'][1]:.2f}]**. This is an exploratory unmatched-frame comparison.\n"
    paired = full['paired_paper_contrasts_revised_minus_original']['legacy']
    block += f"\nOn the same Nature paper frame, revised-minus-original uptake-rate change: **{100*paired['difference']:.2f} points**, paired 95% interval **[{100*paired['ci95'][0]:.2f}, {100*paired['ci95'][1]:.2f}]**. Eligibility changes are reported in the machine-readable summaries.\n"
    block += '\nThe final fresh-paper publisher-reference and repaired legacy/source-target classifier checks are complete. Model inputs, inference/configuration hashes, exact-text cache provenance, branch fallbacks, and prediction payloads are saved under `classifier/`. No model accuracy against new human construct labels is inferred from these rates.\n'
    report = RUN/'REPAIR_REPORT.md'
    text = report.read_text().replace(
        'Status: **RUNNING — do not describe the full revised classifier experiment as completed until `classifier/all_final_runs_complete.json` exists.**',
        'Status: **COMPLETE.** All final classifier runs and the conservative deduplication comparison have finished; see `classifier/all_final_runs_complete.json` and `COMPLETION.json`.')
    report.write_text(text+block)
    print(json.dumps({'completion': str(target), 'recovery': validation['citance_recovery'],
                      'nature_uptake': full['branches']['legacy']['uptake_share_among_claims'],
                      'chi_minus_nature': contrast}, indent=2))


if __name__ == '__main__':
    main()
