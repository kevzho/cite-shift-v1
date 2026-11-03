#!/usr/bin/env python3
"""Verify final extraction provenance and arrays without changing research data."""
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT/'external_data/nature_repair_v2_2026-10-25'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    original = json.loads((RUN/'protocol.json').read_text())
    for name, expected in original['preserve_inputs'].items():
        assert sha(ROOT/name) == expected, name
    code = ROOT/'analysis/experiments/42d_nature_pdf_reconstruction_v6.py'
    frozen = json.loads((RUN/'fresh_round4/protocol_round4.json').read_text())
    assert sha(code) == frozen['parser_sha256']
    dev = {r['pmcid'] for r in csv.DictReader((ROOT/'external_data/nature_arm_2026-10-04/parity/parity_gate_edges.csv').open())}
    splits = [dev]
    for plan in [ROOT/'external_data/nature_repair_2026-10-25/fresh_50_paper_validation_plan.csv',
                 RUN/'fresh_round2/fresh_100_paper_validation_plan.csv',
                 RUN/'fresh_round3/fresh_50_paper_validation_plan.csv',
                 RUN/'fresh_round4/fresh_50_paper_validation_plan.csv']:
        ids = {r['pmcid'] for r in csv.DictReader(plan.open())}
        assert all(not (ids & previous) for previous in splits), plan
        splits.append(ids)
    selected = {r['pmcid'] for r in csv.DictReader((RUN/'full/papers_1200.csv').open())}
    assert len(selected) == 1200
    totals = {}
    for name in ['reconstructed_pdf_1200_v6.jsonl', 'reconstructed_pdf_1200_v6_dedup.jsonl']:
        path = RUN/'full'/name
        manifest = json.loads(path.with_suffix('.manifest.json').read_text())
        assert sha(path) == manifest['output_sha256']
        seen = set()
        paper_ids = set()
        contexts = 0
        for line in path.open():
            row = json.loads(line)
            identity = row['citing_paper_id'], row['cited_ref_id']
            assert identity not in seen
            seen.add(identity)
            paper_ids.add(identity[0])
            n = row['n_citances']
            assert n > 0
            for field in ['citances', 'marked_citances', 'context_provenance', 'sections']:
                assert len(row[field]) == n, (identity, field)
            for provenance in row['context_provenance']:
                if provenance['source'].startswith('pdf'):
                    assert provenance['source_reference_number'] in row['pdf_reference_numbers']
                    assert provenance['page'] >= 1 and len(provenance['bbox']) == 4
            contexts += n
        assert paper_ids <= selected
        totals[name] = {'edges': len(seen), 'contexts': contexts,
                        'papers_with_edges': len(paper_ids), 'papers_without_edges': sorted(selected-paper_ids)}
    raw = json.loads((RUN/'fresh_round4/evaluation_revised_source_label.json').read_text())
    dedup = json.loads((RUN/'fresh_round4/evaluation_revised_dedup_source_label.json').read_text())
    for metric in ['citance_recovery', 'complete_edge_match', 'strict_unique_citance_recovery']:
        assert raw[metric] == dedup[metric], metric
    regression = json.loads((RUN/'pdf_development/v6_regression_checks.json').read_text())
    assert regression['five_known_false_numeric_contexts_removed']
    assert set(range(26, 40)) <= set(regression['citation_range_26_through_39_recovered'])
    result = {'status': 'passed structural/provenance checks; not a claim of semantic precision',
              'original_inputs_unchanged': True, 'parser_sha256': sha(code),
              'disjoint_split_sizes': list(map(len, splits)), 'full_corpus': totals,
              'deduplication_preserves_fresh_recovery': True,
              'real_source_numeric_marker_regressions': 'passed'}
    target = RUN/'artifact_checks.json'
    if target.exists():
        raise SystemExit('Check record exists; refusing overwrite')
    target.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
