#!/usr/bin/env python3
"""Exploratory revised-Nature MultiCite inference, exact-input cache and coverage.

Uses the locally cached pinned checkpoint. Never changes original model inputs,
outputs or the registered RQ4(b) verdict. The legacy branch isolates changed
context text; source_target is a separate sensitivity analysis requiring
explicit parser-supplied target markup. Cached baseline predictions retain their
original six-decimal probabilities and are reused only for identical model text.
"""
from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
HARNESS = ROOT / 'analysis/external_systems/aclarc_multicite'
BASE = ROOT / 'analysis/results/external_systems/aclarc_multicite'
REV = '7696929cf8ba26c21cc1da0daeb76010575dff80'
MODEL = 'allenai/multicite-multilabel-scibert'
ELIGIBLE = {'uses', 'extends', 'motivation', 'differences', 'background'}


def lines(path):
    with Path(path).open() as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def digest(s):
    return hashlib.sha256((MODEL + '\x1f' + REV + '\x1f' + s).encode()).hexdigest()


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def cache_baseline(cache, tag):
    marked_path = BASE / 'inputs' / f'marked_inputs_{tag}.jsonl'
    preds_path = BASE / 'outputs' / f'multicite_scibert_{tag}.jsonl'
    if not marked_path.exists() or not preds_path.exists():
        return 0
    marked = {r['edge_id']: r for r in lines(marked_path)}
    count = 0
    for row in lines(preds_path):
        if row.get('revision') != REV or row.get('model') != MODEL:
            raise ValueError('Baseline checkpoint mismatch')
        inp = marked.get(row['edge_id'])
        if inp is None or len(inp['marked']) != len(row['per_citance']):
            raise ValueError('Baseline input/prediction correspondence mismatch')
        for text, pred in zip(inp['marked'], row['per_citance']):
            key = digest(text)
            if key not in cache:
                cache[key] = {'key': key, 'text': text, 'probs': pred['probs'],
                              'source': 'frozen_baseline:' + tag}
                count += 1
    return count


def classify(probs, resolutions):
    keep = [p for p, r in zip(probs, resolutions)
            if r in {'single_group', 'intersection', 'source_target'}]
    if not keep:
        return 'excluded_marker'
    maxima = {label: max(p[label] for p in keep) for label in keep[0]}
    eligible = [(v, label) for label, v in maxima.items() if label in ELIGIBLE and v > .5]
    return ('none' if max(eligible)[1] == 'background' else 'uptake') if eligible else 'no_claim'


def statistics(records):
    counts = Counter(r['classification'] for r in records)
    claims = counts['none'] + counts['uptake']
    return {'edges': len(records), 'papers': len({r['citing_paper_id'] for r in records}),
            'counts': dict(counts), 'claims': claims,
            'uptake_share_among_claims': counts['uptake'] / claims if claims else None}


def baseline_records(papers):
    mapping = {}
    inp = BASE / 'at_scale/input_nature_arm.csv'
    for r in csv.DictReader(inp.open()):
        if r['citing_paper_id'] in papers:
            mapping[r['edge_id']] = r
    records = []
    for r in lines(BASE / 'outputs/multicite_scibert_at_scale_nature_arm.jsonl'):
        source = mapping.get(r['edge_id'])
        if source is None:
            continue
        records.append({'edge_id': r['edge_id'], 'citing_paper_id': source['citing_paper_id'],
                        'cited_ref_id': source['cited_ref_id'], 'branch': 'original_grobid',
                        'classification': classify([x['probs'] for x in r['per_citance']], r['marker_resolution'])})
    return records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--cache', required=True)
    ap.add_argument('--threads', type=int, default=2)
    ap.add_argument('--batch', type=int, default=4)
    ap.add_argument('--branches', nargs='+', choices=['legacy', 'source_target'], default=['legacy', 'source_target'])
    ap.add_argument('--bootstrap', type=int, default=2000)
    ap.add_argument('--limit', type=int)
    ap.add_argument('--reference-id-space', choices=['grobid', 'jats', 'none'], default='none')
    ap.add_argument('--chi-comparison', action='store_true', help='Descriptive contrast to unchanged frozen CHI predictions')
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    result_path = out / 'summary.json'
    if result_path.exists():
        raise SystemExit('Completed output exists; choose a new output directory')
    script_bytes = Path(__file__).read_bytes()
    (out / 'runner_source.py').write_bytes(script_bytes)
    cache_path = Path(args.cache)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache = {r['key']: r for r in lines(cache_path)} if cache_path.exists() else {}
    for tag in ['nature_parity_grobid', 'nature_parity_jats', 'at_scale_nature_arm']:
        cache_baseline(cache, tag)
    markers = load_module('repair_markers', HARNESS / 'markers.py')
    rows = list(lines(args.input))
    if args.limit:
        rows = rows[:args.limit]
    prepared = []
    tasks = {}
    for row in rows:
        contexts = row.get('citances', [])
        if isinstance(contexts, str):
            contexts = [c.strip() for c in contexts.split('|||') if c.strip()]
        if not contexts:
            continue
        eid = row.get('edge_id', f"repair:{row['citing_paper_id']}:{row['cited_ref_id']}")
        legacy_texts, legacy_resolutions, legacy_numbers = markers.mark_edge(contexts, 'nature_oa')
        for branch in args.branches:
            if branch == 'legacy':
                texts, resolutions, numbers = legacy_texts, legacy_resolutions, legacy_numbers
            else:
                supplied = row.get('marked_citances')
                if supplied is None:
                    continue
                if len(supplied) != len(contexts):
                    raise ValueError('Source-marked/context length mismatch')
                provenance = row.get('context_provenance', [{} for _ in contexts])
                if len(provenance) != len(contexts):
                    raise ValueError('Provenance/context length mismatch')
                texts, resolutions = [], []
                for fallback, fallback_resolution, marked, source in zip(legacy_texts, legacy_resolutions, supplied, provenance):
                    tagged = bool(marked) and '<cite>' in marked and '</cite>' in marked
                    certified = source.get('target_span_unambiguous') is True and source.get('target_group_size') == 1
                    texts.append(marked if tagged and certified else fallback)
                    resolutions.append('source_target' if tagged and certified else fallback_resolution)
                numbers = []
            for t in texts:
                if digest(t) not in cache:
                    tasks[digest(t)] = t
            prepared.append({'edge_id': eid, 'citing_paper_id': row['citing_paper_id'],
                             'cited_ref_id': row['cited_ref_id'], 'branch': branch,
                             'texts': texts, 'marker_resolution': resolutions, 'target_numbers': numbers})
    t0 = time.monotonic()
    print(json.dumps({'stage': 'prepared', 'input_edges': len(rows), 'branch_edges': len(prepared),
                      'cached_texts': len(cache), 'new_unique_texts': len(tasks)}), flush=True)
    # Write imported frozen cache once, then append batches for safe resume.
    existing_keys = {r['key'] for r in lines(cache_path)} if cache_path.exists() else set()
    with cache_path.open('a') as f:
        for key, value in cache.items():
            if key not in existing_keys:
                f.write(json.dumps(value) + '\n')
        f.flush()
    if tasks:
        sys.path.insert(0, str(HARNESS))
        harness = load_module('repair_multicite', HARNESS / 'run_multicite.py')
        harness.set_determinism(args.threads)
        tok, model, labels = harness.load('scibert')
        items = list(tasks.items())
        with cache_path.open('a') as f:
            for offset in range(0, len(items), args.batch):
                batch = items[offset:offset + args.batch]
                pred = harness.predict(tok, model, [t for _, t in batch], args.batch)
                for (key, text), probability in zip(batch, pred):
                    rec = {'key': key, 'text': text,
                           'probs': {label: round(float(probability[i]), 6) for i, label in enumerate(labels)},
                           'source': 'new_inference_pinned_checkpoint',
                           'inference_threads': args.threads, 'inference_batch': args.batch,
                           'max_length': 512}
                    cache[key] = rec
                    f.write(json.dumps(rec) + '\n')
                f.flush()
                if offset % (args.batch * 100) == 0:
                    print(json.dumps({'stage': 'inference', 'done': min(offset + args.batch, len(items)),
                                      'total': len(items), 'seconds': round(time.monotonic() - t0, 1)}), flush=True)
    records = []
    with (out / 'predictions.jsonl').open('w') as f:
        for row in prepared:
            probabilities = [cache[digest(t)]['probs'] for t in row['texts']]
            record = {**row, 'model': MODEL, 'revision': REV,
                      'per_citance': [{'probs': p} for p in probabilities],
                      'inference_sources': [cache[digest(t)]['source'] for t in row['texts']],
                      'classification': classify(probabilities, row['marker_resolution'])}
            records.append(record)
            f.write(json.dumps(record) + '\n')
    by_branch = {b: [r for r in records if r['branch'] == b] for b in args.branches}
    papers = sorted({r['citing_paper_id'] for r in records})
    baseline = baseline_records(set(papers))
    by_branch['original_grobid'] = baseline
    summary = {'status': 'EXPLORATORY; original registered RQ4(b) remains failed/not run',
               'model': MODEL, 'revision': REV,
               'input_sha256': hashlib.sha256(Path(args.input).read_bytes()).hexdigest(),
               'script_sha256': hashlib.sha256(script_bytes).hexdigest(),
               'reference_id_space': args.reference_id_space,
               'inference_configuration': {'threads': args.threads, 'batch': args.batch,
                                           'max_length': 512, 'device': 'CPU', 'deterministic_algorithms': True},
               'new_inference_unique_texts': len(tasks), 'elapsed_seconds': time.monotonic() - t0,
               'cache_reuse_rule': 'identical UTF-8 marked text, model identifier and revision; original six-decimal probabilities; batch/thread arithmetic history may differ, so bitwise equivalence is not asserted',
               'branches': {b: statistics(rs) for b, rs in by_branch.items() if rs},
               'limitations': ['Automatic classifier output rates, not annotation accuracy.',
                               'Legacy and source-target branches are distinct measurement procedures.',
                               'Original-versus-revised corpus rates use the same citing-paper frame but may have different edge/claim coverage.',
                               'No unchanged CHI prediction was rerun.']}
    summary['source_target_override_contexts'] = sum(r == 'source_target' for x in records for r in x['marker_resolution'])
    summary['source_target_fallback_contexts'] = sum(r != 'source_target' for x in records if x['branch'] == 'source_target' for r in x['marker_resolution'])
    baseline_keys = {(r['citing_paper_id'], r['cited_ref_id']): r for r in baseline} if args.reference_id_space == 'grobid' else {}
    summary['common_reference_classification_changes'] = {}
    for b, rs in by_branch.items():
        if b == 'original_grobid':
            continue
        pairs = [(baseline_keys[(r['citing_paper_id'], r['cited_ref_id'])]['classification'], r['classification'])
                 for r in rs if (r['citing_paper_id'], r['cited_ref_id']) in baseline_keys]
        binary = [(x, y) for x, y in pairs if x in {'none', 'uptake'} and y in {'none', 'uptake'}]
        summary['common_reference_classification_changes'][b] = {
            'paired_edges': len(pairs), 'all_category_changes': sum(x != y for x, y in pairs),
            'claim_eligible_in_both': len(binary), 'uptake_binary_changes': sum(x != y for x, y in binary),
            'transition_counts': dict(Counter(x + ' -> ' + y for x, y in pairs)),
            'caveat': 'Pairs use identical source reference IDs; no cross-parser reference IDs are inferred here.'}
    rng = random.Random(20261025)
    grouped = {b: defaultdict(list) for b in by_branch}
    for b, rs in by_branch.items():
        for r in rs:
            grouped[b][r['citing_paper_id']].append(r)
    # Identical resampling, computed from per-paper sufficient statistics.
    paper_counts = {b: {p: (sum(r['classification'] == 'uptake' for r in rs),
                             sum(r['classification'] in {'none', 'uptake'} for r in rs))
                        for p, rs in gp.items()} for b, gp in grouped.items()}
    bootstrap = defaultdict(list)
    contrast_bootstrap = defaultdict(list)
    for _ in range(args.bootstrap):
        sample = rng.choices(papers, k=len(papers)) if papers else []
        replicate = {}
        for b in by_branch:
            numerator = sum(paper_counts[b].get(p, (0, 0))[0] for p in sample)
            denominator = sum(paper_counts[b].get(p, (0, 0))[1] for p in sample)
            rate = numerator / denominator if denominator else None
            if rate is not None:
                bootstrap[b].append(rate)
                replicate[b] = rate
        if 'original_grobid' in replicate:
            for b, rate in replicate.items():
                if b != 'original_grobid':
                    contrast_bootstrap[b].append(rate - replicate['original_grobid'])
    for b, values in bootstrap.items():
        values.sort()
        summary['branches'][b]['uptake_share_ci95_paper_bootstrap'] = [values[int(.025 * len(values))], values[int(.975 * len(values)) - 1]]
    summary['paired_paper_contrasts_revised_minus_original'] = {}
    for b, values in contrast_bootstrap.items():
        values.sort()
        summary['paired_paper_contrasts_revised_minus_original'][b] = {
            'difference': summary['branches'][b]['uptake_share_among_claims'] - summary['branches']['original_grobid']['uptake_share_among_claims'],
            'ci95': [values[int(.025 * len(values))], values[int(.975 * len(values)) - 1]]}
    if args.chi_comparison:
        scale = load_module('repair_scale_summary', Path(__file__).with_name('28_multicite_at_scale_summary.py'))
        chi, chi_counts = scale.load('chi')
        chi_by_paper = [(sum(xs), len(xs)) for xs in chi.values()]
        chi_rate = sum(x for x, n in chi_by_paper) / sum(n for x, n in chi_by_paper)
        summary['chi_frozen_baseline'] = {'counts': dict(chi_counts), 'claim_bearing_citing_papers': len(chi_by_paper), 'uptake_share': chi_rate}
        summary['chi_minus_nature_descriptive'] = {}
        nature_counts = {b: [(sum(r['classification'] == 'uptake' for r in grouped[b][p]),
                              sum(r['classification'] in {'none', 'uptake'} for r in grouped[b][p]))
                             for p in papers] for b in by_branch if grouped[b]}
        rng = random.Random(20261025)
        boot = defaultdict(list)
        for _ in range(args.bootstrap):
            selected = rng.choices(chi_by_paper, k=len(chi_by_paper))
            cr = sum(x for x, n in selected) / sum(n for x, n in selected)
            for b, counts in nature_counts.items():
                selected = rng.choices(counts, k=len(counts))
                denominator = sum(n for x, n in selected)
                if denominator:
                    boot[b].append(cr - sum(x for x, n in selected) / denominator)
        for b, values in boot.items():
            values.sort()
            summary['chi_minus_nature_descriptive'][b] = {
                'difference': chi_rate - summary['branches'][b]['uptake_share_among_claims'],
                'ci95': [values[int(.025 * len(values))], values[int(.975 * len(values)) - 1]],
                'status': 'Exploratory unmatched frames; source_target additionally changes Nature marker handling; no registered verdict'}
    result_path.write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
