#!/usr/bin/env python3
"""Ordered parallel execution of an explicitly frozen Nature parser.

This runner changes execution scheduling, not extraction rules. Worker processes
read only PDF/TEI sources. Output appears in the input paper order, allowing
downstream inference to consume completed papers. Existing outputs are refused.
"""
import argparse
import csv
import hashlib
import importlib.util
import json
import multiprocessing as mp
from pathlib import Path
import time

PARSER = None


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def initialize(parser_path, frozen_hash):
    global PARSER
    if digest(parser_path) != frozen_hash:
        raise RuntimeError('Parser changed before worker initialization')
    spec = importlib.util.spec_from_file_location('frozen_nature_parser', parser_path)
    PARSER = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(PARSER)


def extract(job):
    pid, tei, pdf = job
    return PARSER.extract_paper(pid, Path(tei), Path(pdf))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--parser', type=Path, required=True)
    ap.add_argument('--frozen-hash', required=True)
    ap.add_argument('--papers', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--workers', type=int, default=4)
    args = ap.parse_args()
    parser_path = args.parser.resolve()
    if digest(parser_path) != args.frozen_hash:
        raise SystemExit('Parser hash differs from requested frozen hash')
    manifest_path = args.out.with_suffix('.manifest.json')
    if args.out.exists() or manifest_path.exists():
        raise SystemExit('Refusing to overwrite output or manifest')
    repo = Path(__file__).resolve().parents[2]
    arm = repo / 'external_data/nature_arm_2026-10-04'
    paper_ids = list(dict.fromkeys(r['pmcid'] for r in csv.DictReader(args.papers.open())))
    tei = {p.name.removesuffix('.tei.xml'): p for p in (arm/'pdfs_grobid').rglob('*.tei.xml')}
    jobs = [(pid, str(tei[pid]), str(arm/'pdfs'/f'{pid}.pdf')) for pid in paper_ids]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    stats = []
    with args.out.open('x') as output:
        with mp.get_context('spawn').Pool(args.workers, initializer=initialize,
                                          initargs=(str(parser_path), args.frozen_hash)) as pool:
            for i, (rows, stat) in enumerate(pool.imap(extract, jobs, chunksize=1), 1):
                for row in rows:
                    output.write(json.dumps(row, ensure_ascii=False)+'\n')
                output.flush()
                stats.append(stat)
                if i % 10 == 0 or i == len(jobs):
                    print(json.dumps({'papers_completed': i, 'papers_total': len(jobs),
                                      'elapsed_seconds': round(time.monotonic()-started, 2)}), flush=True)
    if digest(parser_path) != args.frozen_hash:
        raise SystemExit('Parser changed during run; output is incomplete, no completed manifest written')
    fields = ['edges', 'contexts', 'pdf_reference_labels', 'mapped_source_labels', 'pdf_contexts']
    manifest = {'status': 'complete exploratory extraction; original audit unchanged',
                'parser_sha256': args.frozen_hash,
                'runner_sha256': digest(__file__),
                'paper_selection_sha256': digest(args.papers),
                'output_sha256': digest(args.out),
                'papers': len(stats), 'workers': args.workers,
                'elapsed_seconds': time.monotonic()-started,
                'totals': {k: sum(s.get(k, 0) for s in stats) for k in fields},
                'paper_stats': stats,
                'execution': 'spawn workers; paper-order output; identical extract_paper function'}
    manifest_path.write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps({'completed_manifest': str(manifest_path), 'papers': len(stats)}), flush=True)


if __name__ == '__main__':
    main()
