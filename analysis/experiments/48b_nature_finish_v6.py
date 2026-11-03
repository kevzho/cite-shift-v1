#!/usr/bin/env python3
"""Complete one authoritative classifier run per final repaired input frame."""
import os as _os
from pathlib import Path as _Path
_CITATION_VENVS = _os.environ.get("CITATION_VENVS", _os.path.expanduser("~/.cache/citation_corpus/venvs"))
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT/'external_data/nature_repair_v2_2026-10-25'
PYTHON = _CITATION_VENVS + '/aclarc_multicite/bin/python'


def main():
    started = time.monotonic()
    while not (RUN/'classifier/full_v6_stream/stream_complete.json').exists():
        if time.monotonic()-started > 14400:
            raise SystemExit('Incomplete cache stream; no completed result claimed')
        time.sleep(10)
    assert (RUN/'fresh_round4/ai_assisted_source_review_summary.json').exists()
    full = RUN/'full/reconstructed_pdf_1200_v6.jsonl'
    assert full.with_suffix('.manifest.json').exists()
    ids = {r['pmcid'] for r in csv.DictReader((ROOT/'external_data/nature_arm_2026-10-04/parity/parity_gate_edges.csv').open())}
    subset = RUN/'pdf_development/reconstructed_v6_subset200.jsonl'
    if not subset.exists():
        with subset.open('x') as f:
            for line in full.open():
                if json.loads(line)['citing_paper_id'] in ids:
                    f.write(line)
        subset.with_suffix('.manifest.json').write_text(json.dumps({
            'status': 'complete subset of final full extraction', 'papers': len(ids),
            'source_sha256': hashlib.sha256(full.read_bytes()).hexdigest(),
            'output_sha256': hashlib.sha256(subset.read_bytes()).hexdigest()}, indent=2)+'\n')
    jobs = [
        ('full_v6', full, ['legacy'], 'grobid', True),
        ('round4_reference50', RUN/'fresh_round4/jats_reference_50.jsonl', ['legacy'], 'jats', False),
        ('round4_repaired_v6', RUN/'fresh_round4/reconstructed_pdf_50.jsonl', ['legacy', 'source_target'], 'grobid', False),
        ('development_repaired_v6', subset, ['legacy'], 'grobid', False),
    ]
    done = []
    for tag, inp, branches, reference, chi in jobs:
        out = RUN/'classifier'/tag
        out.mkdir(exist_ok=True)
        summary = out/'summary.json'
        if summary.exists():
            if json.loads(summary.read_text())['input_sha256'] != hashlib.sha256(inp.read_bytes()).hexdigest():
                raise SystemExit('Existing result/input mismatch')
        else:
            cmd = [PYTHON, str(Path(__file__).with_name('44_nature_multicite_repair.py')),
                   '--input', str(inp), '--out', str(out), '--cache', str(RUN/'classifier/cache.jsonl'),
                   '--branches', *branches, '--reference-id-space', reference, '--threads', '2', '--batch', '4']
            if chi:
                cmd.append('--chi-comparison')
            print(json.dumps({'stage': 'start', 'job': tag}), flush=True)
            with (out/'run.log').open('a') as log:
                subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True)
        done.append(tag)
        print(json.dumps({'stage': 'complete', 'job': tag}), flush=True)
    (RUN/'classifier/final_v6_runs_complete.json').write_text(json.dumps({
        'status': 'complete final revised MultiCite runs', 'jobs': done,
        'elapsed_seconds': time.monotonic()-started,
        'scope': 'Full Nature legacy mapping; fresh50 source-target sensitivity; unchanged CHI outputs reused; no revised SciCite claim'}, indent=2)+'\n')


if __name__ == '__main__':
    main()
