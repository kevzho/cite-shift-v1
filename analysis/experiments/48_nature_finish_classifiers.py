#!/usr/bin/env python3
"""Finish authoritative revised classifier runs after streamed cache warming."""
import os as _os
from pathlib import Path as _Path
_CITATION_VENVS = _os.environ.get("CITATION_VENVS", _os.path.expanduser("~/.cache/citation_corpus/venvs"))
from pathlib import Path
import hashlib
import json
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT/'external_data/nature_repair_v2_2026-10-25'
PYTHON = _CITATION_VENVS + '/aclarc_multicite/bin/python'


def main():
    stream = RUN/'classifier/full_v4_stream/stream_complete.json'
    started = time.monotonic()
    while not stream.exists():
        if time.monotonic()-started > 14400:
            raise SystemExit('Stream has not completed; no final classifier status claimed')
        time.sleep(10)
    jobs = [
        ('full_v4', RUN/'full/reconstructed_pdf_1200_v4.jsonl', ['legacy'], True),
        ('round2_repaired_v4', RUN/'fresh_round2/reconstructed_pdf_100.jsonl', ['legacy', 'source_target'], False),
        ('development_repaired_v4', RUN/'pdf_development/reconstructed_v4_all200.jsonl', ['legacy'], False),
    ]
    completed = []
    for tag, inp, branches, chi in jobs:
        manifest = inp.with_suffix('.manifest.json')
        if not manifest.exists():
            raise SystemExit(f'Missing completed extraction manifest: {manifest}')
        out = RUN/'classifier'/tag
        out.mkdir(exist_ok=True)
        summary = out/'summary.json'
        if summary.exists():
            s = json.loads(summary.read_text())
            if s['input_sha256'] != hashlib.sha256(inp.read_bytes()).hexdigest():
                raise SystemExit('Existing classifier result does not match input')
        else:
            cmd = [PYTHON, str(Path(__file__).with_name('44_nature_multicite_repair.py')),
                   '--input', str(inp), '--out', str(out), '--cache', str(RUN/'classifier/cache.jsonl'),
                   '--branches', *branches, '--reference-id-space', 'grobid', '--threads', '2', '--batch', '4']
            if chi:
                cmd.append('--chi-comparison')
            print(json.dumps({'stage': 'start', 'job': tag}), flush=True)
            with (out/'run.log').open('a') as log:
                subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True)
        completed.append(tag)
        print(json.dumps({'stage': 'completed', 'job': tag}), flush=True)
    (RUN/'classifier/final_runs_complete.json').write_text(json.dumps({
        'status': 'complete revised MultiCite runs; original SciCite results not revised',
        'jobs': completed, 'elapsed_seconds': time.monotonic()-started}, indent=2)+'\n')


if __name__ == '__main__':
    main()
