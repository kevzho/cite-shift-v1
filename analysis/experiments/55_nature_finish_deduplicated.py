#!/usr/bin/env python3
"""Compare format-deduplicated inputs after raw-format cache completion."""
import os as _os
from pathlib import Path as _Path
_CITATION_VENVS = _os.environ.get("CITATION_VENVS", _os.path.expanduser("~/.cache/citation_corpus/venvs"))
import json
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT/'external_data/nature_repair_v2_2026-10-25'
PYTHON = _CITATION_VENVS + '/aclarc_multicite/bin/python'


def main():
    started = time.monotonic()
    while not (RUN/'classifier/final_v6_runs_complete.json').exists():
        if time.monotonic()-started > 14400:
            raise SystemExit('Raw-format cache runs incomplete')
        time.sleep(10)
    jobs = [
        ('full_v6_dedup', RUN/'full/reconstructed_pdf_1200_v6_dedup.jsonl', ['legacy'], True),
        ('round4_repaired_v6_dedup', RUN/'fresh_round4/reconstructed_pdf_50_dedup.jsonl', ['legacy', 'source_target'], False),
    ]
    for tag, inp, branches, chi in jobs:
        out = RUN/'classifier'/tag
        out.mkdir(exist_ok=True)
        if (out/'summary.json').exists():
            continue
        cmd = [PYTHON, str(Path(__file__).with_name('44_nature_multicite_repair.py')),
               '--input', str(inp), '--out', str(out), '--cache', str(RUN/'classifier/cache.jsonl'),
               '--branches', *branches, '--reference-id-space', 'grobid', '--threads', '2', '--batch', '4']
        if chi:
            cmd.append('--chi-comparison')
        print(json.dumps({'stage': 'start', 'job': tag}), flush=True)
        with (out/'run.log').open('a') as log:
            subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True)
        print(json.dumps({'stage': 'complete', 'job': tag}), flush=True)
    (RUN/'classifier/all_final_runs_complete.json').write_text(json.dumps({
        'status': 'complete raw-format and conservative lexical-deduplicated revised MultiCite runs',
        'elapsed_seconds': time.monotonic()-started}, indent=2)+'\n')


if __name__ == '__main__':
    main()
