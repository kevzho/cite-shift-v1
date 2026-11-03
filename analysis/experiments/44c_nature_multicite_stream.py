#!/usr/bin/env python3
"""Warm exact-text cache on completed-paper chunks during PDF extraction.

Chunks and their predictions are nonfinal. A separate authoritative full run
must use the complete extraction manifest. Never includes the current final
paper before that manifest exists; skips a partial trailing JSON record.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True, type=Path)
    ap.add_argument('--manifest', required=True, type=Path)
    ap.add_argument('--cache', required=True, type=Path)
    ap.add_argument('--out', required=True, type=Path)
    ap.add_argument('--papers-per-chunk', type=int, default=100)
    ap.add_argument('--max-idle-seconds', type=int, default=1200)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    completed = set()
    for p in sorted(args.out.glob('chunk_*/summary.json')):
        original = p.parent / 'input.jsonl'
        for line in original.open():
            completed.add(json.loads(line)['citing_paper_id'])
    last_progress = time.monotonic()
    last_size = -1
    while True:
        data = {}
        last_paper = None
        if args.input.exists():
            size = args.input.stat().st_size
            if size != last_size:
                last_progress, last_size = time.monotonic(), size
            with args.input.open() as f:
                for line in f:
                    try:
                        r = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    last_paper = r['citing_paper_id']
                    if last_paper not in completed:
                        data.setdefault(last_paper, []).append(line)
        manifest_complete = args.manifest.exists()
        if not manifest_complete and last_paper is not None:
            data.pop(last_paper, None)
        selected = list(data)[:args.papers_per_chunk]
        if len(selected) < args.papers_per_chunk and not manifest_complete:
            selected = []
        if selected:
            number = len(list(args.out.glob('chunk_*/summary.json'))) + 1
            folder = args.out / f'chunk_{number:03d}'
            folder.mkdir(exist_ok=True)
            inp = folder / 'input.jsonl'
            content = ''.join(line for pid in selected for line in data[pid])
            inp.write_text(content)
            (folder / 'chunk_manifest.json').write_text(json.dumps({
                'status': 'NONFINAL cache warmup; completed-paper snapshot',
                'papers': selected, 'source': str(args.input),
                'input_sha256': hashlib.sha256(content.encode()).hexdigest(),
                'excluded_current_writing_paper': not manifest_complete,
                'full_source_manifest_present': manifest_complete}, indent=2) + '\n')
            cmd = [sys.executable, str(Path(__file__).with_name('44_nature_multicite_repair.py')),
                   '--input', str(inp), '--out', str(folder), '--cache', str(args.cache),
                   '--branches', 'legacy', '--reference-id-space', 'grobid',
                   '--threads', '2', '--batch', '4', '--bootstrap', '0']
            print(json.dumps({'stage': 'chunk_start', 'chunk': number, 'papers': len(selected),
                              'previously_completed_papers': len(completed)}), flush=True)
            with (folder / 'run.log').open('a') as log:
                subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True)
            completed.update(selected)
            last_progress = time.monotonic()
            summary = json.loads((folder / 'summary.json').read_text())
            print(json.dumps({'stage': 'chunk_complete', 'chunk': number,
                              'completed_papers': len(completed),
                              'new_model_texts': summary['new_inference_unique_texts']}), flush=True)
            continue
        if manifest_complete:
            (args.out / 'stream_complete.json').write_text(json.dumps({
                'status': 'NONFINAL cache warmup complete; authoritative full run still required',
                'completed_papers': len(completed), 'full_input': str(args.input),
                'full_input_sha256': hashlib.sha256(args.input.read_bytes()).hexdigest(),
                'manifest': str(args.manifest)}, indent=2) + '\n')
            print(json.dumps({'stage': 'stream_complete', 'completed_papers': len(completed)}), flush=True)
            break
        if time.monotonic() - last_progress > args.max_idle_seconds:
            raise SystemExit('No extraction progress or manifest before idle timeout; incomplete stream preserved.')
        time.sleep(10)


if __name__ == '__main__':
    main()
