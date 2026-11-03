#!/usr/bin/env python3
"""Fixed-sample inference configuration benchmark; reports numerical differences."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
HARNESS = ROOT / 'analysis/external_systems/aclarc_multicite'
sys.path.insert(0, str(HARNESS))
import run_multicite as model
import markers
import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True, type=Path)
    ap.add_argument('--out', required=True, type=Path)
    ap.add_argument('--sample', default=256, type=int)
    args = ap.parse_args()
    if args.out.exists():
        raise SystemExit('Output exists')
    texts = set()
    for line in args.input.open():
        r = json.loads(line)
        texts.update(markers.mark_edge(r['citances'], 'nature_oa')[0])
        for text, p in zip(r.get('marked_citances', []), r.get('context_provenance', [])):
            if text and p.get('target_span_unambiguous') is True and p.get('target_group_size') == 1:
                texts.add(text)
    sample = random.Random(20261025).sample(sorted(texts), min(args.sample, len(texts)))
    model.set_determinism(2)
    tok, net, labels = model.load('scibert')
    # Exclude one small warmup from all reported timing.
    model.predict(tok, net, sample[:4], 4)
    runs = []
    baseline = None
    eligible = [i for i, label in enumerate(labels) if label in {'uses', 'extends', 'motivation', 'differences', 'background'}]
    def decisions(p):
        return [max([(float(row[i]), labels[i]) for i in eligible if row[i] > .5], default=(0, 'no_claim'))[1] for row in p]
    for threads, batch in [(2, 4), (4, 8), (4, 16)]:
        torch.set_num_threads(threads)
        started = time.monotonic()
        p = model.predict(tok, net, sample, batch)
        elapsed = time.monotonic() - started
        if baseline is None:
            baseline = p.copy()
        result = {'threads': threads, 'batch': batch, 'seconds': elapsed,
                  'texts_per_second': len(sample) / elapsed,
                  'maximum_absolute_probability_difference': float(np.max(np.abs(p - baseline))),
                  'threshold_0_5_binary_decision_differences': int(np.sum((p > .5) != (baseline > .5))),
                  'eligible_max_label_differences': sum(a != b for a, b in zip(decisions(p), decisions(baseline))),
                  'bitwise_equal': bool(np.array_equal(p, baseline))}
        runs.append(result)
        print(json.dumps(result), flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({'status': 'Performance benchmark, not research validation',
         'model': model.MODELS['scibert'], 'sample_texts': len(sample),
         'sample_seed': 20261025, 'sample_sha256': hashlib.sha256(json.dumps(sample).encode()).hexdigest(),
         'input_sha256': hashlib.sha256(args.input.read_bytes()).hexdigest(),
         'runs': runs, 'caveat': 'No result establishes general numerical equivalence; original frozen cached probabilities retain their original inference history.'}, indent=2) + '\n')


if __name__ == '__main__':
    main()
