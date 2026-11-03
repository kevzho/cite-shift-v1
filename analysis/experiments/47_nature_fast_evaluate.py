#!/usr/bin/env python3
"""Execute frozen evaluator rules with a safe SequenceMatcher upper-bound filter.

quick_ratio is an upper bound on ratio. Rejecting candidates whose upper bound
is below the existing 0.9 threshold cannot change any accepted title match.
Original evaluator files are preserved. Snapshot/hash is saved before execution.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--evaluator', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--reference', type=Path, required=True)
    p.add_argument('--jats-dir', type=Path, required=True)
    p.add_argument('--predictions', type=Path, required=True)
    p.add_argument('--parser', type=Path, required=True)
    p.add_argument('--frozen-hash', required=True)
    p.add_argument('--tag', required=True)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    source = a.evaluator.read_text()
    old = 'score=SequenceMatcher(None,t,pt).ratio()'
    new = 'sm=SequenceMatcher(None,t,pt); score=sm.ratio() if sm.quick_ratio()>=.9 else 0.0'
    if source.count(old) != 1:
        raise SystemExit('Expected precisely one title-ratio expression')
    source = source.replace(old, new)
    snapshot = a.out / ('snapshot_' + a.tag + '_' + a.evaluator.name)
    if snapshot.exists():
        raise SystemExit('Snapshot exists; refusing overwrite')
    snapshot.write_text(source)
    reference_name = 'jats_reference_100.jsonl' if 'jats_reference_100.jsonl' in source else 'jats_reference_50.jsonl'
    link = a.out / reference_name
    if not link.exists():
        link.symlink_to(a.reference.resolve())
    xml = a.out / 'jats'
    if not xml.exists():
        xml.symlink_to(a.jats_dir.resolve(), target_is_directory=True)
    freeze = {'original_evaluator_sha256': sha(a.evaluator), 'snapshot_sha256': sha(snapshot),
              'parser_sha256': a.frozen_hash,
              'optimization': 'quick_ratio upper-bound <0.9 prefilter only; accepted scores unchanged',
              'frozen_before_evaluation': True}
    (a.out / ('evaluator_snapshot_' + a.tag + '.json')).write_text(json.dumps(freeze, indent=2)+'\n')
    spec = importlib.util.spec_from_file_location('fast_nature_evaluator_' + a.tag, snapshot)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    m.OUT = a.out.resolve()
    m.evaluate(str(a.predictions), str(a.parser), a.frozen_hash, a.tag)


if __name__ == '__main__':
    main()
