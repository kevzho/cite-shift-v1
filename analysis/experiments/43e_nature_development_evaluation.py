#!/usr/bin/env python3
"""Run unchanged evaluator algorithms on old200 development data, separate snapshots."""
import argparse,hashlib,importlib.util,json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
ARM=ROOT/'external_data/nature_arm_2026-10-04'
EXPERIMENTS=Path(__file__).parent

def main(predictions,parser,frozen_hash,tag):
    base=ROOT/'external_data/nature_repair_v2_2026-10-25/pdf_development'/f'eval_{tag}'
    base.mkdir(exist_ok=False)
    (base/'jats_reference_200.jsonl').symlink_to(ARM/'parity/jats_citances_200.jsonl')
    (base/'jats').symlink_to(ARM/'jats',target_is_directory=True)
    hashes={}
    for name,tag2 in [('43_nature_fresh_validation.py','original_alignment'),('43c_nature_enhanced_validation.py','enhanced_alignment')]:
        source=EXPERIMENTS/name
        # Only paths and study-status text change; algorithms remain frozen.
        code=source.read_text().replace("OUT = REPO / 'external_data/nature_repair_v2_2026-10-25/fresh'",f'OUT = Path({str(base)!r})')
        code=code.replace('jats_reference_50.jsonl','jats_reference_200.jsonl')
        code=code.replace('EXPLORATORY fresh validation; no retuning allowed on this holdout','EXPLORATORY development evaluation on previously examined original200; NOT fresh validation')
        snapshot=base/name;snapshot.write_text(code)
        spec=importlib.util.spec_from_file_location('dev_'+tag2,snapshot)
        m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
        m.evaluate(predictions,parser,frozen_hash,tag2)
        hashes[name]={'original_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'snapshot_sha256':hashlib.sha256(snapshot.read_bytes()).hexdigest()}
    (base/'evaluation_snapshots.json').write_text(json.dumps(hashes,indent=2)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--predictions',required=True);p.add_argument('--parser',required=True);p.add_argument('--frozen-hash',required=True);p.add_argument('--tag',required=True)
    a=p.parse_args();main(a.predictions,a.parser,a.frozen_hash,a.tag)
