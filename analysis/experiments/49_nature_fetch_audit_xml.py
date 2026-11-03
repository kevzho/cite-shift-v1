#!/usr/bin/env python3
"""Fetch a fixed public-PMC audit sample without replacing papers or inputs."""
import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
ARM = ROOT/'external_data/nature_arm_2026-10-04'
S3 = 'https://pmc-oa-opendata.s3.amazonaws.com/'


def get(url):
    request = urllib.request.Request(url, headers={'User-Agent': 'Nature-parser-audit/0.1 academic research'})
    last = None
    for attempt in range(4):
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                return response.read()
        except Exception as e:
            last = e
            time.sleep(2**attempt)
    raise last


def fetch(metadata):
    url = metadata['xml_url'].replace('s3://pmc-oa-opendata/', S3)
    data = get(url.split('?')[0])
    expected = re.search(r'md5=([a-f0-9]{32})', url)
    return data, bool(expected and hashlib.md5(data).hexdigest() == expected[1])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--papers', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    xml = a.out/'jats'
    xml.mkdir(exist_ok=True)
    records = list(csv.DictReader(a.papers.open()))
    assert len(records) == len({r['pmcid'] for r in records})
    results = []
    for i, row in enumerate(records, 1):
        pid = row['pmcid']
        meta = json.loads((ARM/f'logs/meta/{pid}.json').read_text())
        file = xml/f'{pid}.xml'
        if file.exists():
            data = file.read_bytes()
            expected = re.search(r'md5=([a-f0-9]{32})', meta['xml_url'])
            valid = bool(expected and hashlib.md5(data).hexdigest() == expected[1])
        else:
            data, valid = fetch(meta)
        refreshed = False
        if not valid:
            current = json.loads(get(S3+meta['_key']))
            if current.get('pdf_url') != meta.get('pdf_url'):
                raise RuntimeError(f'{pid}: PDF version changed; do not silently substitute')
            data, valid = fetch(current)
            if not valid:
                raise RuntimeError(f'{pid}: current XML checksum also failed')
            (a.out/f'{pid}_refreshed_metadata.json').write_text(json.dumps(current, indent=2)+'\n')
            refreshed = True
        if b'<article' not in data[:10000]:
            raise RuntimeError(f'{pid}: not article XML')
        if file.exists() and file.read_bytes() != data:
            raise RuntimeError(f'{pid}: existing audit XML differs; preserve and inspect')
        file.write_bytes(data)
        results.append({'pmcid': pid, 'sha256': hashlib.sha256(data).hexdigest(),
                        'bytes': len(data), 'metadata_refreshed_same_pdf': refreshed, 'md5_verified': valid})
        print(f'{i}/{len(records)} {pid} verified', flush=True)
        time.sleep(.25)
    extractor = ROOT/'external_data/nature_oa_jats_pilot_2026-09-27/code/extract_citances_jats.py'
    spec = importlib.util.spec_from_file_location('frozen_reference_extractor', extractor)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    out = a.out/'jats_reference_50.jsonl'
    if out.exists():
        raise SystemExit('Reference exists; refusing overwrite')
    with out.open('x') as f:
        for row in records:
            for edge in m.extract(xml/f"{row['pmcid']}.xml", venue=row['journal']):
                f.write(json.dumps(edge, ensure_ascii=False)+'\n')
    (a.out/'acquisition_result.json').write_text(json.dumps({
        'papers': len(records), 'files': results,
        'reference_sha256': hashlib.sha256(out.read_bytes()).hexdigest(),
        'extractor_sha256': hashlib.sha256(extractor.read_bytes()).hexdigest(),
        'selection_sha256': hashlib.sha256(a.papers.read_bytes()).hexdigest()}, indent=2)+'\n')


if __name__ == '__main__':
    main()
