#!/usr/bin/env python3
"""Collapse exact lexical duplicates across TEI/PDF citation formatting.

Retains numeric values and operators. Expands short integer ranges and ignores
bracket/punctuation/whitespace differences only when resulting token sequences
are identical. No fuzzy text similarity or reference labels are used. This is
classifier preprocessing; frozen raw parser outputs remain unchanged.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import unicodedata


def key(text):
    text = unicodedata.normalize('NFKC', text).lower()
    def expand(m):
        lo, hi = int(m[1]), int(m[2])
        return ','.join(map(str, range(lo, hi+1))) if 0 < hi-lo <= 40 else m[0]
    text = re.sub(r'(?<!\d)(\d{1,3})\s*[–−-]\s*(\d{1,3})(?!\d)', expand, text)
    return tuple(re.findall(r'[^\W\d_]+|\d+(?:\.\d+)?|[+−=<>≤≥-]', text, re.UNICODE))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    if a.out.exists():
        raise SystemExit('Refusing overwrite')
    assert key('Sentence 1–3.') == key('Sentence [1,2,3].')
    assert key('Dose 5 mg [1].') != key('Dose 10 mg 1.')
    assert key('Sentence [1,23].') != key('Sentence [12,3].')
    assert key('Value 1.2 [1].') != key('Value 12 [1].')
    assert key('Value −1 [1].') != key('Value 1 [1].')
    counts = Counter()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open('x') as f:
        for line in a.input.open():
            row = json.loads(line)
            provenance = row.get('context_provenance', [{} for _ in row['citances']])
            marked = row.get('marked_citances', [None for _ in row['citances']])
            selected = {}
            for i, (context, p) in enumerate(zip(row['citances'], provenance)):
                k = key(context)
                rank = (int(p.get('source', '').startswith('pdf')), int(p.get('target_span_unambiguous', False)))
                if k not in selected or rank > selected[k][0]:
                    selected[k] = (rank, i)
            indices = sorted(v[1] for v in selected.values())
            counts['edges'] += 1
            counts['raw_contexts'] += len(row['citances'])
            counts['retained_contexts'] += len(indices)
            row['citances'] = [row['citances'][i] for i in indices]
            row['marked_citances'] = [marked[i] for i in indices]
            row['context_provenance'] = [provenance[i] for i in indices]
            row['sections'] = [row.get('sections', [''] * len(provenance))[i] for i in indices]
            row['n_citances'] = len(indices)
            f.write(json.dumps(row, ensure_ascii=False)+'\n')
    manifest = {'status': 'complete lexical-format deduplication; raw extraction unchanged',
                'input_sha256': hashlib.sha256(a.input.read_bytes()).hexdigest(),
                'output_sha256': hashlib.sha256(a.out.read_bytes()).hexdigest(),
                'preprocessor_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'counts': dict(counts), 'dropped_contexts': counts['raw_contexts']-counts['retained_contexts'],
                'selection': 'prefer PDF source provenance among identical token sequences; no fuzzy merging',
                'inference': 'same pinned classifier; compare deduplicated result to raw-format sensitivity'}
    a.out.with_suffix('.manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == '__main__':
    main()
