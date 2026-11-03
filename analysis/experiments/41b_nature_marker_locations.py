#!/usr/bin/env python3
"""Locate TEI markers for bibliography matches with no exported citance edge.

Development diagnostic only. Does not alter extraction or classifier inputs.
"""
import csv
import json
from collections import Counter
from pathlib import Path
from xml.etree import ElementTree as ET

REPO = Path(__file__).resolve().parents[2]
ARM = REPO / 'external_data/nature_arm_2026-10-04'
OUT = REPO / 'external_data/nature_repair_2026-10-25'
NS = {'t': 'http://www.tei-c.org/ns/1.0'}


def run():
    rows = [x for x in csv.DictReader((OUT/'unmatched_reference_inspection.csv').open())
            if x['status']=='bibliography_present_no_extracted_edge' and int(x['tei_marker_count'])>0]
    paths = {p.name.removesuffix('.tei.xml'):p for p in (ARM/'pdfs_grobid').rglob('*.tei.xml')}
    examples = []
    for pid in sorted({x['pmcid'] for x in rows}):
        tree = ET.parse(paths[pid])
        parent = {x:p for p in tree.iter() for x in p}
        body = tree.find('.//t:text/t:body',NS)
        for row in [x for x in rows if x['pmcid']==pid]:
            for el in tree.findall('.//t:ref[@type="bibr"]',NS):
                if '#'+row['grobid_ref'] not in (el.get('target') or '').split():
                    continue
                ancestors = []
                a = el
                while a in parent:
                    a = parent[a]
                    ancestors.append(a)
                sentence = next((a for a in ancestors if a.tag.endswith('}s')),None)
                sent = ' '.join(''.join(sentence.itertext()).split()) if sentence is not None else ''
                inbody = body in ancestors
                div = any(a.tag.endswith('}div') for a in ancestors)
                category = ('outside_body' if not inbody else 'no_sentence' if sentence is None
                            else 'no_div' if not div else 'short_sentence' if len(sent)<20
                            else 'eligible_should_extract')
                examples.append({'pmcid':pid,'jats_ref':row['jats_ref'],
                    'grobid_ref':row['grobid_ref'],'category':category,
                    'ancestry':'/'.join(a.tag.split('}')[-1] for a in ancestors),
                    'text':sent or ''.join(parent[el].itertext())[:1500]})
    with (OUT/'missing_export_marker_locations.csv').open('w') as f:
        w = csv.DictWriter(f,fieldnames=list(examples[0]))
        w.writeheader()
        w.writerows(examples)
    summary = {'status':'exploratory diagnostic; no extraction changes',
        'reference_candidates_with_markers':len(rows),'marker_occurrences':len(examples),
        'occurrence_categories':dict(Counter(x['category'] for x in examples)),
        'caution':'Outside-body markers may be spurious bibliography links, including affiliation numbers. Do not import all as citation contexts.',
        'next_step':'Inspect real body contexts in PDF/TEI and reconstruct short sentence segments conservatively; validate before rerunning classifiers.'}
    (OUT/'marker_locations_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    run()
