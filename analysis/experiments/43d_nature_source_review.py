#!/usr/bin/env python3
"""Prepare source evidence for fixed 30-context AI-assisted audit, no auto judgments."""
from __future__ import annotations
import argparse,csv,hashlib,json,random,re
from collections import Counter,defaultdict
from pathlib import Path
from xml.etree import ElementTree as ET
from pypdf import PdfReader

REPO=Path(__file__).resolve().parents[2]
ARM=REPO/'external_data/nature_arm_2026-10-04'
OUT=REPO/'external_data/nature_repair_v2_2026-10-25/fresh'
def load(p):return [json.loads(x) for x in Path(p).open()]
def tok(s):return set(re.findall('[a-z]{3,}',s.lower()))
def rec(a,b):return len(tok(a)&tok(b))/max(1,len(tok(a)))
def quant(v,q):
    v=sorted(v);return v[min(len(v)-1,int(q*(len(v)-1)))] if v else None

def main(revised):
    target=OUT/'source_review_evidence.json'
    if target.exists():raise SystemExit('Source-review evidence exists; not overwriting')
    new=load(revised); fresh={r['pmcid'] for r in csv.DictReader((REPO/'external_data/nature_repair_2026-10-25/fresh_50_paper_validation_plan.csv').open())}
    old=[r for r in load(ARM/'citances/nature_arm_citances.jsonl') if r['citing_paper_id'] in fresh]
    oldstrings=defaultdict(set)
    for r in old:oldstrings[r['citing_paper_id']].update(r['citances'])
    refs={(r['citing_paper_id'],r['cited_ref_id']):r for r in load(OUT/'jats_reference_50.jsonl')}
    preds={(r['citing_paper_id'],r['cited_ref_id']):r for r in new}
    audit=list(csv.DictReader((OUT/'evaluation_edges_revised_enhanced.csv').open()))
    population=[]
    for a in audit:
        if not a['predicted_ref']:continue
        p=preds[(a['pmcid'],a['predicted_ref'])];ref=refs[(a['pmcid'],a['jats_ref'])]
        for i,c in enumerate(p['citances']):
            if c in oldstrings[a['pmcid']]:continue
            provenance=p['context_provenance'][i]
            if not provenance['source'].startswith('pdf'):continue
            jc=max(ref['citances'],key=lambda x:rec(x,c))
            if rec(jc,c)<.8:continue
            population.append({'pmcid':a['pmcid'],'jats_ref':a['jats_ref'],'predicted_ref':a['predicted_ref'],
                'predicted_title':p.get('cited_title',''),'jats_title':ref.get('cited_title',''),
                'predicted_context':c,'jats_context':jc,'token_recall':rec(jc,c),
                'provenance':provenance,'predicted_reference':p.get('cited_raw_reference',''),
                'pdf_raw_references':p.get('pdf_raw_references',[])})
    population.sort(key=lambda x:(x['pmcid'],x['jats_ref'],x['predicted_context']))
    rng=random.Random(20261025);sample=rng.sample(population,min(30,len(population)))
    readers={};xmlroots={}
    for n,r in enumerate(sample,1):
        r['review_id']=n;pid=r['pmcid']
        if pid not in readers:readers[pid]=PdfReader(ARM/'pdfs'/f'{pid}.pdf')
        page=r['provenance'].get('page')
        raw=readers[pid].pages[page-1].extract_text() if page else ''
        r['pdf_page_text']=raw
        prev=r['provenance'].get('previous_source') or {}
        pp=prev.get('page')
        r['previous_pdf_page_text']=readers[pid].pages[pp-1].extract_text() if pp and pp!=page else ''
        if pid not in xmlroots:xmlroots[pid]=ET.parse(OUT/'jats'/f'{pid}.xml').getroot()
        node=xmlroots[pid].find(f".//ref-list/ref[@id='{r['jats_ref']}']")
        r['jats_raw_reference']=' '.join(node.itertext()) if node is not None else ''
        r['assessment']='';r['assessment_notes']=''
        print(f'Prepared source evidence {n}/{len(sample)} {pid}',flush=True)
    def sizes(rr):
        v=[len(c.split()) for r in rr for c in r['citances']]
        return {'edges':len(rr),'contexts':len(v),'median_words':quant(v,.5),'p95_words':quant(v,.95),'max_words':max(v,default=0)}
    source=Counter();fragment=Counter();sourcewords=defaultdict(list)
    for r in new:
        for c,p in zip(r['citances'],r['context_provenance']):
            s=p['source'];source[s]+=1;sourcewords[s].append(len(c.split()))
            if p.get('fragment_reconstructed'):fragment[s]+=1
    stats={'baseline':sizes(old),'revised':sizes(new),'context_sources':dict(source),
        'fragment_reconstructed':dict(fragment),'by_source':{s:{'median_words':quant(v,.5),'p95_words':quant(v,.95),'max_words':max(v)} for s,v in sourcewords.items()},
        'new_recovered_pdf_context_population':len(population),'source_review_sample':len(sample),
        'source_review_status':'evidence pack only; judgments not automatically inferred from overlap',
        'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (OUT/'context_length_diagnostics.json').write_text(json.dumps(stats,indent=2)+'\n')
    target.write_text(json.dumps({'sampling_seed':20261025,'population_size':len(population),'assessment_type':'AI-assisted source inspection, not independent human validation','items':sample},indent=2,ensure_ascii=False)+'\n')
    with (OUT/'independent_human_source_review.csv').open('w',newline='') as f:
        fields=['review_id','pmcid','jats_ref','predicted_ref','predicted_context','jats_context','assessment','assessment_notes']
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows({k:r[k] for k in fields} for r in sample)
    print(json.dumps(stats,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--revised',required=True);a=p.parse_args();main(a.revised)
