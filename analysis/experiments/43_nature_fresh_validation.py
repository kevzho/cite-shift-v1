#!/usr/bin/env python3
"""Fetch locked fresh Nature audit references and evaluate independent predictions.

Publisher XML is audit reference only, never a reconstruction-parser input.
The original 200-paper audit and registered RQ4(b) status are preserved.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import importlib.util
import json
import random
import re
import time
import urllib.request
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ARM = REPO / 'external_data/nature_arm_2026-10-04'
OUT = REPO / 'external_data/nature_repair_v2_2026-10-25/fresh'
PLAN = REPO / 'external_data/nature_repair_2026-10-25/fresh_50_paper_validation_plan.csv'
EXTRACTOR = REPO / 'external_data/nature_oa_jats_pilot_2026-09-27/code/extract_citances_jats.py'

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def dump(p, obj):
    p.write_text(json.dumps(obj, indent=2) + '\n')

def rows(p):
    return [json.loads(x) for x in Path(p).open()]

def fetch():
    OUT.mkdir(parents=True, exist_ok=True)
    xml = OUT/'jats'; xml.mkdir(exist_ok=True)
    plan = list(csv.DictReader(PLAN.open()))
    original = {x['pmcid'] for x in csv.DictReader((ARM/'parity/parity_gate_edges.csv').open())}
    assert len(plan) == len({x['pmcid'] for x in plan}) == 50
    assert not ({x['pmcid'] for x in plan} & original)
    frozen = {'status':'fresh references acquisition only; no outcome examined',
              'sample_sha256':sha(PLAN), 'reference_extractor_sha256':sha(EXTRACTOR),
              'papers':plan, 'excluded_original_200':True,
              'sampling':'up to 3 per journal, then random remainder; descriptive not population-weighted'}
    freeze = OUT/'reference_acquisition_freeze.json'
    if freeze.exists():
        assert json.loads(freeze.read_text())['sample_sha256'] == frozen['sample_sha256']
    else:
        dump(freeze, frozen)
    t0 = time.time(); records=[]
    for i, r in enumerate(plan):
        pid=r['pmcid']; meta=json.loads((ARM/f'logs/meta/{pid}.json').read_text())
        url=meta['xml_url'].replace('s3://pmc-oa-opendata/', 'https://pmc-oa-opendata.s3.amazonaws.com/')
        path=xml/f'{pid}.xml'; expected=re.search(r'md5=([a-f0-9]{32})',url)
        if not path.exists():
            last=None
            for attempt in range(5):
                try:
                    req=urllib.request.Request(url.split('?')[0],headers={'User-Agent':'Nature-parser-validation/0.1 academic research'})
                    with urllib.request.urlopen(req,timeout=90) as response: data=response.read()
                    assert b'<article' in data[:10000], 'not JATS article'
                    if expected: assert hashlib.md5(data).hexdigest()==expected[1], 'MD5 mismatch'
                    path.write_bytes(data); last=None; break
                except Exception as exc:
                    last=repr(exc)
                    with (OUT/'download_log.jsonl').open('a') as f:
                        f.write(json.dumps({'pmcid':pid,'attempt':attempt,'error':last})+'\n')
                    if attempt < 4: time.sleep(min(2**attempt,10))
            if last: raise RuntimeError(f'{pid}: {last}')
            time.sleep(.25)
        data=path.read_bytes()
        if expected: assert hashlib.md5(data).hexdigest()==expected[1]
        records.append({'pmcid':pid,'sha256':sha(path),'bytes':len(data)})
        print(f'{i+1}/50 {pid} {len(data)} bytes',flush=True)
    spec=importlib.util.spec_from_file_location('frozen_jats_extractor',EXTRACTOR)
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    ref=OUT/'jats_reference_50.jsonl'
    if ref.exists(): raise SystemExit('Reference already exists; fetched XML verified, not overwriting.')
    with ref.open('w') as f:
        for r in plan:
            for edge in module.extract(xml/f"{r['pmcid']}.xml", venue=r['journal']):
                f.write(json.dumps(edge,ensure_ascii=False)+'\n')
    dump(OUT/'acquisition_result.json',{'status':'fresh reference acquired; not validation result',
        'papers':len(records),'files':records,'seconds':round(time.time()-t0,2),
        'reference_sha256':sha(ref),'extractor_sha256':sha(EXTRACTOR),'sample_sha256':sha(PLAN)})
    print('Fresh references ready; metrics intentionally not computed.',flush=True)

def norm(s): return re.sub('[^a-z0-9]','',(s or '').lower())
def tok(s): return set(re.findall('[a-z]{3,}',(s or '').lower()))
def recall(a,b): return len(tok(a)&tok(b))/max(1,len(tok(a)))

def evaluate(predictions, parser, frozen_hash, tag):
    if sha(parser)!=frozen_hash: raise SystemExit('Parser hash changed after freeze; fresh evaluation refused')
    target=OUT/f'evaluation_{tag}.json'
    if target.exists(): raise SystemExit('Refusing to overwrite fresh evaluation')
    refs=rows(OUT/'jats_reference_50.jsonl'); preds=rows(predictions)
    byp=defaultdict(list)
    for p in preds: byp[p['citing_paper_id']].append(p)
    audits=[]; per=defaultdict(Counter); targets=Counter()
    for r in refs:
        pid=r['citing_paper_id']; options=byp[pid]
        doi=r.get('cited_doi','').lower()
        hits=[p for p in options if doi and p.get('cited_doi','').lower()==doi]
        how='doi'
        if not hits:
            t=norm(r.get('cited_title','')); scored=[]
            for p in options:
                pt=norm(p.get('cited_title',''))
                if t and pt:
                    score=SequenceMatcher(None,t,pt).ratio()
                    if score>=.9: scored.append((score,p))
            scored.sort(key=lambda x:x[0],reverse=True)
            # Retain ambiguity instead of selecting an arbitrary equal-title candidate.
            hits=[p for s,p in scored if s==scored[0][0]] if scored else []
            how='title'
        unique=len(hits)==1
        p=hits[0] if unique else None
        recovered=[any(recall(c,x)>=.8 for x in p['citances']) if p else False for c in r['citances']]
        full=bool(p) and all(recovered)
        if p: targets[(pid,p['cited_ref_id'])]+=1
        per[pid].update({'contexts':len(recovered),'recovered':sum(recovered),'edges':1,'complete':int(full)})
        audits.append({'pmcid':pid,'jats_ref':r['cited_ref_id'],'predicted_ref':p['cited_ref_id'] if p else '',
            'matching':how if unique else ('ambiguous' if hits else 'none'),
            'n_contexts':len(recovered),'n_recovered':sum(recovered),'complete':int(full),
            'jats_title':r.get('cited_title',''),'predicted_title':p.get('cited_title','') if p else '',
            'jats_contexts':json.dumps(r['citances'],ensure_ascii=False),
            'predicted_contexts':json.dumps(p['citances'],ensure_ascii=False) if p else '',
            'manual_match_correct':'','manual_context_correct':''})
    keys=sorted(per)
    def rate(ps):
        counts=sum((per[x] for x in ps),Counter())
        return counts['recovered']/counts['contexts'],counts['complete']/counts['edges']
    cr,er=rate(keys); rng=random.Random(20261025); cs=[]; es=[]
    for _ in range(2000):
        c,e=rate(rng.choices(keys,k=len(keys)));cs.append(c);es.append(e)
    def ci(v):
        v=sorted(v); return [v[int(.025*len(v))],v[int(.975*len(v))-1]]
    collision_refs = [a for a in audits if a['predicted_ref'] and targets[(a['pmcid'],a['predicted_ref'])]>1]
    strict_rec = sum(a['n_recovered'] for a in audits if not a['predicted_ref'] or targets[(a['pmcid'],a['predicted_ref'])]==1)
    strict_full = sum(a['complete'] for a in audits if not a['predicted_ref'] or targets[(a['pmcid'],a['predicted_ref'])]==1)
    n_contexts=sum(len(r['citances']) for r in refs)
    dump(target,{'status':'EXPLORATORY fresh validation; no retuning allowed on this holdout',
        'parser_sha256':frozen_hash,'predictions_sha256':sha(predictions),'reference_sha256':sha(OUT/'jats_reference_50.jsonl'),
        'papers':len(keys),'reference_edges':len(refs),'reference_contexts':sum(len(r['citances']) for r in refs),
        'citance_recovery':cr,'citance_recovery_ci95':ci(cs),'complete_edge_match':er,'complete_edge_match_ci95':ci(es),
        'target_recovery':.95,'achieved_95_percent_point_estimate':cr>=.95,
        'duplicate_target_groups':sum(v>1 for v in targets.values()),
        'reference_edges_in_duplicate_target_groups':len(collision_refs),
        'strict_unique_citance_recovery':strict_rec/n_contexts,
        'strict_unique_complete_edge_match':strict_full/len(refs),
        'ambiguity_counts':dict(Counter(a['matching'] for a in audits)),
        'precision_caveat':'Metadata and token recall are automatic proxies, not independently labeled precision. Manual judgments remain unfilled.',
        'historical_RQ4b':'original failed gate/not run unchanged'})
    with (OUT/f'evaluation_edges_{tag}.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(audits[0]));w.writeheader();w.writerows(audits)
    rng.shuffle(audits)
    sample=audits[:100]
    with (OUT/f'precision_review_{tag}.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(sample[0]));w.writeheader();w.writerows(sample)
    print(target.read_text())

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['fetch','evaluate'])
    p.add_argument('--predictions');p.add_argument('--parser');p.add_argument('--frozen-hash');p.add_argument('--tag',default='frozen')
    a=p.parse_args()
    if a.action=='fetch':fetch()
    else:evaluate(a.predictions,a.parser,a.frozen_hash,a.tag)
