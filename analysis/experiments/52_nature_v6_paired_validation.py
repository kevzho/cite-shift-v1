#!/usr/bin/env python3
"""Frozen paired fresh-paper comparison, run only after both fixed evaluations."""
from __future__ import annotations
import argparse,csv,hashlib,json,random
from collections import Counter,defaultdict
from pathlib import Path

REPO=Path(__file__).resolve().parents[2]
OUT=REPO/'external_data/nature_repair_v2_2026-10-25/fresh_round4'

def main(baseline,revised):
    target=OUT/'paired_validation.json'
    if target.exists(): raise SystemExit('Refusing to overwrite paired fresh validation')
    load=lambda p:list(csv.DictReader(Path(p).open()))
    old,new=load(baseline),load(revised)
    assert {(r['pmcid'],r['jats_ref']) for r in old}=={(r['pmcid'],r['jats_ref']) for r in new}
    def stats(rows):
        per=defaultdict(Counter); counts=Counter((r['pmcid'],r['predicted_ref']) for r in rows if r['predicted_ref'])
        for r in rows:
            c=int(r['n_contexts']);rec=int(r['n_recovered']);complete=int(r['complete'])
            unique=not r['predicted_ref'] or counts[(r['pmcid'],r['predicted_ref'])]==1
            per[r['pmcid']].update(contexts=c,recovered=rec,edges=1,complete=complete,
                unique_recovered=rec if unique else 0,unique_complete=complete if unique else 0)
        return per
    a,b=stats(old),stats(new);papers=sorted(a);assert papers==sorted(b)
    def rates(per,ps):
        c=sum((per[p] for p in ps),Counter())
        return dict(recovery=c['recovered']/c['contexts'],edge_match=c['complete']/c['edges'],
                    strict_unique_recovery=c['unique_recovered']/c['contexts'],
                    strict_unique_edge_match=c['unique_complete']/c['edges'])
    ar,br=rates(a,papers),rates(b,papers);rng=random.Random(20261028);samples=defaultdict(list)
    for _ in range(2000):
        ps=rng.choices(papers,k=len(papers)); av,bv=rates(a,ps),rates(b,ps)
        for key in av:samples[key].append(bv[key]-av[key])
    def ci(v):
        v=sorted(v);return [v[int(.025*len(v))],v[int(.975*len(v))-1]]
    result={'status':'EXPLORATORY paired fresh validation; original RQ4(b) not reinstated',
        'bootstrap':{'unit':'citing paper','resamples':2000,'seed':20261028},'papers':len(papers),
        'baseline':ar,'revised':br,
        'revised_minus_baseline':{k:{'difference':br[k]-ar[k],'ci95':ci(samples[k])} for k in ar},
        'reference_edges':len(old),'independent_manual_precision':'not established; source review required',
        'input_sha256':{str(p):hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in [baseline,revised,Path(__file__)]}}
    target.write_text(json.dumps(result,indent=2)+'\n')
    review=load(OUT/'precision_review_revised.csv')
    tei={p.name.removesuffix('.tei.xml'):p for p in (REPO/'external_data/nature_arm_2026-10-04/pdfs_grobid').rglob('*.tei.xml')}
    for r in review:
        pid=r['pmcid']
        r.update(jats_source=str(OUT/'jats'/f'{pid}.xml'),jats_reference_id=r['jats_ref'],
                 pdf_source=str(REPO/'external_data/nature_arm_2026-10-04/pdfs'/f'{pid}.pdf'),
                 tei_source=str(tei.get(pid,'')))
    with (OUT/'precision_review_with_sources.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(review[0]));w.writeheader();w.writerows(review)
    print(json.dumps(result,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--baseline',required=True);p.add_argument('--revised',required=True)
    args=p.parse_args();main(args.baseline,args.revised)
