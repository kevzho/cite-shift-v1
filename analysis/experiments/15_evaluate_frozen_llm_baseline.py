#!/usr/bin/env python3
"""Evaluate completed frozen LLM outputs; refuses incomplete or invalid runs."""
from __future__ import annotations
import argparse,csv,json
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];D=ROOT/'analysis/derived/hci_contribution_coding_human_reviewed.csv';O=ROOT/'analysis/results/experiments'
def norm(x):return ' '.join((x or '').lower().split())
def read(p):
 with p.open(newline='',encoding='utf-8-sig') as f:return list(csv.DictReader(f))
def write(p,rows,fields):
 with p.open('w',newline='',encoding='utf-8') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--responses',type=Path,required=True,help='CSV with cited_id,citing_id,generic_intent,contribution_use,raw_response');ap.add_argument('--human',type=Path,default=D);a=ap.parse_args()
 if not a.responses.exists(): raise SystemExit(f'Response file missing: {a.responses}. Submit frozen_llm_baseline_requests.jsonl through an authorized model interface first.')
 human={(r['cited_id'],r['citing_id']):r for r in read(a.human)};responses=read(a.responses)
 required={'cited_id','citing_id','generic_intent','contribution_use','raw_response'}
 if not responses or required-set(responses[0]):raise SystemExit(f'Response CSV requires {sorted(required)}')
 joined=[(r,human.get((r['cited_id'],r['citing_id']))) for r in responses if human.get((r['cited_id'],r['citing_id']))]
 if len(joined)!=len(human):raise SystemExit(f'Evaluation refused: {len(joined)}/{len(human)} human edges have completed joined responses; supply a complete run.')
 allowed_intent={'background','methodology','result','comparison','other','unclear'};allowed_use={'realized','not_realized','unclear'}
 if any(norm(r['generic_intent']) not in allowed_intent or norm(r['contribution_use']) not in allowed_use or not r['raw_response'].strip() for r,_ in joined):raise SystemExit('Evaluation refused: blank/invalid baseline output.')
 pairs=[(norm(r['contribution_use']),'realized' if norm(h['realizes_contribution']) in {'yes','partial'} else 'not_realized') for r,h in joined];correct=sum(a==b for a,b in pairs if a!='unclear');conf=Counter(pairs);summary=[{'source_type':'frozen_llm_generic_baseline','n_completed':len(joined),'n_non_unclear':sum(a!='unclear' for a,_ in pairs),'accuracy_non_unclear':round(correct/sum(a!='unclear' for a,_ in pairs),4) if sum(a!='unclear' for a,_ in pairs) else ''}]
 write(O/'frozen_llm_baseline_summary.csv',summary,list(summary[0]));write(O/'frozen_llm_baseline_confusion.csv',[{'llm_contribution_use':a,'human_realization_binary':b,'n':n} for (a,b),n in sorted(conf.items())],['llm_contribution_use','human_realization_binary','n']);print(f'Evaluated {len(joined)} complete frozen LLM responses.')
if __name__=='__main__':main()
