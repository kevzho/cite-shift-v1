#!/usr/bin/env python3
"""Create a blinded audit sample from completed S2/LLM disagreement cases."""
from __future__ import annotations
import argparse,csv,random
from collections import defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];D=ROOT/'analysis/derived/hci_contribution_coding_human_reviewed.csv';O=ROOT/'analysis/results/experiments';BASELINES=ROOT/'analysis/results/baselines';SEED=20260711
def norm(x):return ' '.join((x or '').lower().split())
def read(p):
 with p.open(newline='',encoding='utf-8-sig') as f:return list(csv.DictReader(f))
def write(p,rows,fields):
 with p.open('w',newline='',encoding='utf-8') as f:w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--human',type=Path,default=D);ap.add_argument('--s2',type=Path,default=BASELINES/'s2_enrichment_edge_signals.csv');ap.add_argument('--llm',type=Path,default=O/'frozen_llm_baseline_responses.csv');ap.add_argument('--n',type=int,default=72);a=ap.parse_args();human=read(a.human);s2={(r['cited_id'],r['citing_id']):r for r in read(a.s2)} if a.s2.exists() else {};llm={(r['cited_id'],r['citing_id']):r for r in read(a.llm)} if a.llm.exists() else {}
 candidates=[]
 for h in human:
  key=(h['cited_id'],h['citing_id']);truth='realized' if norm(h['realizes_contribution']) in {'yes','partial'} else 'not_realized';ss=s2.get(key,{});ll=llm.get(key,{});s2pred=''
  if norm(ss.get('s2_intents')):s2pred='not_realized' if set(norm(ss['s2_intents']).split(';')) <= {'background'} else 'realized'
  llpred=norm(ll.get('contribution_use'))
  if (s2pred and s2pred!=truth) or (llpred and llpred not in {'unclear',truth}):candidates.append((h,s2pred,llpred))
 if not candidates:raise SystemExit('No completed S2/LLM disagreement cases yet; run enrichment and/or provide completed LLM responses.')
 rng=random.Random(SEED);strata=defaultdict(list)
 for item in candidates:strata[(item[0].get('research_contribution'),item[0].get('citance_function'),item[1] or 'no_s2',item[2] or 'no_llm')].append(item)
 sample=[rng.choice(v) for _,v in sorted(strata.items())];sample=rng.sample(sample,min(a.n,len(sample))) if len(sample)>a.n else sample
 public=[];keyrows=[]
 for i,(h,sp,lp) in enumerate(sample,1):
  aid=f'disagreement_audit_{i:03d}';public.append({'audit_id':aid,'cited_id':h['cited_id'],'citing_id':h['citing_id'],'cited_title':h['cited_title'],'citing_title':h['citing_title'],'citances':h['citances'],'reviewed_realization':'','reviewed_function':'','confidence':'','rationale':'','reviewer_id':''});keyrows.append({'audit_id':aid,'human_realizes_contribution':h['realizes_contribution'],'human_citance_function':h['citance_function'],'s2_binary_prediction':sp,'llm_binary_prediction':lp})
 write(O/'s2_llm_disagreement_audit_sample_blinded.csv',public,list(public[0]));write(O/'s2_llm_disagreement_audit_key.csv',keyrows,list(keyrows[0]));print(f'Wrote {len(sample)} blinded disagreement-audit rows from {len(candidates)} candidate disagreements.')
if __name__=='__main__':main()
