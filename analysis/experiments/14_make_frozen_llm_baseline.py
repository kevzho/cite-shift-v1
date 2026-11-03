#!/usr/bin/env python3
"""Create a non-leaky, frozen request batch for a generic citation baseline."""
from __future__ import annotations
import argparse,csv,hashlib,json
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];D=ROOT/'analysis/derived/hci_contribution_coding_human_reviewed.csv';O=ROOT/'analysis/results/experiments';SEED=20260711
def read(p):
 with p.open(newline='',encoding='utf-8-sig') as f:return list(csv.DictReader(f))
def prompt(r):
 return f'''You are a generic scientific citation-intent classifier. Use only the evidence below. Do not infer or mention hidden human annotations. Return one JSON object with keys "generic_intent" (background, methodology, result, comparison, other, or unclear), "contribution_use" (realized, not_realized, or unclear), and "rationale" (one sentence).\n\nCited-paper title: {r.get("cited_title","")}\nCiting-paper title: {r.get("citing_title","")}\nSection: {r.get("section","")}\nCitation context: {r.get("citances","")}\n'''
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--input',type=Path,default=D);ap.add_argument('--model-id',default='UNSET_REQUIRES_AUTHORIZED_MODEL');ap.add_argument('--output',type=Path,default=O/'frozen_llm_baseline_requests.jsonl');ap.add_argument('--manifest',type=Path,default=O/'frozen_llm_baseline_manifest.csv');a=ap.parse_args();rows=read(a.input);manifest=[]
 with a.output.open('w',encoding='utf-8') as f:
  for i,r in enumerate(rows,1):
   p=prompt(r);request_id=f'hci_generic_baseline_{i:04d}';payload={'request_id':request_id,'cited_id':r.get('cited_id'),'citing_id':r.get('citing_id'),'model_id':a.model_id,'randomization_seed':SEED,'temperature':0,'prompt':p,'prompt_hash':hashlib.sha256(p.encode()).hexdigest(),'response_schema':{'generic_intent':['background','methodology','result','comparison','other','unclear'],'contribution_use':['realized','not_realized','unclear']}}
   f.write(json.dumps(payload,ensure_ascii=False)+'\n');manifest.append({'request_id':request_id,'cited_id':r.get('cited_id'),'citing_id':r.get('citing_id'),'model_id':a.model_id,'prepared_at':datetime.now(timezone.utc).isoformat(),'temperature':0,'randomization_seed':SEED,'prompt_hash':payload['prompt_hash'],'generation_status':'prepared_no_authorized_credentials'})
 with a.manifest.open('w',newline='',encoding='utf-8') as f:w=csv.DictWriter(f,fieldnames=list(manifest[0]));w.writeheader();w.writerows(manifest)
 print(f'Wrote {len(rows)} frozen, non-leaky baseline requests. Submit only through an authorized model interface.')
if __name__=='__main__':main()
