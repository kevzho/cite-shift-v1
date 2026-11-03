#!/usr/bin/env python3
"""Post hoc, exploratory (decided 1 Nov 2026 after the function-retest result was known): is the low function-retest
kappa noise, or a shift in the coder's reading between 26 Sep (v1 first coding) and 31 Oct (retest)? Compares each coding
with canonical and with the coder's own v2 A250 uptake_mode (7 Oct), mapped to v1 function by an ad hoc map defined here
(method->uses_method, artifact->reuses_artifact, finding/extension->builds_on_finding, contrast->compares_against,
none/framing/design_knowledge->background). unclear excluded. Prints only; enters no registered verdict.

    python3 analysis/experiments/65_function_retest_drift_posthoc.py
"""
import csv, sys, importlib.util
from collections import Counter
sys.path.insert(0,'analysis/experiments')
spec=importlib.util.spec_from_file_location('S','analysis/experiments/22_score_systems.py'); S=importlib.util.module_from_spec(spec); spec.loader.exec_module(S)
def rd(p): return list(csv.DictReader(open(p,encoding='utf-8-sig')))
def kappa(pairs):
    n=len(pairs); po=sum(a==b for a,b in pairs)/n
    ca,cb=Counter(a for a,_ in pairs),Counter(b for _,b in pairs)
    pe=sum(ca[k]*cb[k] for k in set(ca)|set(cb))/n/n; return (po-pe)/(1-pe)
cw=S.crosswalk(); alias,_=S.components()
RT='annotations/generated/retest_2026-10-24/'
key={r['item']:r['edge_id'] for r in rd(RT+'function_retest_key.csv')}
ret={key[r['item']]:S.norm('citation_function',r['citation_function'],cw) for r in rd(RT+'function_retest_coded.csv')}
v1=S.ref_v1(cw)['citation_function']; can=S.ref_canonical(cw)['citation_function']
first={e:v1[e] for e in ret}
def cmp(name,a,b):
    es=[e for e in ret if a.get(e) and b.get(e) and 'unclear' not in (a[e],b[e])]
    p=[(a[e],b[e]) for e in es]; print(f'{name}: n={len(p)} raw={sum(x==y for x,y in p)/len(p):.2f} kappa={kappa(p):.2f}')
canm={e:can.get(alias.get(e,e)) for e in ret}
print('canonical found', sum(v is not None for v in canm.values()))
cmp('first vs retest',first,ret); cmp('first vs canonical',first,canm); cmp('retest vs canonical',ret,canm)
# A250 v2 uptake_mode
ak={r['item']:r['edge_id'] for r in rd('annotations/generated/v2_coding_2026-10-02/A250_pass2_key.csv')}
A={ak[r['item']]:r['uptake_mode'].strip() for r in rd('annotations/generated/v2_coding_2026-10-02/A250_pass2.csv')}
ov=[e for e in ret if e in A]; print('overlap with A250', len(ov))
M={'method':'uses_method','artifact':'reuses_artifact','finding':'builds_on_finding','extension':'builds_on_finding','contrast':'compares_against','none':'background','framing':'background','design_knowledge':'background'}
Am={e:M.get(A[e]) for e in ov}
cmp('first vs A250-mapped',{e:first[e] for e in ov},Am); cmp('retest vs A250-mapped',{e:ret[e] for e in ov},Am)
ch=[e for e in ov if first[e]!=ret[e] and 'unclear' not in (first[e],ret[e])]
print('changed items in overlap',len(ch),'; retest matches A250 mapping:',sum(ret[e]==Am[e] for e in ch),' first matches:',sum(first[e]==Am[e] for e in ch))
