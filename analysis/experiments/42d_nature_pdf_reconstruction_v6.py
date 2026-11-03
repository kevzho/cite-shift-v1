#!/usr/bin/env python3
"""Exploratory PDF/TEI citance reconstruction; no JATS input in extraction.

Retain the original TEI body contexts, supplement with PDF-native superscript
citations and PDF reference-list labels. Explicit source geometry/provenance is
retained. This is a new extraction method, not a change to the registered audit.
Requires PyMuPDF==1.26.7 (installed separately; no model/network calls here).
"""
from __future__ import annotations
import argparse, csv, hashlib, json, re, sys, unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from xml.etree import ElementTree as ET
from difflib import SequenceMatcher
try:
    import fitz
except ImportError:
    sys.path.insert(0, '/tmp/nature_parser_deps')
    import fitz

ROOT=Path(__file__).resolve().parents[2]
ARM=ROOT/'external_data/nature_arm_2026-10-04'
NS={'t':'http://www.tei-c.org/ns/1.0'}
XID='{http://www.w3.org/XML/1998/namespace}id'

def norm(s):
    return re.sub('[^a-z0-9]','',unicodedata.normalize('NFKD',s or '').lower())

def clean(s):
    s=unicodedata.normalize('NFKC',s)
    s=re.sub(r'([A-Za-z])-\s*\n\s*([a-z])',r'\1\2',s)
    return re.sub(r'\s+',' ',s).strip()

def txt(e):
    return clean(''.join(e.itertext())) if e is not None else ''

def nums(s,maxn=999):
    s=s.strip().strip('[]()').replace('–','-').replace('−','-')
    if not re.fullmatch(r'\d{1,3}(?:\s*[-,]\s*\d{1,3})*\s*[,;]?',s): return []
    out=[]
    for g in s.rstrip(',;').split(','):
        parts=g.split('-')
        if len(parts)==1: out.append(int(parts[0]))
        elif len(parts)==2 and 0<=int(parts[1])-int(parts[0])<=40:
            out.extend(range(int(parts[0]),int(parts[1])+1))
    return sorted(set(n for n in out if 1<=n<=maxn))

def bibmeta(root):
    out={}
    for b in root.findall('.//t:listBibl/t:biblStruct',NS):
        bid=b.get(XID)
        if not bid: continue
        titles=[txt(t) for t in b.findall('.//t:title',NS) if t.get('level') in ('a','m') and txt(t)]
        variants=list(dict.fromkeys(titles+[txt(t) for t in b.findall('.//t:title',NS) if txt(t)]))
        raw=txt(b.find("t:note[@type='raw_reference']",NS))
        date=b.find('.//t:date',NS)
        year=((date.get('when') or txt(date))[:4]) if date is not None else ''
        doi=next((txt(x) for x in b.findall('.//t:idno',NS) if x.get('type','').lower()=='doi'),'')
        out[bid]={'cited_title':titles[0] if titles else (variants[0] if variants else ''), 'cited_title_variants':variants,
                  'cited_raw_reference':raw,'cited_year':year,'cited_doi':doi,
                  'cited_first_author':txt(b.find('.//t:surname',NS))}
    return out

def references_from_pdf(doc):
    pages=[p.get_text() for p in doc]
    joined='\n'.join(pages)
    head=re.search(r'(?mi)^\s*References\s*$',joined)
    if not head: return {},None
    tail=joined[head.end():]
    matches=list(re.finditer(r'(?m)^\s*(\d{1,3})\.\s*(?:\n)?',tail))
    refs={}
    expected=1
    for i,m in enumerate(matches):
        n=int(m[1])
        if n!=expected: continue
        end=matches[i+1].start() if i+1<len(matches) else min(len(tail),m.end()+1000)
        raw=tail[m.end():end]
        raw=re.split(r'(?mi)^\s*(?:Acknowledgements|Acknowledgments|Author contributions|Competing interests|Additional information|Publisher.s note)\s*$',raw)[0]
        refs[n]=clean(raw); expected+=1
    return refs,head.start()

def reference_map(root,bib,pdfrefs):
    # Link source numbers through PDF bibliography text, never through bN+1.
    mapping={}; evidence={}; used=set()
    for n,raw in pdfrefs.items():
        nr=norm(raw)
        scores=[]
        for bid,b in bib.items():
            br=norm(b['cited_raw_reference'])
            title=norm(b['cited_title'])
            exact=bool(br and min(len(br),len(nr))>25 and (br in nr or nr in br))
            titleok=len(title)>=25 and title in nr
            author=norm(b['cited_first_author']) in nr if b['cited_first_author'] else False
            year=b['cited_year'] in raw if b['cited_year'] else False
            ratio=SequenceMatcher(None,br,nr,autojunk=False).ratio() if br and (exact or titleok or (author and year)) else 0
            if exact or (titleok and author and year) or (ratio>=.86 and author and year):
                scores.append((int(exact)*2+int(titleok)+ratio,bid,exact,titleok,ratio))
        scores.sort(reverse=True)
        if scores and (len(scores)==1 or scores[0][0]-scores[1][0]>.15) and scores[0][1] not in used:
            _,bid,exact,titleok,ratio=scores[0]
            mapping[n]=bid;used.add(bid)
            evidence[n]={'rule':'pdf_reference_text','similarity':round(ratio,4)}
    # Supplement source numbers using unambiguous existing TEI body links.
    links=defaultdict(set)
    body=root.find('.//t:body',NS)
    if body is not None:
        for r in body.findall('.//t:ref[@type="bibr"]',NS):
            bids=(r.get('target') or '').replace('#','').split()
            nn=nums(txt(r))
            if len(nn)==len(bids)==1 and bids[0] in bib: links[nn[0]].add(bids[0])
    for n,bs in links.items():
        if n not in mapping and len(bs)==1:
            bid=next(iter(bs))
            if bid not in used:
                mapping[n]=bid;used.add(bid);evidence[n]={'rule':'unique_tei_body_marker'}
    return mapping,evidence

def tei_contexts(root,bib):
    rows=defaultdict(list)
    body=root.find('.//t:body',NS)
    if body is None:return rows
    for p in body.findall('.//t:p',NS):
        ss=p.findall('.//t:s',NS) or [p]
        for i,s in enumerate(ss):
            sent=txt(s)
            start=i;end=i+1
            # GROBID sentence breaks inside taxonomic initials and author
            # abbreviations are not actual sentence boundaries.
            while start>0 and i-start<4 and (re.match(r'^[a-z]',txt(ss[start])) or re.search(r'(?:\b[A-Z]\.|\bet al\.|\b(?:Fig|Figs|Dr|Prof)\.)[)\]]?\s*$',txt(ss[start-1]))):
                start-=1
            # A bracketed aside and its following statement remain one source
            # context, consistent with the visible paragraph structure.
            if start>0 and txt(ss[start-1]).startswith('[') and txt(ss[start-1]).endswith(']'):start-=1
            if txt(s).startswith('[') and txt(s).endswith(']') and end<len(ss):end+=1
            if start!=i or end!=i+1:sent=clean(' '.join(txt(x) for x in ss[start:end]))
            if len(sent)<20:
                # Reconstruct a fragment within its paragraph, not from unrelated text.
                sent=clean(' '.join(txt(x) for x in ss[max(0,i-1):min(len(ss),i+2)]))
            if len(sent)<20:continue
            for ref in s.findall('.//t:ref[@type="bibr"]',NS):
                for bid in (ref.get('target') or '').replace('#','').split():
                    if bid in bib:
                        rows[bid].append((sent,None,{'source':'tei_body','fragment_reconstructed':len(txt(s))<20 or start!=i or end!=i+1}))
    return rows

def pdf_contexts(doc,mapping,maxn):
    rows=defaultdict(list); narrative=False; in_refs=False; previous_tail='';previous_source=None
    sizes=Counter()
    reference_started=False
    for page in doc:
        for b in page.get_text('dict')['blocks']:
            blocktext=' '.join(s['text'] for l in b.get('lines',[]) for s in l['spans'])
            if re.match(r'^\s*References(?:\s|$)',blocktext):reference_started=True
            if reference_started:continue
            spans=[s for l in b.get('lines',[]) for s in l['spans']]
            has_source_marker=any(s['flags']&1 and nums(s['text']) for s in spans)
            if len(blocktext.split())<35 or not has_source_marker:continue
            for l in b.get('lines',[]):
                for s in l['spans']:
                    if not(s['flags']&1):sizes[round(s['size'],1)]+=len(re.findall('[A-Za-z]',s['text']))
    # Nature formats may use a smaller Methods prose font. Retain substantive
    # citation-bearing prose font modes rather than selecting exactly one.
    top_size_count=sizes.most_common(1)[0][1] if sizes else 0
    body_sizes=[size for size,count in sizes.items() if count>=top_size_count*.20] if top_size_count else []
    for pi,page in enumerate(doc):
        blocks=page.get_text('dict')['blocks']
        for bi,b in enumerate(blocks):
            if 'lines' not in b:continue
            block_plain=' '.join(s['text'] for l in b['lines'] for s in l['spans'])
            if re.match(r'^\s*References(?:\s|$)',block_plain):in_refs=True;continue
            if in_refs and re.fullmatch(r'\s*(?:Methods|Materials and methods|Methods and materials)\s*',block_plain,re.I):in_refs=False
            if in_refs:continue
            # Headers, footers, author affiliations and tiny labels are not context.
            if b['bbox'][1]<25 or b['bbox'][3]>page.rect.height-25:continue
            if re.search(r'(?i)(?:\be-?mail\b|@|\breceived\s*:|\baccepted\s*:|\bdepartment of\b|\bcorrespondence\b|\bcheck for updates\b)',block_plain):continue
            if re.match(r'(?i)^\s*(?:Acknowledg|Author contributions|Competing interests|Additional information|Publisher.s note|Code availability|Data availability|Data and code availability|Code and data availability|Ethics declarations|Reporting summary)',block_plain):in_refs=True;continue
            bsizes=Counter()
            for l in b['lines']:
                for s in l['spans']:
                    if not(s['flags']&1):bsizes[round(s['size'],1)]+=len(re.findall('[A-Za-z]',s['text']))
            block_size=bsizes.most_common(1)[0][0] if bsizes else 0
            if body_sizes and min(abs(block_size-size) for size in body_sizes)>.6:continue
            if not narrative and len(block_plain.split())>=35 and re.search(r'[a-z]{3}[.!?]\s',block_plain):narrative=True
            if not narrative:continue
            if re.match(r'(?i)^\s*(?:Fig(?:ure)?\.?\s*\d|Table\s*\d)',block_plain):continue
            rendered=[];marks=[]
            for li,l in enumerate(b['lines']):
                spans=l['spans']; normal=[s['size'] for s in spans if not(s['flags']&1) and re.search('[A-Za-z]',s['text'])]
                main=max(normal,default=0)
                for si,s in enumerate(spans):
                    value=s['text']; ns=nums(value,maxn)
                    superscript=bool(s['flags']&1) and (not main or s['size']<=main*.85)
                    if ns and superscript:
                        # Reject common superscript powers/units, not source citation groups.
                        before=''.join(rendered)[-40:]
                        after=''.join(x['text'] for x in spans[si+1:])
                        exponent_or_unit=bool(re.search(r'(?:[−–+\-]\s*$|\b(?:cm|mm|nm|km|μm|µm|m|sec|s|mol|Å|K)\s*[−–+\-]?\s*$|\d\s*$)',before))
                        isotope=bool(re.match(r'^\s*(?:H|P|C|N|O|F|S|Fe|Ca)\b',after))
                        # A dash after an already certified citation marker
                        # denotes a citation range, not a negative exponent.
                        range_endpoint=bool(re.search(r'@@REF\d+@@\s*[–−-]\s*$',before))
                        if (exponent_or_unit and not range_endpoint) or isotope:
                            rendered.append(value);continue
                        key=f'@@REF{len(marks)}@@'
                        marks.append({'key':key,'numbers':ns,'marker':value,'page':pi+1,'bbox':list(s['bbox'])})
                        rendered.append(key)
                    else:rendered.append(value)
                rendered.append('\n')
            text=clean(''.join(rendered))
            # A PDF range can be split into three spans: superscript start,
            # dash, superscript end. Expand only the geometry-certified range.
            remove=set()
            for m in re.finditer(r'@@REF(\d+)@@\s*[–−-]\s*@@REF(\d+)@@',text):
                leftmark,rightmark=marks[int(m[1])],marks[int(m[2])]
                if len(leftmark['numbers'])==len(rightmark['numbers'])==1:
                    lo,hi=leftmark['numbers'][0],rightmark['numbers'][0]
                    if 0<hi-lo<=40:
                        leftmark['numbers']=list(range(lo,hi+1));leftmark['marker']=f'{lo}–{hi}'
                        text=text.replace(m[0],leftmark['key']);remove.add(int(m[2]))
            marks=[m for i,m in enumerate(marks) if i not in remove]
            continuation=False
            if previous_tail and re.match(r'^[a-z]',text) and not re.search(r'[.!?]["\u201d\u2019)]?\s*$',previous_tail):
                text=previous_tail+' '+text;continuation=True
            if marks or len(re.findall('[A-Za-z]{3,}',text))>=15:
                tail=re.split(r'(?<=[.!?])\s+(?=[A-Z])',text)[-1]
                previous_tail=re.sub(r'@@REF\d+@@','',tail)[-1000:]
                source_before=previous_source;previous_source={'page':pi+1,'block':bi}
            # Avoid abstract/title/author fragments and figure symbol-only blocks.
            if len(re.findall(r'[A-Za-z]{3,}',text))<8:continue
            for mark in marks:
                pos=text.find(mark['key'])
                if pos<0:continue
                boundaries=[0]+[m.end() for m in re.finditer(r'[.!?]\s+(?=[A-Z])',text)
                    if not re.search(r'(?:\b[A-Z]\.|\bet al\.|\b(?:Fig|Figs|Dr|Prof|vs)\.)\s*$',text[:m.end()])]+[len(text)]
                idx=next((i for i in range(len(boundaries)-1) if boundaries[i]<=pos<boundaries[i+1]),len(boundaries)-2)
                left,right=boundaries[idx:idx+2]
                # Sentence fragments from layout splits receive adjacent source sentence.
                if len(re.findall('[A-Za-z]{3,}',text[left:right]))<8:
                    left=boundaries[max(0,idx-1)];right=boundaries[min(len(boundaries)-1,idx+2)]
                sent=text[left:right].strip()
                targets=[n for n in mark['numbers'] if n in mapping]
                for n in targets:
                    plain=sent;marked=sent
                    for m in marks:
                        display='['+','.join(str(x) for x in m['numbers'])+']'
                        plain=plain.replace(m['key'],display)
                        tagged=('<cite>'+display+'</cite>') if m['key']==mark['key'] else display
                        marked=marked.replace(m['key'],tagged)
                    rows[mapping[n]].append((plain,marked,{'source':'pdf_superscript_geometry','page':pi+1,'block':bi,
                        'bbox':mark['bbox'],'source_reference_number':n,'source_marker':mark['marker'],
                        'cross_block_continuation':continuation,
                        'previous_source':source_before if continuation else None,
                        'target_group_size':len(mark['numbers']),'target_span_unambiguous':len(mark['numbers'])==1}))
    return rows

def extract_paper(pid,tei,pdf):
    root=ET.parse(tei).getroot();bib=bibmeta(root)
    with fitz.open(pdf) as doc:
        refs,_=references_from_pdf(doc)
        mapping,ev=reference_map(root,bib,refs)
        rows=tei_contexts(root,bib)
        extra=pdf_contexts(doc,mapping,max(refs,default=len(bib)))
    for bid,contexts in extra.items():rows[bid].extend(contexts)
    output=[]
    for bid,contexts in rows.items():
        unique={}
        for sent,marked,provenance in contexts:
            if sent not in unique or (marked and not unique[sent][0]): unique[sent]=(marked,provenance)
        meta=bib[bid].copy()
        numbers=[n for n,b in mapping.items() if b==bid]
        # Raw PDF reference fills metadata field-boundary losses, never context text.
        meta['pdf_reference_numbers']=numbers
        meta['pdf_raw_references']=[refs[n] for n in numbers if n in refs]
        output.append({'citing_paper_id':pid,'venue':tei.parent.name,'cited_ref_id':bid,**meta,
                       'n_citances':len(unique),'citances':list(unique),
                       'marked_citances':[x[0] for x in unique.values()],
                       'sections':['' for _ in unique],
                       'context_provenance':[x[1] for x in unique.values()]})
    return output,{'paper':pid,'edges':len(output),'contexts':sum(r['n_citances'] for r in output),
                  'pdf_reference_labels':len(refs),'mapped_source_labels':len(mapping),
                  'pdf_contexts':sum(p['source'].startswith('pdf') for r in output for p in r['context_provenance'])}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--papers',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--limit',type=int);a=ap.parse_args()
    if a.out.exists():raise SystemExit(f'Refusing overwrite {a.out}')
    ids=list(dict.fromkeys(r['pmcid'] for r in csv.DictReader(a.papers.open())))
    if a.limit:ids=ids[:a.limit]
    files={p.stem.replace('.tei',''):p for p in (ARM/'pdfs_grobid').rglob('*.tei.xml')}
    a.out.parent.mkdir(parents=True,exist_ok=True);stats=[]
    with a.out.open('w') as f:
        for i,pid in enumerate(ids):
            rr,st=extract_paper(pid,files[pid],ARM/'pdfs'/f'{pid}.pdf');stats.append(st)
            for r in rr:f.write(json.dumps(r,ensure_ascii=False)+'\n')
            if (i+1)%10==0:print(f'{i+1}/{len(ids)} papers',flush=True)
    manifest={'parser_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'pymupdf_version':fitz.VersionBind,
              'papers':len(ids),'totals':dict(sum((Counter({k:v for k,v in s.items() if isinstance(v,int)}) for s in stats),Counter())),
              'paper_stats':stats,'status':'exploratory; PDF/TEI-only extraction; no JATS context input'}
    a.out.with_suffix('.manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({k:v for k,v in manifest.items() if k!='paper_stats'},indent=2))

if __name__=='__main__':main()
