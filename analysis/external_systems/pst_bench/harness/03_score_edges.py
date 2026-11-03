"""Step 3: score the 20 shared test edges with PST-Bench baselines, using the official code paths.

For each edge (cited_id -> citing_id) we need the CITING paper's GROBID TEI (full text with inline
<ref type="bibr" target="#bN"> markers), exactly the input PST-Bench uses. Edges without a TEI are
reported as NO_INPUT (never approximated from the citance alone).

Systems:
  rule    : official rule.extract_one_paper_via_rule(xml)  (released rule baseline, no training)
  scibert : official SciBERT recipe (bert.py) re-trained on the released PST train split
            (no fine-tuned weights are released); inference replicates
            bert.gen_kddcup_valid_submission_bert(): one score per bib = logits[:,1] over the joined
            +-100-char raw-XML contexts of that bib (utils.find_bib_context).
Target-bib matching: official labeling rule (fuzz.ratio(ref_title, cited_title) >= 80); if that fails,
fall back to the project's own GROBID citance map (chi_chiea_citances.jsonl cited_ref_id) and log it.
"""
import sys, os, csv, json, glob, argparse
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
REPO = os.path.join(ROOT, "paper-source-trace")
sys.path.insert(0, REPO)
import numpy as np, torch
from bs4 import BeautifulSoup
from fuzzywuzzy import fuzz
import utils, rule, bert
from transformers import AutoTokenizer, BertForSequenceClassification

ap = argparse.ArgumentParser()
ap.add_argument("--edges", default=os.path.join(ROOT, "..", "test_edges.csv"))
ap.add_argument("--tei_dir", default=os.path.join(ROOT, "inputs", "tei"))
ap.add_argument("--citances", default="/Volumes/kz_extended/SALT-UIUC/citation_iConf/citation_corpus/external_data/chi_chiea_grobid_2026-09-15/citances/chi_chiea_citances.jsonl")
ap.add_argument("--model", default=os.path.join(REPO, "out", "kddcup", "scibert", "pytorch_model.bin"))
ap.add_argument("--out", default=os.path.join(ROOT, "outputs", "pst_edge_scores.csv"))
args = ap.parse_args()

edges = list(csv.DictReader(open(args.edges)))
# project citance map (fallback only)
need = {e["citing_id"] for e in edges}
cit_map = {}
if os.path.exists(args.citances):
    for line in open(args.citances):
        d = json.loads(line)
        if d["citing_paper_id"] in need:
            cit_map.setdefault(d["citing_paper_id"], []).append((d.get("cited_title") or "", d["cited_ref_id"]))

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = tok = None
if os.path.exists(args.model):
    tok = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
    model = BertForSequenceClassification.from_pretrained("allenai/scibert_scivocab_uncased", num_labels=2)
    model.load_state_dict(torch.load(args.model, map_location="cpu"))
    model.to(device).eval()
else:
    print("WARNING: no trained SciBERT weights at", args.model, "-> scibert columns left empty")

cache = {}
def paper_scores(pid, xml):
    if pid in cache:
        return cache[pid]
    bs = BeautifulSoup(xml, "xml")
    bid_to_title = {}
    for ref in bs.find_all("biblStruct"):
        if "xml:id" not in ref.attrs or ref.analytic is None or ref.analytic.title is None:
            continue
        bid_to_title[ref.attrs["xml:id"]] = ref.analytic.title.text.lower()
    all_bids = [r.attrs["xml:id"] for r in bs.find_all("biblStruct") if "xml:id" in r.attrs]
    rule_hits = set(rule.extract_one_paper_via_rule(xml))
    bib_to_contexts = utils.find_bib_context(xml)
    bib_sorted = sorted(bib_to_contexts.keys())
    scores = {}
    if model is not None and bib_sorted:
        ctx = [" ".join(bib_to_contexts[b]) for b in bib_sorted]
        feats = bert.convert_examples_to_inputs(ctx, [0] * len(ctx), bert.MAX_SEQ_LENGTH, tok)
        dl = bert.get_data_loader(feats, bert.MAX_SEQ_LENGTH, 16, shuffle=False)
        logit1, prob1 = [], []
        with torch.no_grad():
            for batch in dl:
                ids, mask, seg, _ = (t.to(device) for t in batch)
                lg = model(ids, attention_mask=mask, token_type_ids=seg).logits
                logit1 += lg[:, 1].cpu().tolist()
                prob1 += torch.softmax(lg, -1)[:, 1].cpu().tolist()
        scores = {b: (l, p) for b, l, p in zip(bib_sorted, logit1, prob1)}
    res = dict(bid_to_title=bid_to_title, all_bids=all_bids, rule_hits=rule_hits,
               contexts=bib_to_contexts, scores=scores)
    cache[pid] = res
    return res

rows = []
for e in edges:
    out = dict(edge_id=e["edge_id"], corpus=e["corpus"], cited_id=e["cited_id"], citing_id=e["citing_id"],
               canonical_function=e["canonical_function"], canonical_realization=e["canonical_realization"],
               bulk_or_core=e["bulk_or_core"])
    tei = os.path.join(args.tei_dir, e["citing_id"] + ".tei.xml")
    if not os.path.exists(tei):
        out.update(status="NO_INPUT", note="no GROBID TEI of citing paper available locally")
        rows.append(out); continue
    xml = open(tei, encoding="utf-8").read()
    ps = paper_scores(e["citing_id"], xml)
    target = [b for b, t in ps["bid_to_title"].items() if fuzz.ratio(t, e["cited_title"].lower()) >= 80]
    how = "official_fuzz>=80"
    if not target:
        target = [rid for t, rid in cit_map.get(e["citing_id"], []) if fuzz.ratio(t.lower(), e["cited_title"].lower()) >= 80]
        how = "project_citance_map"
    if not target:
        out.update(status="NO_BIB_MATCH", note="cited paper not found in citing TEI bibliography"); rows.append(out); continue
    b = target[0]
    out.update(status="SCORED", target_bib=b, bib_match=how, n_bibs_total=len(ps["all_bids"]),
               n_bibs_with_context=len(ps["contexts"]), n_mentions_of_target=len(ps["contexts"].get(b, [])),
               rule_is_source=int(b in ps["rule_hits"]), rule_n_sources_in_paper=len(ps["rule_hits"]))
    if ps["scores"]:
        if b in ps["scores"]:
            l, p = ps["scores"][b]
            ranked = sorted(ps["scores"], key=lambda k: -ps["scores"][k][0])
            rank = ranked.index(b) + 1
            out.update(scibert_logit1=round(l, 4), scibert_prob_source=round(p, 4), scibert_rank=rank,
                       scibert_n_ranked=len(ranked), scibert_pct_rank=round(rank / len(ranked), 3),
                       scibert_is_source_p05=int(p > 0.5), scibert_is_top1=int(rank == 1),
                       scibert_top3_bibs=";".join(f"{k}:{ps['bid_to_title'].get(k,'?')[:40]}" for k in ranked[:3]))
        else:
            out.update(note="target bib has no inline mention -> official pipeline gives it score 0 (unranked)")
    rows.append(out)

os.makedirs(os.path.dirname(args.out), exist_ok=True)
keys = []
for r in rows:
    for k in r:
        if k not in keys: keys.append(k)
with open(args.out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
for r in rows:
    print({k: r.get(k) for k in ["edge_id", "corpus", "canonical_function", "canonical_realization", "status",
                                 "target_bib", "rule_is_source", "scibert_prob_source", "scibert_rank", "scibert_n_ranked"]})
print("wrote", args.out)
