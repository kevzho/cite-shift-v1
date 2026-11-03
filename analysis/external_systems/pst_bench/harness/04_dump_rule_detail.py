"""Per-citing-paper detail of the official rule baseline: which bibs it flags as ref-sources, and the
raw +-100-char TEI contexts of the test edge's target bib (the exact text PST models see)."""
import sys, os, csv, json
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "paper-source-trace"))
from bs4 import BeautifulSoup
import rule, utils
rows = [r for r in csv.DictReader(open(os.path.join(ROOT, "outputs", "pst_edge_scores_rule_only.csv"))) if r["status"] == "SCORED"]
out = []
for r in rows:
    xml = open(os.path.join(ROOT, "inputs", "tei", r["citing_id"] + ".tei.xml"), encoding="utf-8").read()
    bs = BeautifulSoup(xml, "xml")
    t = {b.attrs["xml:id"]: (b.analytic.title.text if b.analytic and b.analytic.title else "?") for b in bs.find_all("biblStruct") if "xml:id" in b.attrs}
    hits = rule.extract_one_paper_via_rule(xml)
    ctx = utils.find_bib_context(xml)[r["target_bib"]]
    out.append(dict(edge_id=r["edge_id"], citing_id=r["citing_id"], target_bib=r["target_bib"],
                    target_title=t.get(r["target_bib"]), rule_flagged=[(h, t.get(h)) for h in hits],
                    target_contexts=ctx))
json.dump(out, open(os.path.join(ROOT, "outputs", "rule_detail.json"), "w"), indent=1, ensure_ascii=False)
for o in out:
    print(o["edge_id"], o["target_bib"], "| rule flagged:", o["rule_flagged"])
    for c in o["target_contexts"]: print("   ctx:", c[:200])
