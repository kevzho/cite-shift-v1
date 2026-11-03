"""Turn a frozen snapshot into per-edge outputs. Pure function of the snapshot (no network).

Writes outputs/<run_tag>/s2_edge_labels.csv and .jsonl. No comparison with any human label is made.
Usage: python 02_extract.py <run_tag> [edges_csv]
"""
import csv
import json
import sys

from s2_client import OUT, REPO

TEST = REPO / "analysis/results/external_systems/test_edges.csv"
COLS = ["edge_id", "corpus", "citing_id", "cited_id", "found_via", "citing_refs_elided",
        "citing_referenceCount", "n_ref_records_returned", "citing_n_refs_with_contexts",
        "citing_n_refs_with_intents", "citing_n_refs_influential", "edge_evidence", "isInfluential", "intents", "n_contexts",
        "context_intents", "contexts", "citing_venue", "citing_year", "retrieved_utc"]


def records(pages):
    for p in pages:
        if p["status"] == 200:
            yield from (p["body"].get("data") or [])


def evidence(hit, d, recs):
    """How much text S2 had behind this pair's flag (descriptive; not a comparison with any label).
    has_contexts      : S2 extracted >=1 citing sentence for this pair
    no_edge_contexts  : S2 has contexts for other references of the citing paper, none for this pair
    no_fulltext       : S2 has (almost) no contexts for the citing paper (<5 references with contexts)
    refs_elided       : citing reference list withheld; pair seen only from the cited side
    not_found         : pair absent from S2"""
    if not hit:
        return "not_found"
    if d.get("contexts"):
        return "has_contexts"
    if not recs:
        return "refs_elided"
    if sum(bool(x.get("contexts")) for x in recs) < 5:
        return "no_fulltext"
    return "no_edge_contexts"


def main(tag, edges_csv=TEST):
    snap = OUT / "snapshots" / tag
    meta = {str(m["corpusId"]): m for m in json.load(open(snap / "paper_batch.json"))["body"] if m}
    rows = []
    for e in csv.DictReader(open(edges_csv, encoding="utf-8")):
        ref = json.load(open(snap / "references" / f"{e['citing_id']}.json"))
        first = ref["pages"][0]["body"]
        disclaimer = ((first.get("citingPaperInfo") or {}).get("openAccessPdf") or {}).get("disclaimer") or ""
        recs = list(records(ref["pages"]))
        hit = [d for d in recs if str((d.get("citedPaper") or {}).get("corpusId")) == e["cited_id"]]
        via, when = ("references", ref["retrieved_utc"]) if hit else ("none", ref["retrieved_utc"])
        cpath = snap / "citations" / f"{e['cited_id']}.json"
        if not hit and cpath.exists():
            cit = json.load(open(cpath))
            hit = [d for d in records(cit["pages"])
                   if str((d.get("citingPaper") or {}).get("corpusId")) == e["citing_id"]]
            if hit:
                via, when = "citations", cit["retrieved_utc"]
        d = hit[0] if hit else {}
        m = meta.get(e["citing_id"], {})
        cwi = d.get("contextsWithIntent")
        rows.append({
            "edge_id": e["edge_id"], "corpus": e["corpus"], "citing_id": e["citing_id"],
            "cited_id": e["cited_id"], "found_via": via,
            "citing_refs_elided": "elided by the publisher: {'references'}" in disclaimer,
            "citing_referenceCount": m.get("referenceCount"), "n_ref_records_returned": len(recs),
            "citing_n_refs_with_contexts": sum(bool(x.get("contexts")) for x in recs) if recs else None,
            "citing_n_refs_with_intents": sum(bool(x.get("intents")) for x in recs) if recs else None,
            "citing_n_refs_influential": sum(bool(x.get("isInfluential")) for x in recs) if recs else None,
            "edge_evidence": evidence(hit, d, recs),
            "isInfluential": d.get("isInfluential") if hit else None,
            "intents": "|".join(sorted(d.get("intents") or [])) if hit else None,
            "n_contexts": len(d.get("contexts") or []) if hit else None,
            "context_intents": json.dumps([sorted(c.get("intents") or []) for c in cwi]) if cwi is not None else None,
            "contexts": json.dumps(d.get("contexts"), ensure_ascii=False) if hit else None,
            "citing_venue": m.get("venue"), "citing_year": m.get("year"), "retrieved_utc": when,
        })
    od = OUT / "outputs" / tag
    if (od / "s2_edge_labels.csv").exists():
        raise SystemExit(f"refusing to overwrite outputs/{tag}")
    od.mkdir(parents=True, exist_ok=True)
    with open(od / "s2_edge_labels.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, COLS)
        w.writeheader()
        w.writerows(rows)
    with open(od / "s2_edge_labels.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")
    for corpus in ("hci", "nature_oa"):
        rs = [r for r in rows if r["corpus"] == corpus]
        print(corpus, "n", len(rs), "found", sum(r["found_via"] != "none" for r in rs),
              "isInfluential_nonnull", sum(r["isInfluential"] is not None for r in rs),
              "intents_nonempty", sum(bool(r["intents"]) for r in rs),
              "contexts_nonempty", sum(bool(r["n_contexts"]) for r in rs),
              "refs_elided", sum(r["citing_refs_elided"] for r in rs),
              "evidence", {k: sum(r["edge_evidence"] == k for r in rs) for k in sorted({r["edge_evidence"] for r in rs})})


if __name__ == "__main__":
    main(sys.argv[1], *sys.argv[2:3])
