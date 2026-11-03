"""Fetch and freeze S2's precomputed per-pair labels for the 20 test edges.

For each test edge we request only public ids:
  1. POST /paper/batch  (citing + cited CorpusIds): metadata for coverage diagnosis
     (referenceCount, citationCount, influentialCitationCount, venue, year, externalIds).
  2. GET /paper/CorpusId:<citing>/references  (all pages): the citation records of the citing paper;
     the one whose citedPaper.corpusId == cited_id is the edge.
  3. Fallback only if (2) has no matching record: GET /paper/CorpusId:<cited>/citations and look for
     citingPaper.corpusId == citing_id (S2 sometimes elides a publisher's reference lists but still
     serves the citation from the cited side).
Raw responses are written verbatim to snapshots/<run_tag>/ with the UTC retrieval time.

Usage: python 01_fetch_snapshot.py <run_tag> [edges_csv]

edges_csv (default: the 20 test edges) needs edge_id, corpus, cited_id, citing_id. A run is
resume-safe: a citing paper whose references were already fetched with status 200 is skipped.
A tag whose run_meta.json exists is refused (snapshots are frozen).
"""
import csv
import json
import datetime as dt
import sys

from s2_client import OUT, REF_FIELDS, REPO, S2, dump

TEST = REPO / "analysis/results/external_systems/test_edges.csv"
META_FIELDS = ("corpusId,externalIds,title,venue,publicationVenue,year,referenceCount,citationCount,"
               "influentialCitationCount,isOpenAccess,publicationTypes")


def main(tag, edges_csv=TEST):
    snap = OUT / "snapshots" / tag
    if (snap / "run_meta.json").exists():
        raise SystemExit(f"refusing to overwrite frozen snapshot {tag}")
    edges = [{k: r[k] for k in ("edge_id", "corpus", "cited_id", "citing_id")}
             for r in csv.DictReader(open(edges_csv, encoding="utf-8"))]  # ids only; no text is read into requests
    api = S2()
    t0 = dt.datetime.now(dt.timezone.utc)
    ids = sorted({e["citing_id"] for e in edges} | {e["cited_id"] for e in edges}, key=int)
    if not (snap / "paper_batch.json").exists():
        statuses, meta = [], []
        for i in range(0, len(ids), 500):  # /paper/batch takes at most 500 ids
            st, body = api.batch(ids[i:i + 500], META_FIELDS)
            statuses.append(st)
            meta += body if isinstance(body, list) else []
        dump(snap / "paper_batch.json", {"retrieved_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                                         "status": max(statuses), "statuses": statuses, "fields": META_FIELDS,
                                         "ids": ids, "body": meta})
    for e in edges:
        rp = snap / "references" / f"{e['citing_id']}.json"
        if rp.exists() and json.load(open(rp)).get("status") == 200:
            prev = json.load(open(rp))
            prev_hit = any(str((d.get("citedPaper") or {}).get("corpusId")) == e["cited_id"]
                           for p in prev["pages"] if p["status"] == 200 for d in (p["body"].get("data") or []))
            if prev_hit or (snap / "citations" / f"{e['cited_id']}.json").exists():
                continue
        st, pages = api.references(e["citing_id"])
        rec = {"retrieved_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "fields": REF_FIELDS,
               "citing_id": e["citing_id"], "status": st, "pages": pages}
        dump(snap / "references" / f"{e['citing_id']}.json", rec)
        hit = any(str((d.get("citedPaper") or {}).get("corpusId")) == e["cited_id"]
                  for p in pages if p["status"] == 200 for d in (p["body"].get("data") or []))
        if not hit:
            st2, pages2 = api.citations(e["cited_id"])
            dump(snap / "citations" / f"{e['cited_id']}.json",
                 {"retrieved_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "fields": REF_FIELDS,
                  "cited_id": e["cited_id"], "status": st2, "pages": pages2})
        print(e["edge_id"], e["corpus"], "refs_status", st, "matched_in_refs", hit, flush=True)
    t1 = dt.datetime.now(dt.timezone.utc)
    dump(snap / "run_meta.json", {"run_tag": tag, "start_utc": t0.isoformat(), "end_utc": t1.isoformat(),
                                  "wall_s": (t1 - t0).total_seconds(), "n_requests": api.n_requests,
                                  "n_429": api.n_429, "base": "https://api.semanticscholar.org/graph/v1"})
    print("requests", api.n_requests, "429s", api.n_429, "wall_s", (t1 - t0).total_seconds())


if __name__ == "__main__":
    main(sys.argv[1], *sys.argv[2:3])
