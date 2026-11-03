"""Throughput and coverage probe on a seeded sample of the 226,793-line CHI/CHI EA citance file.

Only CorpusId:<citing_paper_id> is sent. Responses are matched to the sampled lines locally (DOI, then
normalised title) and then DISCARDED: only timing and coverage counts are written.
Usage: python 04_throughput_chi.py [n_lines=200] [seed=42]
"""
import datetime as dt
import difflib
import json
import random
import re
import sys
import time

from s2_client import OUT, REPO, S2

SRC = REPO / "external_data/chi_chiea_grobid_2026-09-15/citances/chi_chiea_citances.jsonl"


def norm(t):
    return re.sub(r"[^a-z0-9]+", " ", (t or "").lower()).strip()


def main(n=200, seed=42):
    lines = SRC.read_text(encoding="utf-8").splitlines()
    total = len(lines)
    idx = sorted(random.Random(seed).sample(range(total), n))
    sample = [json.loads(lines[i]) for i in idx]
    all_citing = {json.loads(l)["citing_paper_id"] for l in lines}
    del lines
    by_citing = {}
    for x in sample:
        by_citing.setdefault(x["citing_paper_id"], []).append(x)
    api = S2()
    counts = dict(lines=n, citing_papers=len(by_citing), refs_status_200=0, refs_elided_or_empty=0,
                  matched=0, matched_by_doi=0, matched_by_title=0, matched_with_contexts=0,
                  matched_with_intents=0, pages=0)
    lat = []
    t0 = time.monotonic()
    for cid, xs in by_citing.items():
        t = time.monotonic()
        st, pages = api.references(cid)
        lat.append(time.monotonic() - t)
        counts["pages"] += len(pages)
        recs = [d for p in pages if p["status"] == 200 for d in (p["body"].get("data") or [])]
        counts["refs_status_200"] += st == 200
        counts["refs_elided_or_empty"] += not recs
        by_doi = {((d.get("citedPaper") or {}).get("externalIds") or {}).get("DOI", "").lower(): d for d in recs}
        by_doi.pop("", None)
        titles = [(norm((d.get("citedPaper") or {}).get("title")), d) for d in recs]
        for x in xs:
            d, how = by_doi.get((x.get("cited_doi") or "").lower()), "doi"
            if d is None and x.get("cited_title"):
                nt = norm(x["cited_title"])
                best = max(titles, key=lambda td: difflib.SequenceMatcher(None, nt, td[0]).ratio(), default=None)
                if best and difflib.SequenceMatcher(None, nt, best[0]).ratio() >= 0.9:
                    d, how = best[1], "title"
            if d is not None:
                counts["matched"] += 1
                counts[f"matched_by_{how}"] += 1
                counts["matched_with_contexts"] += bool(d.get("contexts"))
                counts["matched_with_intents"] += bool(d.get("intents"))
        del pages, recs, by_doi, titles  # outputs are not kept
    wall = time.monotonic() - t0
    per_call = wall / max(1, api.n_requests - api.n_429)
    est_chi = len(all_citing) * wall / len(by_citing)
    res = {"date_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "seed": seed, "file_lines": total,
           "file_unique_citing_papers": len(all_citing), **counts, "wall_s": round(wall, 1),
           "http_requests": api.n_requests, "http_429": api.n_429,
           "mean_s_per_citing_paper": round(wall / len(by_citing), 3),
           "median_latency_s_per_citing_paper": round(sorted(lat)[len(lat) // 2], 3),
           "est_hours_all_chi_citing_papers_at_measured_rate": round(est_chi / 3600, 2),
           "est_hours_all_chi_at_ideal_1rps": round(len(all_citing) * 1.0 / 3600, 2),
           "est_hours_2937_canonical_edges_at_measured_rate": round(2937 * wall / len(by_citing) / 3600, 2),
           "est_hours_2937_canonical_edges_at_ideal_1rps": round(2937 / 3600, 2)}
    (OUT / "outputs" / "throughput_chi_sample.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main(*(int(a) for a in sys.argv[1:]))
