#!/usr/bin/env python3
"""Draw the Nature arm frame: ~1,200 PMC open-access research articles from the 6 fully-OA Nature
Portfolio journals in PMC, 2016-2025, year-matched to the CHI corpus, plus a 15% reserve.

Allocation: the 27 Sep sibling agent's journal x year allocation
(../nature_oa_jats_pilot_2026-09-27/frame/frame_proposed_year_matched_allocation.csv, columns alloc_*,
summing to 3,758 = the CHI per-year paper counts) scaled to 1,200 by largest remainder over cells.

Pool per cell: Europe PMC search `ISSN:"<issn>" AND PUB_YEAR:<y> AND OPEN_ACCESS:y AND IN_PMC:y`
(resultType=lite, cursor paging, 1000/page); kept only if pubType contains "research-article" and the
record has a PMCID and a DOI. Excluded: every pilot frame DOI / pilot PMCID, and every DOI of a citing
paper in the canonical nature_oa record (analysis/derived/nature_oa_contribution_coding_human_reviewed.csv
citing_id -> DOI via citation_corpus_fulltext/papers.parquet), so the arm is disjoint from labelled data.

Draw: per cell, random.Random(f"{SEED}:{journal}:{year}").sample(pool, quota + reserve), where
reserve = ceil(0.15 * quota). The first `quota` are primary (rank 1..quota), the rest reserve.
Writes frame/frame.csv, frame/allocation_1200.csv, frame/pool_counts.csv, frame/frame_build_log.json.
Sends only ISSNs and years to Europe PMC.
"""
import csv, json, math, random, sys, urllib.parse
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).parent))
from common import EPMC, REPO, ROOT, get, utc

SEED = 20261004
TOTAL = 1200
RESERVE = 0.15
PILOT = REPO / "external_data/nature_oa_jats_pilot_2026-09-27"
JOURNALS = {"natcomms": "2041-1723", "commsbio": "2399-3642", "commschem": "2399-3669",
            "commsmed": "2730-664X", "commseng": "2731-3395", "commspsych": "2731-9121"}

alloc_src = pd.read_csv(PILOT / "frame/frame_proposed_year_matched_allocation.csv")
alloc_src = alloc_src[alloc_src.year.between(2016, 2025)]
cells = []
for _, r in alloc_src.iterrows():
    for j in JOURNALS:
        v = r[f"alloc_{j}"]
        if pd.notna(v) and v > 0:
            cells.append({"journal": j, "year": int(r.year), "alloc_3758": int(v)})
src_total = sum(c["alloc_3758"] for c in cells)
for c in cells:
    c["exact"] = c["alloc_3758"] * TOTAL / src_total
    c["quota"] = int(math.floor(c["exact"]))
short = TOTAL - sum(c["quota"] for c in cells)
for c in sorted(cells, key=lambda c: (-(c["exact"] - c["quota"]), c["year"], c["journal"]))[:short]:
    c["quota"] += 1
cells = [c for c in cells if c["quota"] > 0]
assert sum(c["quota"] for c in cells) == TOTAL

# exclusions
canon = pd.read_csv(REPO / "analysis/derived/nature_oa_contribution_coding_human_reviewed.csv",
                    usecols=["citing_id"])
papers = pd.read_parquet(REPO / "citation_corpus_fulltext/papers.parquet", columns=["corpusid", "doi"])
canon_dois = set(papers[papers.corpusid.isin(set(canon.citing_id))].doi.dropna().str.lower())
pilot_frame = pd.read_csv(PILOT / "frame/pilot_frame.csv")
pilot_dois = set(pilot_frame.doi.str.lower())
pilot_pmcids = {p.stem for p in (PILOT / "jats").glob("PMC*.xml")} | \
               {p.stem for p in (PILOT / "jats_s3_compare").glob("PMC*.xml")}
excl_dois = canon_dois | pilot_dois

rows, pool_counts = [], []
for c in cells:
    q = f'ISSN:"{JOURNALS[c["journal"]]}" AND PUB_YEAR:{c["year"]} AND OPEN_ACCESS:y AND IN_PMC:y'
    cursor, pool, hits, seen = "*", [], 0, set()
    while True:
        url = f"{EPMC}/search?" + urllib.parse.urlencode(
            {"query": q, "format": "json", "resultType": "lite", "pageSize": 1000, "cursorMark": cursor})
        s, b, _ = get(url)
        if s != 200:
            raise SystemExit(f"Europe PMC {s} for {c}")
        j = json.loads(b)
        hits = j.get("hitCount", 0)
        res = j.get("resultList", {}).get("result", [])
        for r in res:
            pmcid, doi = r.get("pmcid"), (r.get("doi") or "").lower()
            if not pmcid or not doi or pmcid in seen:
                continue
            seen.add(pmcid)
            if "research-article" not in (r.get("pubType") or ""):
                continue
            if doi in excl_dois or pmcid in pilot_pmcids:
                continue
            pool.append((pmcid, doi))
        nxt = j.get("nextCursorMark")
        if not res or not nxt or nxt == cursor:
            break
        cursor = nxt
    pool.sort()
    k = min(len(pool), c["quota"] + math.ceil(RESERVE * c["quota"]))
    pick = random.Random(f"{SEED}:{c['journal']}:{c['year']}").sample(pool, k)
    for i, (pmcid, doi) in enumerate(pick, 1):
        rows.append({"journal": c["journal"], "year": c["year"], "rank": i,
                     "role": "primary" if i <= c["quota"] else "reserve", "pmcid": pmcid, "doi": doi})
    pool_counts.append({"journal": c["journal"], "year": c["year"], "epmc_hits": hits,
                        "eligible_pool": len(pool), "quota": c["quota"], "drawn": k})
    print(f"{c['journal']:10s} {c['year']} hits {hits:6d} pool {len(pool):6d} quota {c['quota']:4d} drawn {k}",
          flush=True)

pd.DataFrame(rows).to_csv(ROOT / "frame/frame.csv", index=False)
pd.DataFrame(cells).to_csv(ROOT / "frame/allocation_1200.csv", index=False)
pd.DataFrame(pool_counts).to_csv(ROOT / "frame/pool_counts.csv", index=False)
json.dump({"built_utc": utc(), "seed": SEED, "total": TOTAL, "reserve_share": RESERVE,
           "source_allocation_total": src_total, "excluded_canonical_citing_dois": len(canon_dois),
           "excluded_pilot_dois": len(pilot_dois), "excluded_pilot_pmcids": len(pilot_pmcids),
           "frame_rows": len(rows)}, open(ROOT / "frame/frame_build_log.json", "w"), indent=1)
print(f"frame rows {len(rows)} (primary {sum(r['role']=='primary' for r in rows)})")
