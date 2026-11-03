#!/usr/bin/env python3
"""Registered Nature-arm parity gate (pre-registration §7 RQ4(b)).

"same GROBID pipeline as CHI; 200-paper JATS parity subsample; passes if citance recovery >= 90%
and edge match >= 85%."

Reference standard: JATS-derived citances for the 200 parity papers (parity/jats_citances_200.jsonl,
produced by the pilot's schema-matched extract_citances_jats.py). Tested: GROBID-derived citances for
the same papers (citances/nature_arm_citances.jsonl, the CHI extractor on GROBID 0.9.0-crf TEIs).

Definitions reused from the 27 Sep pilot (code/06_validate_extractor_jats.py, PILOT_NOTES.md §7):
  locate    a JATS edge's GROBID counterpart in the same citing paper: cited_doi exact (case-folded),
            else normalised cited_title (lowercase alphanumerics) SequenceMatcher ratio >= 0.9;
  tokens    lowercase [a-z]{3,};  recall(c, x) = |tok(c) & tok(x)| / |tok(c)|;
  matched   a JATS citance c is recovered if some GROBID citance x on the located edge has recall >= 0.8;
  edge fully matched  every JATS citance on the edge is recovered (unlocated edges count as not matched).
Citance recovery = recovered JATS citances / all JATS citances.
Edge match       = fully matched JATS edges / all JATS edges.
Direction: the pilot used the PDF-derived canonical record as reference and JATS as tested; here the
registered gate makes JATS the reference and GROBID the tested pipeline. The matching rule is identical.
Reverse direction (GROBID citances on located edges with no JATS citance at recall >= 0.8 the other
way) is reported descriptively. Bootstrap CI over papers, 2,000 resamples, seed 20261003.
"""
from __future__ import annotations

import csv
import json
import random
import re
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path

ROOT = Path(__import__("os").environ.get("NATURE_ARM_DIR", Path(__file__).resolve().parents[2] / "external_data/nature_arm_2026-10-04"))
JATS = ROOT / "parity/jats_citances_200.jsonl"
GROB = ROOT / "citances/nature_arm_citances.jsonl"
MANI = ROOT / "frame/manifest.csv"
OUT = ROOT / "parity"
THR_CIT, THR_EDGE = 0.90, 0.85

tok = lambda s: set(re.findall(r"[a-z]{3,}", (s or "").lower()))
norm_t = lambda s: re.sub(r"[^a-z0-9]", "", (s or "").lower())


def load(p):
    d = defaultdict(list)
    for line in open(p, encoding="utf-8"):
        r = json.loads(line)
        d[r["citing_paper_id"]].append(r)
    return d


def main():
    journal = {r["pmcid"]: r["journal"] for r in csv.DictReader(open(MANI, encoding="utf-8"))}
    jats, grob = load(JATS), load(GROB)
    papers = sorted(jats)
    per_paper, edge_rows = {}, []
    how_counts = defaultdict(int)
    for pid in papers:
        g_edges = grob.get(pid, [])
        g_by_doi = {e["cited_doi"].lower(): e for e in g_edges if e["cited_doi"]}
        g_titles = [(norm_t(e["cited_title"]), e) for e in g_edges if e["cited_title"]]
        n_c = rec_c = n_e = full_e = g_on_loc = g_unmatched = 0
        for e in jats[pid]:
            hit, how = None, "none"
            doi = (e["cited_doi"] or "").lower()
            if doi and doi in g_by_doi:
                hit, how = g_by_doi[doi], "doi"
            if hit is None and e["cited_title"]:
                nt = norm_t(e["cited_title"])
                best = max(((SequenceMatcher(None, t, nt).ratio(), ge) for t, ge in g_titles if t),
                           key=lambda x: x[0], default=(0, None))
                if best[0] >= 0.9:
                    hit, how = best[1], "title"
            how_counts[how] += 1
            xs = hit["citances"] if hit else []
            rec = [any(len(tok(c) & tok(x)) / max(len(tok(c)), 1) >= 0.8 for x in xs) for c in e["citances"]]
            n_c += len(rec); rec_c += sum(rec); n_e += 1; full_e += bool(hit) and all(rec)
            if hit:
                g_on_loc += len(xs)
                g_unmatched += sum(1 for x in xs if max((len(tok(c) & tok(x)) / max(len(tok(x)), 1)
                                                         for c in e["citances"]), default=0) < 0.8)
            edge_rows.append({"pmcid": pid, "journal": journal.get(pid, ""), "jats_ref": e["cited_ref_id"],
                              "located": how, "grobid_ref": hit["cited_ref_id"] if hit else "",
                              "n_jats_citances": len(rec), "n_recovered": sum(rec),
                              "fully_matched": int(bool(hit) and all(rec)), "n_grobid_citances": len(xs)})
        per_paper[pid] = dict(n_c=n_c, rec_c=rec_c, n_e=n_e, full_e=full_e, g_on_loc=g_on_loc,
                              g_unmatched=g_unmatched, journal=journal.get(pid, ""))

    def rates(pids):
        s = lambda k: sum(per_paper[p][k] for p in pids)
        return s("rec_c") / s("n_c"), s("full_e") / s("n_e")

    cit, edge = rates(papers)
    rng = random.Random(20261003)
    bc, be = [], []
    for _ in range(2000):
        smp = rng.choices(papers, k=len(papers))
        a, b = rates(smp); bc.append(a); be.append(b)
    bc.sort(); be.sort()
    ci = lambda v: [round(v[int(0.025 * len(v))], 4), round(v[int(0.975 * len(v)) - 1], 4)]
    by_j = {}
    for j in sorted({v["journal"] for v in per_paper.values()}):
        ps = [p for p in papers if per_paper[p]["journal"] == j]
        a, b = rates(ps)
        by_j[j] = {"papers": len(ps), "citance_recovery": round(a, 4), "edge_match": round(b, 4)}
    tot = lambda k: sum(v[k] for v in per_paper.values())
    res = {
        "registered_rule": "PASS if citance recovery >= 0.90 and edge match >= 0.85",
        "papers": len(papers), "jats_edges": tot("n_e"), "jats_citances": tot("n_c"),
        "papers_missing_in_grobid": sum(1 for p in papers if p not in grob),
        "located": dict(how_counts),
        "citance_recovery": round(cit, 4), "citance_recovery_ci95": ci(bc),
        "edge_match": round(edge, 4), "edge_match_ci95": ci(be),
        "reverse_grobid_citances_on_located_edges": tot("g_on_loc"),
        "reverse_grobid_citances_without_jats_counterpart": tot("g_unmatched"),
        "by_journal": by_j,
        "verdict": "PASS" if cit >= THR_CIT and edge >= THR_EDGE else "FAIL",
    }
    OUT.mkdir(exist_ok=True)
    (OUT / "parity_gate_result.json").write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")
    with open(OUT / "parity_gate_edges.csv", "w", newline="", encoding="utf-8") as h:
        w = csv.DictWriter(h, fieldnames=list(edge_rows[0])); w.writeheader(); w.writerows(edge_rows)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
