#!/usr/bin/env python3
"""Exploratory (spec private/preregistration/S2_PUBLISHER_COVERAGE_SPEC_2026-10-21.md, fixed before any fetch):
does S2 full-text coverage differ by publisher, and how much of a Traditional vs New-OA gap in the share of
S2-influential citations (Cunningham & Smyth, CACM 2025) is coverage?

Stages (resume-safe; cached under OUT): sample -> citations -> analyse.
Only public identifiers (DOIs, CorpusIds) are sent. Per cited paper only counts are stored (no citance text).
Usage: python3 37_publisher_coverage.py [--smoke]
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
import urllib.parse
from pathlib import Path

import numpy as np
import requests

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "analysis/external_systems/s2_influential/harness"))
from s2_client import S2  # noqa: E402  (key from .env, 1 req/s, retries)

OUT = ROOT / "analysis/results/exploratory/publisher_coverage_2026-10-21"
PUBS = {  # name: (group, crossref filter)
    "ACM": ("Traditional", "member:320"),
    "IEEE": ("Traditional", "member:263"),            # IEEE Access removed below
    "Elsevier": ("Traditional", "member:78"),
    "Springer": ("Traditional", "member:297"),
    "IEEE Access": ("New OA", "issn:2169-3536"),
    "MDPI": ("New OA", "member:1968"),
    "Hindawi": ("New OA", "member:98"),
    "Frontiers": ("New OA", "member:1965"),
}
BASE_FILTER = "type:journal-article,from-pub-date:2016-01-01,until-pub-date:2020-12-31"
TARGET, MAX_CALLS, CAP = 150, 60, 1000
N_BOOT, SEED = 10_000, 20261003


def crossref_sample(flt):
    url = f"https://api.crossref.org/works?sample=100&filter={BASE_FILTER},{flt}&select=DOI,container-title"
    for t in range(6):
        try:
            r = requests.get(url, timeout=60, headers={"User-Agent": "sts-research-script/1.0"})
            if r.status_code == 200:
                return r.json()["message"]["items"]
        except requests.RequestException:
            pass
        time.sleep(2 ** t)
    return []


def stage_sample(s2, target, max_calls):
    path = OUT / "sample.json"
    done = json.loads(path.read_text()) if path.exists() else {}
    for pub, (_, flt) in PUBS.items():
        st = done.get(pub, {"seen": [], "included": [], "calls": 0})
        seen = set(st["seen"])
        while len(st["included"]) < target and st["calls"] < max_calls:
            items = crossref_sample(flt)
            st["calls"] += 1
            dois = []
            for it in items:
                ct = " ".join(it.get("container-title") or [])
                if pub == "IEEE" and "IEEE Access" in ct:
                    continue
                d = it["DOI"].lower()
                if d not in seen:
                    seen.add(d); dois.append(d)
            for i in range(0, len(dois), 500):
                r = s2.request("POST", "/paper/batch", params={"fields": "externalIds,year,citationCount,venue"},
                               body={"ids": [f"DOI:{d}" for d in dois[i:i + 500]]})
                body = r.json() if r.status_code == 200 else []
                for d, p in zip(dois[i:i + 500], body):
                    if p and (p.get("externalIds") or {}).get("DBLP") and (p.get("citationCount") or 0) >= 1:
                        if len(st["included"]) < target:
                            st["included"].append({"doi": d, "corpusId": p["externalIds"].get("CorpusId"),
                                                   "year": p.get("year"), "citationCount": p.get("citationCount"),
                                                   "venue": p.get("venue")})
            st["seen"] = sorted(seen)
            done[pub] = st
            path.write_text(json.dumps(done))
            print(f"[sample] {pub}: calls {st['calls']} seen {len(seen)} included {len(st['included'])}", flush=True)
    return done


def stage_citations(s2, sample):
    path = OUT / "citations.jsonl"
    have = {json.loads(l)["corpusId"] for l in open(path)} if path.exists() else set()
    with open(path, "a") as f:
        for pub, st in sample.items():
            for p in st["included"]:
                if p["corpusId"] in have:
                    continue
                n = ctx = infl = infl_noctx = 0
                offset, status, trunc = 0, 200, False
                while True:
                    r = s2.request("GET", f"/paper/CorpusId:{p['corpusId']}/citations",
                                   params={"fields": "isInfluential,contexts", "limit": CAP, "offset": offset})
                    if r.status_code != 200:
                        status = r.status_code; break
                    body = r.json()
                    for c in body.get("data", []):
                        n += 1
                        has = bool(c.get("contexts"))
                        ctx += has
                        if c.get("isInfluential"):
                            infl += 1; infl_noctx += not has
                    nxt = body.get("next")
                    if nxt is None:
                        break
                    if nxt >= CAP:
                        trunc = True; break
                    offset = nxt
                rec = {"publisher": pub, "group": PUBS[pub][0], "corpusId": p["corpusId"], "status": status,
                       "n_citations": n, "n_with_context": ctx, "n_influential": infl,
                       "n_influential_without_context": infl_noctx, "truncated_at_cap": trunc,
                       "s2_citationCount": p["citationCount"], "year": p["year"]}
                f.write(json.dumps(rec) + "\n"); f.flush()
                have.add(p["corpusId"])
        print("[citations] done", flush=True)


def kendall(a, b):
    ks = list(a)
    c = d = 0
    for i in range(len(ks)):
        for j in range(i + 1, len(ks)):
            x = np.sign(a[ks[i]] - a[ks[j]]) * np.sign(b[ks[i]] - b[ks[j]])
            c += x > 0; d += x < 0
    return (c - d) / (len(ks) * (len(ks) - 1) / 2)


def stage_analyse():
    recs = [json.loads(l) for l in open(OUT / "citations.jsonl")]
    recs = [r for r in recs if r["status"] == 200 and r["n_citations"] > 0]
    pubs = list(PUBS)
    N = np.array([r["n_citations"] for r in recs], float)
    X = np.array([r["n_with_context"] for r in recs], float)
    I = np.array([r["n_influential"] for r in recs], float)
    P = np.array([pubs.index(r["publisher"]) for r in recs])
    G = np.array([r["group"] == "Traditional" for r in recs])
    rng = np.random.default_rng(SEED)
    # stratified cluster bootstrap: resample cited papers within each publisher
    idx_by = {k: np.where(P == k)[0] for k in range(len(pubs))}
    W = np.zeros((N_BOOT, len(recs)))
    for k, ix in idx_by.items():
        if len(ix):
            draws = rng.integers(0, len(ix), size=(N_BOOT, len(ix)))
            for b in range(N_BOOT):
                np.add.at(W[b], ix[draws[b]], 1)

    def stats(w, mask):
        n, x, i = (w * mask * N).sum(-1), (w * mask * X).sum(-1), (w * mask * I).sum(-1)
        return i / n, x / n, i / x                      # F_all, C, F_ctx

    one = np.ones(len(recs))
    ci = lambda v: [round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)]
    per = {}
    for k, pub in enumerate(pubs):
        m = (P == k).astype(float)
        if not m.sum():
            continue
        fa, c, fc = stats(one, m); bfa, bc, bfc = stats(W, m)
        paper_avg = float(np.mean(I[P == k] / N[P == k]))
        per[pub] = {"group": PUBS[pub][0], "papers": int(m.sum()), "citations": int(N[P == k].sum()),
                    "F_all": round(float(fa), 4), "F_all_ci95": ci(bfa), "coverage": round(float(c), 4),
                    "coverage_ci95": ci(bc), "F_ctx": round(float(fc), 4), "F_ctx_ci95": ci(bfc),
                    "F_all_paper_avg": round(paper_avg, 4),
                    "truncated_papers": int(sum(r["truncated_at_cap"] for r in recs if r["publisher"] == pub))}
    T, Nn = G.astype(float), (~G).astype(float)
    fT, cT, xT = stats(one, T); fN, cN, xN = stats(one, Nn)
    bfT, bcT, bxT = stats(W, T); bfN, bcN, bxN = stats(W, Nn)
    tot, cov = math.log(fT / fN), math.log(cT / cN)
    btot, bcov = np.log(bfT / bfN), np.log(bcT / bcN)
    share, bshare = cov / tot if tot else float("nan"), bcov / btot
    group = {"Traditional": {"F_all": round(float(fT), 4), "coverage": round(float(cT), 4), "F_ctx": round(float(xT), 4)},
             "New OA": {"F_all": round(float(fN), 4), "coverage": round(float(cN), 4), "F_ctx": round(float(xN), 4)},
             "diff_F_all": round(float(fT - fN), 4), "diff_F_all_ci95": ci(bfT - bfN),
             "diff_coverage": round(float(cT - cN), 4), "diff_coverage_ci95": ci(bcT - bcN),
             "diff_F_ctx": round(float(xT - xN), 4), "diff_F_ctx_ci95": ci(bxT - bxN),
             "log_ratio_F_all": round(tot, 4), "log_ratio_coverage": round(cov, 4),
             "log_ratio_F_ctx": round(math.log(xT / xN), 4),
             "coverage_share": round(share, 4), "coverage_share_ci95": ci(bshare)}
    tau = kendall({p: v["F_all"] for p, v in per.items()}, {p: v["F_ctx"] for p, v in per.items()})
    reproduced = fT > fN
    lo_all, hi_all = group["diff_F_all_ci95"]; lo_c, hi_c = group["diff_F_ctx_ci95"]
    excl0 = lambda lo, hi: lo > 0 or hi < 0
    confound = (share >= 0.5) or (excl0(lo_all, hi_all) and not excl0(lo_c, hi_c)) or \
        (np.sign(fT - fN) != np.sign(xT - xN))
    none = share < 0.25 and excl0(lo_c, hi_c) and np.sign(fT - fN) == np.sign(xT - xN)
    reading = "coverage confound present" if confound else "no material confound" if none else "partial"
    out = {"spec": "private/preregistration/S2_PUBLISHER_COVERAGE_SPEC_2026-10-21.md", "n_boot": N_BOOT, "seed": SEED,
           "influential_without_context": int(sum(r["n_influential_without_context"] for r in recs)),
           "per_publisher": per, "groups": group, "kendall_tau_F_all_vs_F_ctx": round(tau, 4),
           "reproduces_trad_gt_newoa": bool(reproduced),
           "reading": reading if reproduced else f"{reading} (descriptive only: direction not reproduced)"}
    p = OUT / "analysis.json"
    if p.exists():
        raise SystemExit(f"refusing to overwrite {p}")
    p.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="2 included papers per publisher, separate dir, no analysis")
    a = ap.parse_args()
    global OUT
    if a.smoke:
        OUT = OUT.parent / (OUT.name + "_SMOKE")
    OUT.mkdir(parents=True, exist_ok=True)
    s2 = S2()
    sample = stage_sample(s2, 2 if a.smoke else TARGET, 2 if a.smoke else MAX_CALLS)
    stage_citations(s2, sample)
    print(f"S2 requests {s2.n_requests}, 429s {s2.n_429}", flush=True)
    if not a.smoke:
        stage_analyse()


if __name__ == "__main__":
    main()
