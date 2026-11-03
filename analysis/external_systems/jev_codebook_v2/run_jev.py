#!/usr/bin/env python3
"""Jev (TypeSafe AI, JEV-as-a-Judge) as a measured system on codebook v2.

Follows the JEV-as-a-Judge checklist: ONE Choice question per request, the threshold
fixed before scoring (none is applied here: the raw choice and the full probability
distribution are recorded), invalid outputs counted as errors.

Per edge, six independent requests, one per v2 field:
  Pass 1: cited_contribution_type, citation_function
  Pass 2: credits_contribution, uptake_mode, uptake_degree, claim_dependent
No request sees another request's answer, so Pass 2 never sees function.

Question text and option definitions are taken from the frozen prompts in
../local_llm_codebook_v2/prompts.py (synced to the frozen codebook, sha256 2acafc2e...),
so Jev is given the same definitions as the hosted LLM. `state` is the same evidence block.

Frozen system: the model is requested BY VERSION (default jev-1.13.0, never jev-latest);
the first response to each request is cached in responses_cache.jsonl and never
re-requested. Jev is not deterministic (identical requests vary by ~0.05 in probability,
3 Oct probe), so --fresh reruns (stability) bypass the cache and are appended to it
without replacing the frozen record.

Egress: titles + citances only, to api.typesafe.ai only. TypeSafe does not train on API
inputs; under the default tier it retains raw inputs and derives telemetry (Kevin, 3 Oct).

Usage (from this directory):
    python3 run_jev.py edges --input <csv> --tag pilot --limit 100
    python3 run_jev.py edges --input <csv> --tag s1 --fresh
    python3 run_jev.py compare s1 s2
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE.parent / "local_llm_codebook_v2"))
import prompts as P  # noqa: E402

OUT = ROOT / "analysis/results/external_systems/jev_codebook_v2"
CACHE = OUT / "responses_cache.jsonl"
MODEL = "jev-1.13.0"
ALLOWED_HOSTS = {"api.typesafe.ai"}
RETRYABLE = {408, 429, 500, 502, 503, 504, 529}

EVIDENCE_RULES = P.PREAMBLE.split("Evidence rules:", 1)[1].strip()

# field -> (pass text, block start marker, block end marker)
BLOCKS = {
    "cited_contribution_type": (P.PASS1, "cited_contribution_type -", "citation_function -"),
    "citation_function": (P.PASS1, "citation_function -", "Return JSON"),
    "credits_contribution": (P.PASS2, "credits_contribution -", "uptake_mode -"),
    "uptake_mode": (P.PASS2, "uptake_mode -", "uptake_degree -"),
    "uptake_degree": (P.PASS2, "uptake_degree -", "claim_dependent -"),
    "claim_dependent": (P.PASS2, "claim_dependent -", "Independence rule."),
}
CLAIM_UNCLEAR = "cannot tell whether a claim of the citing paper depends on the cited paper"
INDEPENDENCE = P.PASS2.split("Independence rule.", 1)[1].split("Return JSON", 1)[0].strip()


def block(field: str) -> str:
    text, start, end = BLOCKS[field]
    return text.split(start, 1)[1].split(end, 1)[0].strip()


def question(field: str) -> dict:
    body = block(field)
    criteria = {}
    for label in P.ENUMS[field]:
        m = re.search(rf"^- {re.escape(label)}: (.+)$", body, re.M)
        if m:
            criteria[label] = m.group(1).strip()
    if field == "claim_dependent":
        criteria.setdefault("unclear", CLAIM_UNCLEAR)
    missing = set(P.ENUMS[field]) - set(criteria)
    assert not missing, f"{field}: no definition for {sorted(missing)}"
    instructions = (f"Code the field `{field}` for this citation edge, using only the evidence in the "
                    f"state.\n\nEvidence rules:\n{EVIDENCE_RULES}\n\n{field} - {body}")
    if BLOCKS[field][0] is P.PASS2:
        instructions += f"\n\nIndependence rule. {INDEPENDENCE}"
    for bad in P.FORBIDDEN_IN_PROMPT:
        assert bad.lower() not in instructions.lower(), f"forbidden string: {bad}"
    return {"type": "choice", "instructions": instructions, "criteria": criteria}


FIELDS = list(BLOCKS)
QUESTIONS = {f: question(f) for f in FIELDS}


def load_env() -> dict:
    env = {}
    p = ROOT / ".env"
    for line in p.read_text().splitlines() if p.exists() else []:
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            env[k.strip()] = v.strip().strip('"').strip("'")
    url, key = env.get("JEV_API_URL", ""), env.get("JEV_API_KEY", "")
    host = urllib.parse.urlparse(url).hostname
    if host not in ALLOWED_HOSTS:
        sys.exit(f"REFUSED: JEV_API_URL host {host!r} is not in {sorted(ALLOWED_HOSTS)}")
    if not key or key.startswith("<"):
        sys.exit("Set JEV_API_KEY in citation_corpus/.env")
    return {"url": url, "key": key}


class Cache:
    def __init__(self, path: Path):
        self.path, self.lock, self.data = path, threading.Lock(), {}
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                rec = json.loads(line)
                if not rec.get("fresh_run"):
                    self.data.setdefault(rec["request_sha256"], rec)

    def get(self, key):
        return self.data.get(key)

    def put(self, rec):
        with self.lock:
            if not rec.get("fresh_run"):
                self.data.setdefault(rec["request_sha256"], rec)
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def call(cfg, body, max_retries=8):
    data = json.dumps(body).encode()
    for attempt in range(max_retries):
        req = urllib.request.Request(cfg["url"], data=data, headers={
            "Authorization": "Bearer " + cfg["key"], "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                return {"status": r.status, "json": json.loads(r.read())}
        except urllib.error.HTTPError as e:
            try:
                err = json.loads(e.read())
            except json.JSONDecodeError:
                err = {}
            if e.code in RETRYABLE and attempt < max_retries - 1:
                time.sleep(min(60, 2 ** attempt))
                continue
            return {"status": e.code, "json": err, "error": True}
        except (urllib.error.URLError, TimeoutError) as e:
            if attempt < max_retries - 1:
                time.sleep(min(60, 2 ** attempt))
                continue
            return {"status": None, "json": {"error": str(e)}, "error": True}
    return {"status": None, "json": {}, "error": True}


def ask(cfg, cache, a, field, state):
    body = {"model": a.model, "state": state, "questions": {field: QUESTIONS[field]}}
    key = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
    rec = None if a.fresh else cache.get(key)
    from_cache = rec is not None
    if rec is None:
        resp = call(cfg, body)
        j = resp.get("json") or {}
        ans = ((j.get("answers") or {}).get(field) or {}) if isinstance(j, dict) else {}
        rec = {"request_sha256": key, "field": field, "retrieved_utc": datetime.now(timezone.utc).isoformat(),
               "fresh_run": a.tag if a.fresh else None, "http_status": resp.get("status"),
               "error": bool(resp.get("error")), "error_detail": j if resp.get("error") else None,
               "response_model": j.get("model") if isinstance(j, dict) else None,
               "choice": ans.get("choice"), "confidence": ans.get("confidence"),
               "probabilities": ans.get("probabilities"), "usage": j.get("usage") if isinstance(j, dict) else None}
        if not rec["error"]:
            cache.put(rec)
    valid = (not rec["error"]) and rec["choice"] in P.ENUMS[field]
    return rec, from_cache, valid


def code_edge(cfg, cache, a, e):
    cits = [c for c in e["citances"].split("|||") if c.strip()]
    state = P.evidence(e.get("citing_title", ""), e.get("cited_title", ""), cits)
    row = {"edge_id": e["edge_id"], "corpus": e.get("corpus", ""), "n_citances": len(cits)}
    recs = {}
    for f in FIELDS:
        rec, from_cache, valid = ask(cfg, cache, a, f, state)
        recs[f] = (rec, from_cache, valid)
        row[f] = rec["choice"] if valid else ""
        row[f"{f}_confidence"] = rec["confidence"] if valid else ""
        row[f"{f}_probs"] = json.dumps(rec["probabilities"], sort_keys=True) if valid else ""
    row["all_valid"] = int(all(v for _, _, v in recs.values()))
    return row, recs


def run_edges(a):
    cfg = load_env()
    OUT.mkdir(parents=True, exist_ok=True)
    out_csv = OUT / f"edges_{a.tag}.csv"
    if out_csv.exists():
        sys.exit(f"refusing to overwrite {out_csv.relative_to(ROOT)}")
    with open(a.input, newline="", encoding="utf-8") as h:
        edges = list(csv.DictReader(h))[: a.limit or None]
    cache = Cache(CACHE)
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        results = list(ex.map(lambda e: code_edge(cfg, cache, a, e), edges))
    rows = [r for r, _ in results]
    with out_csv.open("w", newline="", encoding="utf-8") as h:
        w = csv.DictWriter(h, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    calls = [rec for _, recs in results for rec in recs.values()]
    usage = {k: sum(((r["usage"] or {}).get(k) or 0) for r, fc, _ in calls if not fc)
             for k in ("input_tokens", "output_tokens")}
    meta = {"run_tag": a.tag, "fresh": a.fresh, "input": str(a.input), "requested_model": a.model,
            "response_models": sorted({str(r["response_model"]) for r, _, _ in calls}),
            "n_edges": len(rows), "requests": len(calls), "from_cache": sum(fc for _, fc, _ in calls),
            "valid": sum(v for _, _, v in calls),
            "errors": sum(r["error"] for r, _, _ in calls),
            "error_codes": sorted({str(r["http_status"]) for r, _, _ in calls if r["error"]}),
            "usage_new_requests": usage,
            "question_sha256": {f: hashlib.sha256(json.dumps(QUESTIONS[f], sort_keys=True).encode()).hexdigest()
                                for f in FIELDS},
            "finished_utc": datetime.now(timezone.utc).isoformat()}
    (OUT / f"meta_{a.tag}.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(meta, indent=2))


def compare(a):
    t1, t2 = a.tags
    r1 = {r["edge_id"]: r for r in csv.DictReader(open(OUT / f"edges_{t1}.csv", encoding="utf-8"))}
    r2 = {r["edge_id"]: r for r in csv.DictReader(open(OUT / f"edges_{t2}.csv", encoding="utf-8"))}
    shared = sorted(set(r1) & set(r2))
    out = {"n_edges": len(shared)}
    for f in FIELDS:
        pairs = [(r1[e][f], r2[e][f]) for e in shared if r1[e][f] and r2[e][f]]
        diffs = []
        for e in shared:
            if r1[e][f"{f}_probs"] and r2[e][f"{f}_probs"]:
                p1, p2 = json.loads(r1[e][f"{f}_probs"]), json.loads(r2[e][f"{f}_probs"])
                diffs.append(max(abs(p1.get(k, 0) - p2.get(k, 0)) for k in set(p1) | set(p2)))
        out[f] = {"n": len(pairs), "label_agreement": round(sum(x == y for x, y in pairs) / len(pairs), 4) if pairs else None,
                  "mean_max_prob_shift": round(sum(diffs) / len(diffs), 4) if diffs else None}
    print(json.dumps(out, indent=2))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=["edges", "compare", "show-questions"])
    ap.add_argument("tags", nargs="*")
    ap.add_argument("--input")
    ap.add_argument("--tag", default="run1")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--model", default=MODEL)
    a = ap.parse_args()
    if a.cmd == "show-questions":
        for f in FIELDS:
            q = QUESTIONS[f]
            print(f"== {f}: {len(q['instructions'])} chars, options {list(q['criteria'])}")
        return
    if a.cmd == "compare":
        return compare(a)
    if a.model == "jev-latest":
        sys.exit("refusing jev-latest: request a pinned version")
    run_edges(a)


if __name__ == "__main__":
    main()
