#!/usr/bin/env python3
"""Project's own frozen-LLM measured system for codebook v2, on a HOSTED model (Azure AI Foundry).

Prompts, schemas, validation and flattening are IMPORTED from ../local_llm_codebook_v2, not copied, so this
system and the local one send byte-identical prompts. What differs:

  * an OpenAI-compatible chat-completions call to the configured endpoint;
  * a response cache that FREEZES the system: the first response for a request is kept forever and never
    re-requested, so the cached snapshot is the measured system (as with the Semantic Scholar snapshot);
  * fresh re-runs (--fresh) that measure run-to-run stability without touching the frozen snapshot.

Hosted models are not guaranteed to repeat exactly even at temperature 0; report `compare` on two fresh runs.
Compare runs of THIS system with each other only -- never with human or canonical labels.

Egress: sends the codebook prompt, both titles and the citances of each edge to the configured endpoint, and
nothing else. Refuses any host that is not *.azure.com or localhost unless --allow-host is given. Reads no
human or canonical label columns. Stdlib only.

Configuration, read from citation_corpus/.env or the environment (the key is never printed or written):
  AZURE_FOUNDRY_CHAT_URL   full chat-completions URL of the deployment (copy it from the Foundry portal)
  AZURE_FOUNDRY_API_KEY    key for that deployment
  AZURE_FOUNDRY_MODEL      deployment name, sent as "model"
  AZURE_FOUNDRY_AUTH       api-key (default) | bearer

Modes
  dry-run    build the pass-1 request for the first edge and print it; sends nothing
  edges      code every edge of --input (default: the 20 shared test edges)
  compare    compare two runs of this system: compare <tagA> <tagB>
  estimate   token volume (and cost, if --usd-in/--usd-out are given) from a run's measured usage
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
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
sys.path.insert(0, str(HERE.parent / "local_llm_codebook_v2"))
import prompts as P  # noqa: E402  (the frozen prompt builder shared with the local system)
import run_codebook_v2 as L  # noqa: E402  (validate / flatten / PASSES / SEED, shared)

ROOT = HERE.parents[2]
TEST = ROOT / "analysis/results/external_systems/test_edges.csv"
OUT = ROOT / "analysis/results/external_systems/hosted_llm_codebook_v2"
CACHE = OUT / "responses_cache.jsonl"
ALLOWED_SUFFIXES = (".azure.com",)
LOCAL_HOSTS = ("localhost", "127.0.0.1")
RETRYABLE = {408, 409, 429, 500, 502, 503, 504}


# ---------------------------------------------------------------- configuration
def load_env():
    env = {}
    p = ROOT / ".env"
    if p.exists():
        for line in p.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip().strip('"').strip("'")
    env.update({k: v for k, v in os.environ.items() if k.startswith("AZURE_FOUNDRY_")})
    return env


def config(a):
    env = load_env()
    url = a.url or env.get("AZURE_FOUNDRY_CHAT_URL")
    model = a.model or env.get("AZURE_FOUNDRY_MODEL")
    key = env.get("AZURE_FOUNDRY_API_KEY", "")
    auth = (a.auth or env.get("AZURE_FOUNDRY_AUTH") or "api-key").lower()
    if not url or not model:
        sys.exit("Set AZURE_FOUNDRY_CHAT_URL and AZURE_FOUNDRY_MODEL in citation_corpus/.env (see README.md)")
    host = urllib.parse.urlparse(url).hostname or ""
    local = host in LOCAL_HOSTS
    if not (local or host.endswith(ALLOWED_SUFFIXES) or a.allow_host == host):
        sys.exit(f"REFUSED: host {host!r} is not Azure or localhost; pass --allow-host {host} to override")
    if not key and not local:
        sys.exit("Set AZURE_FOUNDRY_API_KEY in citation_corpus/.env")
    return {"url": url, "host": host, "model": model, "key": key or "local", "auth": auth}


def params(a):
    """Decoding parameters. 'standard' for chat models; 'reasoning' for o-series / reasoning models, which
    reject temperature and top_p and count hidden reasoning tokens against max_completion_tokens."""
    if a.profile == "reasoning":
        out = {"max_completion_tokens": a.max_tokens or 4096, "seed": L.SEED}
        if a.reasoning_effort:
            out["reasoning_effort"] = a.reasoning_effort
        return out
    return {"temperature": 0, "top_p": 1, "seed": L.SEED, "max_tokens": a.max_tokens or L.NUM_PREDICT}


def response_format(a, pass_name):
    if a.response_format == "none":
        return None
    if a.response_format == "json_object":
        return {"type": "json_object"}
    schema = dict(P.schema(pass_name), additionalProperties=False)  # strict mode needs this; prompt unchanged
    return {"type": "json_schema", "json_schema": {"name": f"codebook_v2_{pass_name}", "strict": True,
                                                  "schema": schema}}


def build_body(cfg, a, pass_name, ev):
    body = {"model": cfg["model"],
            "messages": [{"role": "system", "content": P.system_prompt(pass_name)},
                         {"role": "user", "content": ev}]}
    body.update(params(a))
    rf = response_format(a, pass_name)
    if rf:
        body["response_format"] = rf
    return body


# ---------------------------------------------------------------- frozen response cache
class Cache:
    """Append-only. The FIRST response stored for a request key is the frozen one; later fresh runs are
    appended for the record but never replace it."""

    def __init__(self, path):
        self.path, self.lock, self.frozen = path, threading.Lock(), {}
        if path.exists():
            for line in path.open(encoding="utf-8"):
                d = json.loads(line)
                self.frozen.setdefault(d["request_sha256"], d)

    def get(self, key):
        return self.frozen.get(key)

    def put(self, key, record):
        with self.lock:
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
            self.frozen.setdefault(key, record)


def request_key(cfg, body):
    ident = {"host": cfg["host"], "path": urllib.parse.urlparse(cfg["url"]).path, "body": body}
    return hashlib.sha256(json.dumps(ident, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


# ---------------------------------------------------------------- HTTP
def call(cfg, body, max_retries=8):
    headers = {"Content-Type": "application/json"}
    if cfg["auth"] == "bearer":
        headers["Authorization"] = f"Bearer {cfg['key']}"
    else:
        headers["api-key"] = cfg["key"]
    data = json.dumps(body).encode()
    for attempt in range(max_retries + 1):
        req = urllib.request.Request(cfg["url"], data=data, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=180) as r:
                return {"status": r.status, "json": json.loads(r.read())}
        except urllib.error.HTTPError as e:
            text = e.read().decode("utf-8", "replace")
            try:
                err = json.loads(text)
            except json.JSONDecodeError:
                err = {"raw": text[:500]}
            code = str((err.get("error") or {}).get("code", "")) if isinstance(err, dict) else ""
            if "content_filter" in code or "content_filter" in text[:2000]:
                return {"status": e.code, "json": err, "filtered": True}
            if e.code not in RETRYABLE or attempt == max_retries:
                return {"status": e.code, "json": err, "error": True}
            ra = e.headers.get("retry-after-ms")
            wait = float(ra) / 1000 if ra else float(e.headers.get("retry-after") or min(60, 2 ** attempt))
        except (urllib.error.URLError, TimeoutError) as e:
            if attempt == max_retries:
                return {"status": None, "json": {"error": str(e)}, "error": True}
            wait = min(60, 2 ** attempt)
        time.sleep(wait + random.uniform(0, 0.5))
    return {"status": None, "json": {}, "error": True}


# ---------------------------------------------------------------- coding
def code_pass(cfg, a, cache, pass_name, ev, n_cit):
    body = build_body(cfg, a, pass_name, ev)
    key = request_key(cfg, body)
    prompt_sha = hashlib.sha256((P.system_prompt(pass_name) + "\n\n" + ev).encode()).hexdigest()
    rec = None if a.fresh else cache.get(key)
    from_cache = rec is not None
    t0 = time.perf_counter()
    if rec is None:
        resp = call(cfg, body)
        j = resp.get("json") or {}
        choice = (j.get("choices") or [{}])[0] if isinstance(j, dict) else {}
        rec = {"request_sha256": key, "prompt_sha256": prompt_sha, "pass": pass_name,
               "retrieved_utc": datetime.now(timezone.utc).isoformat(), "fresh_run": a.tag if a.fresh else None,
               "http_status": resp.get("status"), "filtered": bool(resp.get("filtered")) or
               choice.get("finish_reason") == "content_filter", "error": bool(resp.get("error")),
               "content": ((choice.get("message") or {}).get("content")) if choice else None,
               "finish_reason": choice.get("finish_reason"), "response_model": j.get("model"),
               "system_fingerprint": j.get("system_fingerprint"), "usage": j.get("usage"),
               "request_params": {k: v for k, v in body.items() if k != "messages"},
               "error_detail": (j.get("error") if isinstance(j, dict) and resp.get("error") else None)}
        if not rec["error"]:
            cache.put(key, rec)
    wall = time.perf_counter() - t0
    raw = rec.get("content") or ""
    try:
        obj, ok = json.loads(raw), True
    except (json.JSONDecodeError, TypeError):
        obj, ok = None, False
    if rec.get("filtered"):
        probs = ["content_filtered"]
    elif rec.get("error"):
        detail = rec.get("error_detail") or {}
        code = detail.get("code") if isinstance(detail, dict) else None
        probs = [f"http_error:{rec.get('http_status')}" + (f":{code}" if code else "")]
    else:
        probs = L.validate(pass_name, obj, n_cit) if ok else ["json_parse_error"]
    return {"raw": raw, "parsed": obj, "json_parse_ok": ok, "schema_problems": probs, "wall_s": round(wall, 3),
            "from_cache": from_cache, "prompt_sha256": prompt_sha, "request_sha256": key,
            "finish_reason": rec.get("finish_reason"), "response_model": rec.get("response_model"),
            "system_fingerprint": rec.get("system_fingerprint"), "usage": rec.get("usage"),
            "filtered": rec.get("filtered"), "retrieved_utc": rec.get("retrieved_utc")}


def code_edge(cfg, a, cache, e):
    cits = [c for c in e["citances"].split("|||") if c.strip()]
    ev = P.evidence(e.get("citing_title", ""), e.get("cited_title", ""), cits)
    res = {p: code_pass(cfg, a, cache, p, ev, len(cits)) for p in L.PASSES}  # passes never see each other
    row = {"edge_id": e["edge_id"], "corpus": e.get("corpus", ""), "cited_id": e.get("cited_id", ""),
           "citing_id": e.get("citing_id", ""), "n_citances": len(cits)}
    row.update(L.flatten(res, len(cits)))
    return e["edge_id"], res, row


def run_edges(a, cfg):
    src = Path(a.input) if a.input else TEST
    with src.open(newline="", encoding="utf-8-sig") as f:
        edges = [{k: r.get(k, "") for k in ("edge_id", "corpus", "cited_id", "citing_id", "citing_title",
                                            "cited_title", "citances")} for r in csv.DictReader(f)]
    if a.limit:
        edges = edges[: a.limit]
    cache, t0 = Cache(CACHE), time.perf_counter()
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        results = list(ex.map(lambda e: code_edge(cfg, a, cache, e), edges))
    elapsed = time.perf_counter() - t0
    results.sort(key=lambda t: t[0])
    with (OUT / f"raw_{a.tag}.jsonl").open("w", encoding="utf-8") as f:
        for eid, res, _ in results:
            f.write(json.dumps({"edge_id": eid, "model": cfg["model"], "passes": res}, ensure_ascii=False) + "\n")
    rows = [r for _, _, r in results]
    with (OUT / f"edges_{a.tag}.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    calls = [res[p] for _, res, _ in results for p in L.PASSES]
    usage = {k: sum(((c["usage"] or {}).get(k) or 0) for c in calls) for k in ("prompt_tokens", "completion_tokens")}
    snap = hashlib.sha256("".join(sorted(c["request_sha256"] + (c["raw"] or "") for c in calls)).encode()).hexdigest()
    meta = {"run_tag": a.tag, "fresh": a.fresh, "input": str(src), "endpoint_host": cfg["host"],
            "model": cfg["model"], "profile": a.profile, "response_format": a.response_format,
            "params": params(a), "n_edges": len(rows), "calls": len(calls),
            "from_cache": sum(c["from_cache"] for c in calls),
            "response_models": sorted({str(c["response_model"]) for c in calls}),
            "system_fingerprints": sorted({str(c["system_fingerprint"]) for c in calls}),
            "json_parse_ok": sum(c["json_parse_ok"] for c in calls),
            "schema_valid": sum(not c["schema_problems"] for c in calls),
            "content_filtered": sum(bool(c["filtered"]) for c in calls),
            "errors": sum(any(s.startswith("http_error") for s in c["schema_problems"]) for c in calls),
            "usage": usage, "tokens_per_edge": {k: round(v / max(1, len(rows)), 1) for k, v in usage.items()},
            "elapsed_s": round(elapsed, 2), "workers": a.workers, "snapshot_sha256": snap,
            "finished_utc": datetime.now(timezone.utc).isoformat()}
    (OUT / f"meta_{a.tag}.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


# ---------------------------------------------------------------- compare / estimate / dry-run
LABELS = ["cited_contribution_type", "citation_function", "credits_contribution", "uptake_mode",
          "uptake_degree", "claim_dependent", "centrality", "rhetorical_polarity"]


def compare(ta, tb):
    load = lambda t: {d["edge_id"]: d for d in map(json.loads, (OUT / f"raw_{t}.jsonl").open())}
    A, B = load(ta), load(tb)
    shared = sorted(set(A) & set(B))
    raw_same = calls = 0
    field = {f: [0, 0] for f in LABELS}
    for e in shared:
        for p in L.PASSES:
            ra, rb = A[e]["passes"][p], B[e]["passes"][p]
            calls += 1
            raw_same += ra["raw"] == rb["raw"]
            for f in P.PASS_FIELDS[p]:
                if f in field:
                    va, vb = (ra["parsed"] or {}).get(f), (rb["parsed"] or {}).get(f)
                    field[f][0] += va == vb
                    field[f][1] += 1
    out = {"runs": [ta, tb], "edges": len(shared), "calls": calls, "raw_identical": raw_same,
           "label_agreement": {f: f"{s}/{n} = {s / n:.3f}" for f, (s, n) in field.items() if n},
           "note": "run-to-run stability of this system only; not compared with any human label"}
    (OUT / f"stability_{ta}_vs_{tb}.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


def estimate(a):
    m = json.loads((OUT / f"meta_{a.tag}.json").read_text())
    per = m["tokens_per_edge"]
    out = {"from_run": a.tag, "tokens_per_edge": per}
    for n in (2937, 226793):
        t_in, t_out = per["prompt_tokens"] * n, per["completion_tokens"] * n
        row = {"input_tokens": int(t_in), "output_tokens": int(t_out)}
        if a.usd_in is not None and a.usd_out is not None:
            row["usd"] = round(t_in / 1e6 * a.usd_in + t_out / 1e6 * a.usd_out, 2)
        out[f"{n}_edges"] = row
    print(json.dumps(out, indent=2))


def dry_run(a, cfg):
    with TEST.open(newline="", encoding="utf-8-sig") as f:
        e = next(csv.DictReader(f))
    cits = [c for c in e["citances"].split("|||") if c.strip()]
    body = build_body(cfg, a, "pass1", P.evidence(e["citing_title"], e["cited_title"], cits))
    shown = dict(body, messages=[{"role": m["role"], "content": m["content"][:160] + " ..."} for m in body["messages"]])
    u = urllib.parse.urlparse(cfg["url"])
    print(f"POST {u.scheme}://{u.netloc}{u.path}  (auth: {cfg['auth']}, key not shown)")
    print(json.dumps(shown, indent=2, ensure_ascii=False))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["dry-run", "edges", "compare", "estimate"])
    ap.add_argument("tags", nargs="*", help="compare: two run tags")
    ap.add_argument("--tag", default="run1")
    ap.add_argument("--input", help="CSV with edge_id, citing_title, cited_title, citances ('|||'-joined)")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--fresh", action="store_true", help="ignore the frozen cache (stability runs)")
    ap.add_argument("--profile", choices=["standard", "reasoning"], default="standard")
    ap.add_argument("--reasoning-effort", help="reasoning profile only: sent as reasoning_effort "
                    "(e.g. none, minimal, low); use the lowest the deployment accepts and record it")
    ap.add_argument("--response-format", choices=["json_schema", "json_object", "none"], default="json_schema")
    ap.add_argument("--max-tokens", type=int)
    ap.add_argument("--url")
    ap.add_argument("--model")
    ap.add_argument("--auth", choices=["api-key", "bearer"])
    ap.add_argument("--allow-host")
    ap.add_argument("--usd-in", type=float, help="USD per 1M input tokens")
    ap.add_argument("--usd-out", type=float, help="USD per 1M output tokens")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.mode == "compare":
        return compare(*a.tags)
    if a.mode == "estimate":
        return estimate(a)
    cfg = config(a)
    return dry_run(a, cfg) if a.mode == "dry-run" else run_edges(a, cfg)


if __name__ == "__main__":
    main()
