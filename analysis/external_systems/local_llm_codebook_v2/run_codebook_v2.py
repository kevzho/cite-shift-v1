#!/usr/bin/env python3
"""Project's own frozen-LLM measured system for codebook v2, on a LOCAL open-weight model (ollama).

Not a published external system. Three independent calls per edge (codebook v2 section 1 passes);
no pass sees another pass's output. Greedy decoding (temperature 0, top_k 1), fixed seed, fixed
context length, JSON-schema-constrained output (ollama structured outputs).

Modes
  edges       : score test_edges.csv; writes raw JSONL + per-edge CSV (no human/canonical columns copied)
  throughput  : time a seeded random sample of chi_chiea_citances.jsonl; writes ONLY timing stats

Pre-registration: this script never reads canonical_* columns and computes no agreement statistic.
Stdlib only.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import prompts as P  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
TEST = ROOT / "analysis/results/external_systems/test_edges.csv"
CIT = ROOT / "external_data/chi_chiea_grobid_2026-09-15/citances/chi_chiea_citances.jsonl"
OUT = ROOT / "analysis/results/external_systems/local_llm_codebook_v2"

SEED = 20260926
NUM_CTX = 8192
NUM_PREDICT = 256
PASSES = ["pass1", "pass2"]  # Pass 3 dropped from the study (plan, 2 Oct); codebook v2 frozen 3 Oct


def options():
    return {"temperature": 0.0, "top_k": 1, "top_p": 1.0, "seed": SEED, "num_ctx": NUM_CTX,
            "num_predict": NUM_PREDICT, "repeat_penalty": 1.0}


def post(host, path, payload, timeout=600):
    req = urllib.request.Request(host + path, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def ollama_version(host):
    with urllib.request.urlopen(host + "/api/version") as r:
        return json.loads(r.read()).get("version")


def model_digest(host, model):
    with urllib.request.urlopen(host + "/api/tags") as r:
        for m in json.loads(r.read())["models"]:
            if m["name"] == model:
                return m["digest"]
    raise SystemExit(f"model {model} not present in local ollama")


def validate(pass_name, obj, n_cit):
    """Return list of problems (empty = valid)."""
    probs = []
    if not isinstance(obj, dict):
        return ["not_object"]
    for f in P.PASS_FIELDS[pass_name]:
        if f not in obj:
            probs.append(f"missing:{f}")
            continue
        v = obj[f]
        if f == "strongest_citance":
            if not isinstance(v, int) or not (1 <= v <= n_cit):
                probs.append("strongest_citance_out_of_range")
        elif f in P.ENUMS and v not in P.ENUMS[f]:
            probs.append(f"bad_enum:{f}={v}")
    return probs


def code_edge(host, model, citing_title, cited_title, citances):
    ev = P.evidence(citing_title, cited_title, citances)
    res = {}
    for p in PASSES:
        sysmsg = P.system_prompt(p)
        payload = {"model": model, "stream": False, "keep_alive": "15m", "format": P.schema(p),
                   "options": options(),
                   "messages": [{"role": "system", "content": sysmsg}, {"role": "user", "content": ev}]}
        t0 = time.perf_counter()
        r = post(host, "/api/chat", payload)
        wall = time.perf_counter() - t0
        raw = r["message"]["content"]
        try:
            obj = json.loads(raw)
            parse_ok = True
        except json.JSONDecodeError:
            obj, parse_ok = None, False
        probs = validate(p, obj, len(citances)) if parse_ok else ["json_parse_error"]
        res[p] = {"raw": raw, "parsed": obj, "json_parse_ok": parse_ok, "schema_problems": probs,
                  "wall_s": round(wall, 3), "prompt_eval_count": r.get("prompt_eval_count"),
                  "eval_count": r.get("eval_count"), "done_reason": r.get("done_reason"),
                  "prompt_sha256": hashlib.sha256((sysmsg + "\n\n" + ev).encode()).hexdigest()}
    return res


def flatten(res, n_cit):
    row = {}
    for p in PASSES:
        o = res[p]["parsed"] or {}
        for f in P.PASS_FIELDS[p]:
            key = f"{p}_{f}" if f in ("strongest_citance", "note") else f
            row[key] = o.get(f, "")
    if row.get("uptake_degree") == "not_applicable":
        row["uptake_degree"] = ""
    mode, deg = row.get("uptake_mode"), row.get("uptake_degree")
    row["degree_rule_ok"] = int((mode in ("none", "unclear")) == (deg == ""))
    row["all_passes_valid"] = int(all(not res[p]["schema_problems"] for p in PASSES))
    row["wall_s_total"] = round(sum(res[p]["wall_s"] for p in PASSES), 3)
    return row


def run_edges(a, host, digest):
    src = Path(a.input) if a.input else TEST
    with src.open(newline="", encoding="utf-8-sig") as f:
        edges = list(csv.DictReader(f))
    if a.order == "reverse":
        edges = edges[::-1]
    tag = a.run_tag
    raw_path, csv_path = OUT / f"raw_{tag}.jsonl", OUT / f"edges_{tag}.csv"
    if csv_path.exists():
        sys.exit(f"refusing to overwrite {csv_path}")
    # resume (11 Oct): edges already in raw_<tag>.jsonl are re-read, not re-sent (greedy decoding, same options)
    done = {}
    if raw_path.exists():
        for line in raw_path.open(encoding="utf-8"):
            d = json.loads(line)
            done[d["edge_id"]] = d["passes"]
    by_id = {e["edge_id"]: e for e in edges}
    rows, t_all = [], time.perf_counter()
    for eid, res in done.items():
        e = by_id[eid]
        cits = [c for c in e["citances"].split("|||") if c.strip()]
        row = {"edge_id": eid, "corpus": e["corpus"], "cited_id": e["cited_id"], "citing_id": e["citing_id"],
               "n_citances": len(cits)}
        row.update(flatten(res, len(cits)))
        rows.append(row)
    with raw_path.open("a", encoding="utf-8") as fr:
        for i, e in enumerate(edges, 1):
            if e["edge_id"] in done:
                continue
            cits = [c for c in e["citances"].split("|||") if c.strip()]
            res = code_edge(host, a.model, e["citing_title"], e["cited_title"], cits)
            fr.write(json.dumps({"edge_id": e["edge_id"], "model": a.model, "model_digest": digest,
                                 "options": options(), "passes": res}, ensure_ascii=False) + "\n")
            fr.flush()
            row = {"edge_id": e["edge_id"], "corpus": e["corpus"], "cited_id": e["cited_id"],
                   "citing_id": e["citing_id"], "n_citances": len(cits)}
            row.update(flatten(res, len(cits)))
            rows.append(row)
            print(f"[{tag}] {i}/{len(edges)} {e['edge_id']} {row['wall_s_total']}s valid={row['all_passes_valid']}",
                  flush=True)
    elapsed = time.perf_counter() - t_all
    rows.sort(key=lambda r: r["edge_id"])
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    meta = {"run_tag": tag, "input": str(src), "resumed_edges": len(done), "order": a.order, "model": a.model,
            "model_digest": digest, "ollama_version": ollama_version(host),
            "options": options(), "n_edges": len(rows), "elapsed_s": round(elapsed, 2),
            "s_per_edge": round(elapsed / len(rows), 3),
            "finished_utc": datetime.now(timezone.utc).isoformat()}
    (OUT / f"meta_{tag}.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


def run_throughput(a, host, digest):
    with CIT.open(encoding="utf-8") as f:
        n_lines = sum(1 for _ in f)
    idx = set(random.Random(SEED).sample(range(n_lines), a.sample))
    sample = []
    with CIT.open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i in idx:
                sample.append(json.loads(line))
    per_edge, n_valid, n_parse, n_calls, prompt_toks, eval_toks = [], 0, 0, 0, 0, 0
    t_all = time.perf_counter()
    for j, d in enumerate(sample, 1):
        cits = [c for c in d.get("citances", []) if c and c.strip()] or [""]
        # citing title is not in this file; timing only, outputs discarded
        res = code_edge(host, a.model, "", d.get("cited_title", ""), cits)
        per_edge.append(sum(res[p]["wall_s"] for p in PASSES))
        for p in PASSES:
            n_calls += 1
            n_parse += res[p]["json_parse_ok"]
            n_valid += not res[p]["schema_problems"]
            prompt_toks += res[p]["prompt_eval_count"] or 0
            eval_toks += res[p]["eval_count"] or 0
        if j % 20 == 0:
            print(f"[throughput] {j}/{len(sample)} mean {sum(per_edge)/len(per_edge):.2f}s/edge", flush=True)
        del res
    elapsed = time.perf_counter() - t_all
    per_edge.sort()
    s = elapsed / len(sample)
    stats = {"model": a.model, "model_digest": digest, "options": options(), "sample_seed": SEED,
             "population_lines": n_lines, "sample_n": len(sample), "elapsed_s": round(elapsed, 1),
             "s_per_edge_mean": round(s, 3), "s_per_edge_median": round(per_edge[len(per_edge)//2], 3),
             "s_per_edge_p95": round(per_edge[int(0.95*len(per_edge))], 3),
             "edges_per_hour": round(3600 / s, 1), "calls": n_calls, "json_parse_ok": n_parse,
             "schema_valid": n_valid, "prompt_tokens_evaluated": prompt_toks, "output_tokens": eval_toks,
             "est_hours_2937": round(2937 * s / 3600, 2), "est_hours_226793": round(226793 * s / 3600, 1),
             "note": "outputs discarded; timing only. prompt_tokens_evaluated excludes ollama prefix-cache hits",
             "finished_utc": datetime.now(timezone.utc).isoformat()}
    (OUT / f"throughput_{a.run_tag}.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["edges", "throughput", "show-prompts"])
    ap.add_argument("--model", default="qwen2.5:7b-instruct")
    ap.add_argument("--host", default="http://localhost:11434")
    ap.add_argument("--run-tag", default="run1")
    ap.add_argument("--order", choices=["forward", "reverse"], default="forward")
    ap.add_argument("--sample", type=int, default=200)
    ap.add_argument("--input", help="edges CSV (edge_id, corpus, cited_id, citing_id, citing_title, cited_title, "
                                    "citances '|||'-joined); default test_edges.csv")
    a = ap.parse_args()
    if not a.host.startswith(("http://localhost", "http://127.0.0.1")):
        sys.exit("REFUSED: host must be local; citances never leave this machine")
    OUT.mkdir(parents=True, exist_ok=True)
    if a.mode == "show-prompts":
        out = {p: {"system": P.system_prompt(p), "schema": P.schema(p),
                   "sha256": hashlib.sha256(P.system_prompt(p).encode()).hexdigest()} for p in PASSES}
        (OUT / "frozen_prompts.json").write_text(json.dumps(out, indent=2))
        for p in PASSES:
            print(p, out[p]["sha256"], len(out[p]["system"]), "chars")
        return
    digest = model_digest(a.host, a.model)
    (run_edges if a.mode == "edges" else run_throughput)(a, a.host, digest)


if __name__ == "__main__":
    main()
