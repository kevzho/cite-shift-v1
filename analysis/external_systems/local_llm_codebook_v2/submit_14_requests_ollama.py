#!/usr/bin/env python3
"""Missing link between experiments/14 and experiments/15: submit 14's frozen requests to a LOCAL ollama model.

Script 14 only WRITES frozen prompts (model_id 'UNSET_REQUIRES_AUTHORIZED_MODEL'); it has no client.
Script 15 only READS a CSV with columns cited_id,citing_id,generic_intent,contribution_use,raw_response.
Nothing in the repo submits requests. This adapter does, to localhost only, at temperature 0 with the
request's own randomization_seed, and writes the CSV that 15 expects.

It is NOT run as part of the codebook-v2 feasibility test (default is --dry-run). Running it for real on
14's 1,040 human-reviewed HCI rows, and then running 15 (which computes accuracy against
realizes_contribution), must wait until the evaluation is pre-registered. Note also that 14's schema is the
older generic one (generic_intent / contribution_use), not codebook v2.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
REQ = ROOT / "analysis/results/experiments/frozen_llm_baseline_requests.jsonl"
OUT = ROOT / "analysis/results/external_systems/local_llm_codebook_v2/frozen_llm_baseline_responses_ollama.csv"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--requests", type=Path, default=REQ)
    ap.add_argument("--output", type=Path, default=OUT)
    ap.add_argument("--model", default="qwen2.5:7b-instruct")
    ap.add_argument("--host", default="http://localhost:11434")
    ap.add_argument("--execute", action="store_true", help="actually call the local model (default: dry run)")
    a = ap.parse_args()
    if not a.host.startswith(("http://localhost", "http://127.0.0.1")):
        sys.exit("REFUSED: host must be local")
    reqs = [json.loads(l) for l in a.requests.open(encoding="utf-8")]
    unset = sum(r["model_id"].startswith("UNSET") for r in reqs)
    print(f"{len(reqs)} requests; {unset} carry model_id=UNSET (regenerate with 14 --model-id "
          f"'ollama:{a.model}@<digest>' so the manifest names the model)")
    if not a.execute:
        r = reqs[0]
        print("dry run; first payload would be:",
              json.dumps({"model": a.model, "format": "<schema from response_schema>",
                          "options": {"temperature": r["temperature"], "top_k": 1, "seed": r["randomization_seed"],
                                      "num_ctx": 4096}, "prompt_sha256": r["prompt_hash"]}))
        return
    rows = []
    for r in reqs:
        rs = r["response_schema"]
        schema = {"type": "object", "required": ["generic_intent", "contribution_use", "rationale"],
                  "properties": {"generic_intent": {"type": "string", "enum": rs["generic_intent"]},
                                 "contribution_use": {"type": "string", "enum": rs["contribution_use"]},
                                 "rationale": {"type": "string"}}}
        payload = {"model": a.model, "stream": False, "format": schema, "keep_alive": "15m",
                   "options": {"temperature": r["temperature"], "top_k": 1, "seed": r["randomization_seed"],
                               "num_ctx": 4096, "num_predict": 200},
                   "messages": [{"role": "user", "content": r["prompt"]}]}
        q = urllib.request.Request(a.host + "/api/chat", data=json.dumps(payload).encode(),
                                   headers={"Content-Type": "application/json"})
        raw = json.loads(urllib.request.urlopen(q, timeout=600).read())["message"]["content"]
        try:
            o = json.loads(raw)
        except json.JSONDecodeError:
            o = {}
        rows.append({"request_id": r["request_id"], "cited_id": r["cited_id"], "citing_id": r["citing_id"],
                     "generic_intent": o.get("generic_intent", ""), "contribution_use": o.get("contribution_use", ""),
                     "raw_response": raw})
    with a.output.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} responses to {a.output}; do NOT run 15 before pre-registration")


if __name__ == "__main__":
    main()
