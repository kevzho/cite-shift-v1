#!/usr/bin/env python3
"""Harness for Arnaout, Sternlicht, Hope & Gurevych (2025) "impact-revealing" citation classifier.

The published classifier (repo UKPLab/arxiv2025-generating-impact-summaries,
src/generations/generate_fine_grained_intents.py) is an in-context-learning prompt
(K=50 examples) sent to OpenAI gpt-4o-mini, temperature 0, max_tokens 80, run 3 times
with the example order shuffled within label, majority vote. There are NO released
model weights. This harness reproduces that pipeline verbatim (prompt strings are
pulled out of the repo source with `ast`, so they are byte-identical) and lets you
choose the backend:

  --backend dry-run   (default) build the exact prompts for every edge/run and write them
                      to disk. Nothing leaves the machine. This is the furthest step of
                      the PUBLISHED system that can be verified without sending project
                      data to OpenAI.
  --backend openai    the published system. Refuses to run unless
                      --allow-external-api is passed, because it transmits citances to
                      OpenAI. NOT run in this evaluation (project rule forbids it).
  --backend ollama    LOCAL SUBSTITUTE: same prompt, same voting, but a local open-weight
                      model served by ollama on 127.0.0.1. This is NOT Arnaout's system
                      (different LLM, no published validation) and every output row is
                      labelled system_variant=substitute.

Usage:
  .venv/bin/python run_arnaout.py --backend dry-run
  .venv/bin/python run_arnaout.py --backend ollama --model qwen2.5:7b-instruct
"""
import argparse
import ast
import csv
import json
import os
import random
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.join(HERE, "repo")
SRC = os.path.join(REPO, "src/generations/generate_fine_grained_intents.py")
EXAMPLES = os.path.join(REPO, "data/training_examples_k=50.csv")
EDGES = os.path.join(HERE, "..", "test_edges.csv")
K = 50
N_RUNS = 3


def load_prompt_strings():
    """Pull the two prompt string literals verbatim from the published script."""
    tree = ast.parse(open(SRC, encoding="utf-8").read())
    header = gold = None
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if "A citation context in a scientific paper" in node.value:
                header = node.value
            elif "Eisenberg (2008) integrates" in node.value:
                gold = node.value
    assert header and gold, "prompt strings not found in repo source"
    return header, gold


def load_examples():
    # same reader/encoding as the published script
    with open(EXAMPLES, mode="r", encoding="ISO-8859-1") as f:
        return [row for row in csv.reader(f)]


def build_prompt(header, gold, examples, context, rng):
    # identical to published code: random.sample(examples, k) sorted by label (stable sort
    # => shuffled within label), then the two gold examples, then the query context.
    examples_of_run = sorted(rng.sample(examples, K), key=lambda x: x[3])
    p = header
    for ex in examples_of_run:
        p += f"<{ex[1]}> => " f'{{"description": "{ex[2]}", "label": "{ex[3]}"}}\n'
    p += gold
    p += f"<{context}> => "
    return p


def normalise_context(citances):
    # Published code: SS_contexts.replace("###", " ") then whitespace-collapse.
    # Our test edges join multiple citances with "|||" instead of S2's "###".
    c = citances.replace("|||", " ").replace("###", " ").strip()
    return " ".join(c.split())


def parse_label(resp):
    # verbatim parsing logic from published code
    label = None
    try:
        data = json.loads(resp)
        label = data.get("label", "").lower()
    except Exception:
        rl = resp.lower()
        if '"impact-revealing"' in rl or "'impact-revealing'" in rl:
            label = "impact-revealing"
        elif '"other"' in rl or "'other'" in rl:
            label = "other"
    return label


def call_ollama(model, prompt, host):
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {"temperature": 0.0, "num_predict": 80, "num_ctx": 16384,
                    "presence_penalty": 0, "frequency_penalty": 0},
    }).encode()
    req = urllib.request.Request(host + "/api/chat", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        d = json.load(r)
    return d["message"]["content"].strip(), d.get("prompt_eval_count")


def call_openai(model, prompt):
    from openai import OpenAI  # not installed in this venv on purpose
    client = OpenAI()
    resp = client.chat.completions.create(model=model, temperature=0.0, max_tokens=80,
                                          presence_penalty=0, frequency_penalty=0,
                                          messages=[{"role": "user", "content": prompt}])
    return resp.choices[0].message.content.strip(), None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["dry-run", "openai", "ollama"], default="dry-run")
    ap.add_argument("--model", default=None)
    ap.add_argument("--host", default="http://127.0.0.1:11434")
    ap.add_argument("--allow-external-api", action="store_true")
    ap.add_argument("--seed", type=int, default=20250520,
                    help="published code is unseeded; we seed for reproducibility")
    ap.add_argument("--edges", default=EDGES)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    if a.backend == "openai" and not a.allow_external_api:
        sys.exit("REFUSED: openai backend sends citances to a hosted API. Project rules forbid this.")
    if a.backend == "ollama" and not a.host.startswith(("http://127.0.0.1", "http://localhost")):
        sys.exit("REFUSED: ollama host must be local")
    model = a.model or {"openai": "gpt-4o-mini", "ollama": "qwen2.5:7b-instruct"}.get(a.backend)

    header, gold = load_prompt_strings()
    examples = load_examples()
    assert len(examples) == 50
    edges = list(csv.DictReader(open(a.edges, encoding="utf-8")))
    tag = a.backend if a.backend != "ollama" else "ollama_" + model.replace(":", "_").replace("/", "_")
    out = a.out or os.path.join(HERE, "outputs", f"preds_{tag}.jsonl")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    pdir = os.path.join(HERE, "outputs", "prompts")
    os.makedirs(pdir, exist_ok=True)

    rng = random.Random(a.seed)
    variant = {"dry-run": "published_prompt_no_llm", "openai": "published_system",
               "ollama": "substitute_local_llm_NOT_published_system"}[a.backend]
    t_all = time.time()
    with open(out, "w") as fo:
        for i, e in enumerate(edges):
            ctx = normalise_context(e["citances"])
            preds, resps, ntok = [], [], []
            for run in range(N_RUNS):
                prompt = build_prompt(header, gold, examples, ctx, rng)
                if a.backend == "dry-run":
                    with open(os.path.join(pdir, f"{e['edge_id']}_run{run}.txt"), "w") as fp:
                        fp.write(prompt)
                    ntok.append(len(prompt))
                    continue
                t0 = time.time()
                if a.backend == "ollama":
                    resp, pt = call_ollama(model, prompt, a.host)
                else:
                    resp, pt = call_openai(model, prompt)
                resps.append(resp)
                ntok.append(pt)
                preds.append(parse_label(resp))
                print(f"[{i+1}/{len(edges)}] {e['edge_id']} run{run} {time.time()-t0:.1f}s -> {resp!r}",
                      flush=True)
            if a.backend == "dry-run":
                final = None
            elif len(preds) == 3:
                final = ("impact-revealing" if preds.count("impact-revealing") > preds.count("other")
                         else "other")
            else:
                final = "error"
            rec = {"edge_id": e["edge_id"], "corpus": e["corpus"], "system_variant": variant,
                   "backend": a.backend, "model": model, "seed": a.seed,
                   "context_sent": ctx, "run_labels": preds, "run_responses": resps,
                   "prompt_size": ntok, "impact_revealing": final,
                   "canonical_function": e["canonical_function"],
                   "canonical_realization": e["canonical_realization"],
                   "bulk_or_core": e["bulk_or_core"]}
            fo.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"wrote {out} in {time.time()-t_all:.1f}s")


if __name__ == "__main__":
    main()
