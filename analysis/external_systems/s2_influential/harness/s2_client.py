"""Minimal, rate-limited Semantic Scholar Graph API client.

Only public paper identifiers (CorpusId:<n>) are ever sent. No citance, title or full text
leaves the machine. The API key is read from the project .env and is never printed,
logged or written anywhere (it only travels in the x-api-key request header).
"""
import os as _os
from pathlib import Path as _Path
_CITATION_ROOT = _os.environ.get("CITATION_ROOT", str(_Path(__file__).resolve().parents[4]))
import json
import os
import time
from pathlib import Path

import requests

REPO = Path(_CITATION_ROOT)
OUT = REPO / "analysis/results/external_systems/s2_influential"
BASE = "https://api.semanticscholar.org/graph/v1"
MIN_INTERVAL = float(os.environ.get("S2_MIN_INTERVAL", "1.05"))  # key tier: 1 request / second

# Edge-level fields of the /references record (citedPaper fields are requested by name, unprefixed).
REF_FIELDS = "isInfluential,intents,contexts,contextsWithIntent,corpusId,externalIds,title,year"


def _load_key():
    for line in (REPO / ".env").read_text().splitlines():
        line = line.strip()
        if line.startswith("S2_API_KEY="):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise SystemExit("S2_API_KEY not found in .env")


class S2:
    def __init__(self):
        self.s = requests.Session()
        self.s.headers["x-api-key"] = _load_key()
        self._last = 0.0
        self.n_requests = 0
        self.n_429 = 0

    def _wait(self):
        dt = time.monotonic() - self._last
        if dt < MIN_INTERVAL:
            time.sleep(MIN_INTERVAL - dt)
        self._last = time.monotonic()

    def request(self, method, path, params=None, body=None, max_tries=8):
        url = BASE + path
        for attempt in range(max_tries):
            self._wait()
            self.n_requests += 1
            try:
                r = self.s.request(method, url, params=params, json=body, timeout=60)
            except requests.exceptions.RequestException:
                if attempt == max_tries - 1:
                    raise
                time.sleep(min(60, 2 ** attempt))
                continue
            if r.status_code == 429 or r.status_code >= 500:
                self.n_429 += r.status_code == 429
                time.sleep(min(60, 2 ** attempt))
                continue
            return r
        return r

    def references(self, citing_corpus_id, page=1000):
        """All reference records of one citing paper (paginated). Returns (status, pages)."""
        pages, offset = [], 0
        while True:
            r = self.request("GET", f"/paper/CorpusId:{citing_corpus_id}/references",
                             params={"fields": REF_FIELDS, "limit": page, "offset": offset})
            pages.append({"url": r.url, "status": r.status_code, "body": _safe_json(r)})
            if r.status_code != 200:
                return r.status_code, pages
            nxt = pages[-1]["body"].get("next")
            if nxt is None:
                return 200, pages
            offset = nxt

    def citations(self, cited_corpus_id, page=1000, cap=10000):
        """Citation records of one cited paper, from the cited side (fallback view)."""
        pages, offset = [], 0
        while offset < cap:
            r = self.request("GET", f"/paper/CorpusId:{cited_corpus_id}/citations",
                             params={"fields": REF_FIELDS, "limit": page, "offset": offset})
            pages.append({"url": r.url, "status": r.status_code, "body": _safe_json(r)})
            if r.status_code != 200:
                return r.status_code, pages
            nxt = pages[-1]["body"].get("next")
            if nxt is None:
                return 200, pages
            offset = nxt
        return 200, pages

    def batch(self, corpus_ids, fields):
        r = self.request("POST", "/paper/batch", params={"fields": fields},
                         body={"ids": [f"CorpusId:{c}" for c in corpus_ids]})
        return r.status_code, _safe_json(r)


def _safe_json(r):
    try:
        return r.json()
    except ValueError:
        return {"_non_json": r.text[:500]}


def dump(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, ensure_ascii=False, sort_keys=True))
