"""Shared HTTP helper for the Nature arm build (adapted from the 27 Sep pilot's code/common.py).

Sequential, polite, no personal identifiers (no mailto/email/api key). Every request is appended
to logs/requests.jsonl (url, status, bytes, seconds, utc, attempt). Retry with backoff on 429/5xx/network.
"""
from __future__ import annotations
import json, time, urllib.request, urllib.error
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__import__("os").environ.get("NATURE_ARM_DIR", Path(__file__).resolve().parents[2] / "external_data/nature_arm_2026-10-04"))
REPO = Path(__import__("os").environ.get("CITATION_ROOT", Path(__file__).resolve().parents[2]))
LOG = ROOT / "logs" / "requests.jsonl"
UA = "nature-arm-build/0.1 (academic research; sequential requests)"
EPMC = "https://www.ebi.ac.uk/europepmc/webservices/rest"
S3 = "https://pmc-oa-opendata.s3.amazonaws.com"


def utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def get(url: str, delay: float = 0.35, tries: int = 5, timeout: int = 120) -> tuple[int, bytes, dict]:
    wait = 2.0
    status, body, hdrs = 0, b"", {}
    for attempt in range(tries):
        t0 = time.time()
        req = urllib.request.Request(url, headers={"User-Agent": UA})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                status, body, hdrs = r.status, r.read(), dict(r.headers)
        except urllib.error.HTTPError as e:
            status, body, hdrs = e.code, e.read() or b"", dict(e.headers or {})
        except Exception as e:
            status, body, hdrs = -1, str(e).encode(), {}
        with LOG.open("a") as fh:
            fh.write(json.dumps({"utc": utc(), "url": url.split("?md5=")[0], "status": status,
                                 "bytes": len(body), "sec": round(time.time() - t0, 3),
                                 "attempt": attempt}) + "\n")
        time.sleep(delay)
        if status == 429 or status >= 500 or status == -1:
            ra = hdrs.get("Retry-After")
            time.sleep(float(ra) if ra and ra.isdigit() else wait)
            wait *= 2
            continue
        break
    return status, body, hdrs
