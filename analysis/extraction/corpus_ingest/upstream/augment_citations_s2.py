"""
Augment GROBID-extracted citations using the S2 Asta MCP API.

Input:  <paper>.citations.json files produced by parse_grobid.py in a directory
Output: <paper>.citations.s2.json enriched with Semantic Scholar metadata

This script is designed for:
- Configurable concurrency (ThreadPoolExecutor)
- Global request rate limiting (e.g. 10 RPS with your API key)
- Failure handling + resume via:
  - A local cache file (so we don't re-query the same DOI/title again)
  - A progress file (for overall status / debugging)

Docs: https://allenai.org/asta/resources/mcp
MCP spec (JSON-RPC over HTTP): https://modelcontextprotocol.io/specification/2025-06-18/basic/transports
"""

from __future__ import annotations

import html
import json
import os
import random
import re
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Literal

import requests
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()


ASTA_MCP_ENDPOINT_DEFAULT = "https://asta-tools.allen.ai/mcp/v1"
MCP_PROTOCOL_VERSION = "2025-06-18"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


class RateLimiter:
    """
    Simple global rate limiter: at most `rps` acquisitions per second.
    Thread-safe. Suitable for cross-thread throttling with requests.
    """

    def __init__(self, rps: float):
        self._min_interval = 1.0 / max(0.0001, rps)
        self._lock = threading.Lock()
        self._next_allowed = time.monotonic()

    def acquire(self) -> None:
        with self._lock:
            now = time.monotonic()
            if now < self._next_allowed:
                time.sleep(self._next_allowed - now)
            self._next_allowed = time.monotonic() + self._min_interval


class McpHttpClient:
    """
    Minimal MCP client over Streamable HTTP transport (JSON-RPC via POST).

    We implement:
    - initialize
    - notifications/initialized
    - tools/call

    We also store Mcp-Session-Id if provided by the server.
    """

    def __init__(
        self,
        *,
        endpoint: str,
        api_key: str,
        timeout_s: int = 60,
        client_name: str = "citation-impact-agent",
        client_version: str = "0.1.0",
    ):
        self.endpoint = endpoint
        self.api_key = api_key
        self.timeout_s = timeout_s
        self.client_name = client_name
        self.client_version = client_version

        self._session_id: str | None = None
        self._negotiated_protocol_version: str | None = None
        self._initialized = False
        self._init_lock = threading.Lock()
        self._state_lock = threading.Lock()

    def _headers(self) -> dict[str, str]:
        headers = {
            "Accept": "application/json, text/event-stream",
            "Content-Type": "application/json",
            "x-api-key": self.api_key,
        }
        with self._state_lock:
            if self._session_id:
                headers["Mcp-Session-Id"] = self._session_id
            if self._negotiated_protocol_version:
                headers["MCP-Protocol-Version"] = self._negotiated_protocol_version
        return headers

    def _post_jsonrpc(self, payload: dict[str, Any]) -> dict[str, Any]:
        resp = requests.post(
            self.endpoint,
            headers=self._headers(),
            data=json.dumps(payload),
            timeout=self.timeout_s,
        )

        # Notifications (and other accepted inputs) may return 202 with no body.
        if resp.status_code == 202:
            return {}

        # Most MCP servers reply with JSON for single-response requests.
        ctype = (resp.headers.get("Content-Type") or "").lower()
        if ctype.startswith("text/event-stream"):
            # Basic SSE parsing: collect "data:" frames and return the first JSON-RPC
            # response matching our request id.
            req_id = payload.get("id")
            for line in resp.text.splitlines():
                line = line.strip()
                if not line.startswith("data:"):
                    continue
                data = line[len("data:") :].strip()
                if not data:
                    continue
                try:
                    msg = json.loads(data)
                except Exception:
                    continue
                if req_id is not None and msg.get("id") == req_id:
                    return msg
            raise RuntimeError(f"SSE response did not contain id={payload.get('id')}")

        resp.raise_for_status()

        # Capture session ID if the server assigns one.
        if "Mcp-Session-Id" in resp.headers:
            with self._state_lock:
                self._session_id = resp.headers["Mcp-Session-Id"]

        if not resp.content:
            return {}

        try:
            return resp.json()
        except Exception as e:
            preview = (resp.text or "")[:500]
            raise RuntimeError(f"Failed to parse JSON-RPC response: {e}; status={resp.status_code}; body[:500]={preview!r}")

    def ensure_initialized(self) -> None:
        if self._initialized:
            return

        # Only one thread should perform MCP initialization; others wait.
        with self._init_lock:
            if self._initialized:
                return

            init_id = str(uuid.uuid4())
            init_payload = {
                "jsonrpc": "2.0",
                "id": init_id,
                "method": "initialize",
                "params": {
                    "protocolVersion": MCP_PROTOCOL_VERSION,
                    "capabilities": {"roots": {}, "sampling": {}, "elicitation": {}},
                    "clientInfo": {"name": self.client_name, "version": self.client_version},
                },
            }
            init_resp = self._post_jsonrpc(init_payload)
            if "error" in init_resp:
                raise RuntimeError(f"MCP initialize error: {init_resp['error']}")

            result = init_resp.get("result") or {}
            negotiated = result.get("protocolVersion") or MCP_PROTOCOL_VERSION
            with self._state_lock:
                self._negotiated_protocol_version = negotiated

            # Send notifications/initialized (no id). Server commonly returns 202 with no body.
            self._post_jsonrpc({"jsonrpc": "2.0", "method": "notifications/initialized"})
            self._initialized = True

    def tools_call(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        self.ensure_initialized()
        req_id = str(uuid.uuid4())
        payload = {
            "jsonrpc": "2.0",
            "id": req_id,
            "method": "tools/call",
            "params": {"name": name, "arguments": arguments},
        }
        resp = self._post_jsonrpc(payload)
        if "error" in resp:
            raise RuntimeError(f"MCP tools/call error for {name}: {resp['error']}")
        return resp.get("result") or {}


def parse_tool_result(result: dict[str, Any]) -> dict[str, Any]:
    """
    Asta tools typically return ToolResult content. Prefer structuredContent if present.
    Otherwise, try to parse JSON from a single text block.
    """
    if "structuredContent" in result and isinstance(result["structuredContent"], dict):
        return result["structuredContent"]

    content = result.get("content") or []
    if not isinstance(content, list):
        return {"_raw_result": result}

    # Common pattern: single text content containing JSON.
    if len(content) == 1 and content[0].get("type") == "text":
        txt = content[0].get("text") or ""
        txt = txt.strip()
        if txt.startswith("{") or txt.startswith("["):
            try:
                parsed = json.loads(txt)
                if isinstance(parsed, dict):
                    return parsed
                return {"_parsed": parsed}
            except Exception:
                return {"_text": txt, "_raw_result": result}

    # Fallback: concatenate text blocks.
    texts = []
    for item in content:
        if isinstance(item, dict) and item.get("type") == "text" and item.get("text"):
            texts.append(item["text"])
    if texts:
        return {"_text": "\n".join(texts), "_raw_result": result}

    return {"_raw_result": result}


def normalize_doi(doi: str) -> str:
    doi = doi.strip()
    doi = doi.removeprefix("doi:").removeprefix("DOI:")
    return doi


QueryMethod = Literal["doi", "title"]


@dataclass(frozen=True)
class Query:
    key: str  # cache key, e.g. "DOI:10.1145/..."
    method: QueryMethod
    doi: str | None
    title: str | None


def build_query(citation: dict[str, Any]) -> Query | None:
    doi = citation.get("doi")
    title = citation.get("title")

    if isinstance(doi, str) and doi.strip():
        nd = normalize_doi(doi)
        return Query(key=f"DOI:{nd}", method="doi", doi=nd, title=title if isinstance(title, str) else None)

    if isinstance(title, str) and title.strip():
        # For title lookups, use a normalized key to avoid duplicates.
        t = " ".join(title.split())
        return Query(key=f"TITLE:{t.lower()}", method="title", doi=None, title=t)

    return None


def slim_paper(paper: dict[str, Any]) -> dict[str, Any]:
    """
    Keep only the fields the user asked for, plus a few identifiers.

    Note: Asta docs list fields like abstract/authors/year/url/isOpenAccess/etc.
    """
    out: dict[str, Any] = {}
    for k in [
        "title",
        "abstract",
        "authors",
        "year",
        "venue",
        "journal",
        "publicationDate",
        "url",
        "isOpenAccess",
        "fieldsOfStudy",
        "tldr",
    ]:
        if k in paper:
            out[k] = paper.get(k)
    # Keep any obvious IDs if present (not guaranteed by docs).
    # User preference: prefer corpusId over paperId.
    for k in ["corpusId", "externalIds"]:
        if k in paper:
            out[k] = paper.get(k)
    return out


def parse_fields_csv(fields: str) -> set[str]:
    # Normalize a comma-separated fields string into a set of lowercase names.
    # e.g. "title, authors, corpusId" -> {"title","authors","corpusid"}
    parts = []
    for p in (fields or "").split(","):
        p = p.strip()
        if p:
            parts.append(p.lower())
    return set(parts)


def cache_entry_missing_required_ids(entry: Any, *, want_corpus_id: bool) -> bool:
    """
    Decide whether a cached entry should be re-fetched because it lacks IDs
    the caller expects (e.g., corpusId).
    """
    if not want_corpus_id:
        return False
    if not isinstance(entry, dict) or entry.get("status") != "ok":
        return False
    paper = entry.get("paper")
    if not isinstance(paper, dict):
        return True
    # If corpusId is absent or null/empty, treat as missing.
    cid = paper.get("corpusId")
    if cid is None:
        return True
    if isinstance(cid, str) and not cid.strip():
        return True
    return False


def sanitize_cache_entries(entries: Any) -> None:
    """
    In-place cleanup of cached entries to match current output expectations.
    - Drop paperId (we prefer corpusId).
    """
    if not isinstance(entries, dict):
        return
    for v in entries.values():
        if not isinstance(v, dict):
            continue
        paper = v.get("paper")
        if isinstance(paper, dict) and "paperId" in paper:
            paper.pop("paperId", None)


def _clean_tei_text(s: str) -> str:
    # Remove nested tags and unescape entities.
    s = re.sub(r"<[^>]+>", " ", s)
    s = html.unescape(s)
    return " ".join(s.split()).strip()


def sniff_target_from_tei(tei_path: Path, *, max_bytes: int = 250_000) -> tuple[str | None, str | None]:
    """
    Best-effort extract of (title, doi) from a GROBID TEI XML file.
    Uses a lightweight regex scan over the first max_bytes for speed.
    """
    try:
        txt = tei_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return None, None

    head = txt[: max(0, int(max_bytes))]

    # DOI is commonly: <idno type="DOI">10....</idno>
    doi = None
    m = re.search(
        r"<(?:\w+:)?idno\b[^>]*\btype\s*=\s*['\"]doi['\"][^>]*>(.*?)</(?:\w+:)?idno>",
        head,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if m:
        doi = _clean_tei_text(m.group(1))

    # Main title is commonly under titleStmt.
    title = None
    m = re.search(
        r"<(?:\w+:)?titleStmt\b[^>]*>.*?<(?:\w+:)?title\b[^>]*>(.*?)</(?:\w+:)?title>",
        head,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if m:
        title = _clean_tei_text(m.group(1))
    else:
        # Fallback: pick the first <title ... type="main"> if present.
        m2 = re.search(
            r"<(?:\w+:)?title\b[^>]*\btype\s*=\s*['\"]main['\"][^>]*>(.*?)</(?:\w+:)?title>",
            head,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if m2:
            title = _clean_tei_text(m2.group(1))

    if doi and not doi.strip():
        doi = None
    if title and not title.strip():
        title = None
    return title, doi


def build_target_query_from_paths(citations_path: Path) -> Query | None:
    """
    Build a Query for the *source/target paper* (the paper whose citations file we are processing),
    primarily to resolve its corpusId for the output meta.

    Preference order:
    1) DOI found in matching TEI file (fast scan)
    2) Title found in matching TEI file
    3) Title derived from the citations filename stem
    """
    tei_path = citations_path.with_name(citations_path.name.replace(".citations.json", ".tei.xml"))
    tei_title: str | None = None
    tei_doi: str | None = None
    if tei_path.exists():
        tei_title, tei_doi = sniff_target_from_tei(tei_path)

    if isinstance(tei_doi, str) and tei_doi.strip():
        nd = normalize_doi(tei_doi)
        return Query(key=f"DOI:{nd}", method="doi", doi=nd, title=tei_title if isinstance(tei_title, str) else None)

    if isinstance(tei_title, str) and tei_title.strip():
        t = " ".join(tei_title.split())
        return Query(key=f"TITLE:{t.lower()}", method="title", doi=None, title=t)

    # Final fallback: filename-derived title
    base = citations_path.stem
    if base.endswith(".citations"):
        base = base[: -len(".citations")]
    base = " ".join(base.split()).strip()
    if base:
        return Query(key=f"TITLE:{base.lower()}", method="title", doi=None, title=base)

    return None


def open_access_url(paper: dict[str, Any]) -> str | None:
    """
    Asta's documented fields don't include openAccessPdf. Use `url` when isOpenAccess is true.
    If the API returns additional fields (e.g. openAccessPdf), prefer those.
    """
    oap = paper.get("openAccessPdf")
    if isinstance(oap, dict) and isinstance(oap.get("url"), str) and oap["url"].strip():
        return oap["url"].strip()
    if paper.get("isOpenAccess") and isinstance(paper.get("url"), str):
        return paper["url"]
    return None


def fetch_one_query(
    *,
    client: McpHttpClient,
    limiter: RateLimiter,
    query: Query,
    fields: str,
    timeout_s: int,
    max_retries: int,
    base_backoff_s: float,
) -> tuple[bool, dict[str, Any]]:
    last_err: str | None = None

    for attempt in range(1, max_retries + 1):
        try:
            limiter.acquire()
            if query.method == "doi":
                result = client.tools_call("get_paper", {"paper_id": f"DOI:{query.doi}", "fields": fields})
                parsed = parse_tool_result(result)
                return True, parsed
            else:
                result = client.tools_call("search_paper_by_title", {"title": query.title, "fields": fields})
                parsed = parse_tool_result(result)
                return True, parsed
        except Exception as e:
            last_err = str(e)
            if attempt < max_retries:
                jitter = random.random() * 0.25
                time.sleep(base_backoff_s * (2 ** (attempt - 1)) + jitter)

    return False, {"error": last_err or "Unknown error"}


def iter_citations_files(input_dir: Path) -> list[Path]:
    return sorted(p for p in input_dir.glob("*.citations.json") if p.is_file())


def augment_directory(
    *,
    input_dir: Path,
    output_dir: Path,
    endpoint: str,
    api_key: str,
    rps: float,
    concurrency: int,
    timeout_s: int,
    fields: str,
    max_retries: int,
    base_backoff_s: float,
    overwrite: bool,
    retry_cache_errors: bool,
    save_every: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    cache_path = output_dir / "s2_cache.json"
    progress_path = output_dir / "s2_augment_progress.json"

    cache: dict[str, Any] = {"meta": {"created_at": utc_now_iso(), "updated_at": utc_now_iso()}, "entries": {}}
    if cache_path.exists():
        cache = read_json(cache_path)
        cache.setdefault("meta", {})
        cache.setdefault("entries", {})

    # Keep cache compatible with current preferences (e.g., no paperId).
    sanitize_cache_entries(cache.get("entries"))

    progress: dict[str, Any] = {
        "meta": {
            "created_at": utc_now_iso(),
            "updated_at": utc_now_iso(),
            "input_dir": str(input_dir),
            "output_dir": str(output_dir),
            "endpoint": endpoint,
            "fields": fields,
        },
        "files": {},
    }
    if progress_path.exists():
        progress = read_json(progress_path)
        progress.setdefault("meta", {})
        progress.setdefault("files", {})

    citations_files = iter_citations_files(input_dir)
    if not citations_files:
        print(f"No .citations.json files found in {input_dir}")
        return

    # Map each file -> citations list + planned output
    file_payloads: dict[str, dict[str, Any]] = {}
    all_queries: dict[str, Query] = {}
    wanted_fields = parse_fields_csv(fields)
    want_corpus_id = "corpusid" in wanted_fields

    for path in citations_files:
        data = read_json(path)
        citations = data.get("citations") if isinstance(data, dict) else None
        if not isinstance(citations, list):
            continue

        out_path = output_dir / (path.stem.replace(".citations", "") + ".citations.s2.json")
        rel_key = str(path.relative_to(input_dir))

        file_payloads[rel_key] = {
            "path": path,
            "out_path": out_path,
            "data": data,
        }

        # Also resolve the *source/target* paper's corpusId for output meta.
        target_q = build_target_query_from_paths(path)
        if target_q:
            file_payloads[rel_key]["target_query_key"] = target_q.key
            file_payloads[rel_key]["target_query_method"] = target_q.method
            file_payloads[rel_key]["target_query_doi"] = target_q.doi
            file_payloads[rel_key]["target_query_title"] = target_q.title
            all_queries[target_q.key] = target_q
        else:
            file_payloads[rel_key]["target_query_key"] = None

        for c in citations:
            if not isinstance(c, dict):
                continue
            q = build_query(c)
            if q:
                all_queries[q.key] = q

    if not file_payloads:
        print("No valid citations files found (missing 'citations' list).")
        return

    # Determine which queries need fetching.
    missing: list[Query] = []
    entries = cache["entries"]
    for key, q in all_queries.items():
        existing = entries.get(key)
        if existing is None:
            missing.append(q)
            continue
        if cache_entry_missing_required_ids(existing, want_corpus_id=want_corpus_id):
            missing.append(q)
            continue
        if retry_cache_errors and isinstance(existing, dict) and existing.get("status") == "error":
            missing.append(q)

    print(f"Found {len(file_payloads)} citations files.")
    print(f"Unique citation lookups: {len(all_queries)} (missing: {len(missing)})")
    print(f"Endpoint: {endpoint}")
    print(f"Concurrency: {concurrency}, RPS limit: {rps}")
    print(f"Cache: {cache_path}")
    print(f"Progress: {progress_path}")

    client = McpHttpClient(endpoint=endpoint, api_key=api_key, timeout_s=timeout_s)
    limiter = RateLimiter(rps=rps)

    lock = threading.Lock()
    completed = 0

    def save_state() -> None:
        cache["meta"]["updated_at"] = utc_now_iso()
        progress["meta"]["updated_at"] = utc_now_iso()
        atomic_write_json(cache_path, cache)
        atomic_write_json(progress_path, progress)

    # Fetch missing queries with concurrency + global RPS limiter.
    if missing:
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = {
                executor.submit(
                    fetch_one_query,
                    client=client,
                    limiter=limiter,
                    query=q,
                    fields=fields,
                    timeout_s=timeout_s,
                    max_retries=max_retries,
                    base_backoff_s=base_backoff_s,
                ): q
                for q in missing
            }

            pbar = tqdm(as_completed(futures), total=len(futures), desc="Fetching S2 metadata (Asta)")
            for fut in pbar:
                q = futures[fut]
                ok, payload = fut.result()

                should_save = False
                with lock:
                    if ok:
                        entries[q.key] = {
                            "status": "ok",
                            "fetched_at": utc_now_iso(),
                            "method": q.method,
                            "paper": slim_paper(payload),
                            "open_access_url": open_access_url(payload),
                        }
                    else:
                        prev = entries.get(q.key) if isinstance(entries.get(q.key), dict) else {}
                        attempts = int((prev or {}).get("attempts", 0) or 0) + 1
                        entries[q.key] = {
                            "status": "error",
                            "fetched_at": utc_now_iso(),
                            "method": q.method,
                            "attempts": attempts,
                            "error": payload.get("error") if isinstance(payload, dict) else str(payload),
                        }

                    completed += 1
                    if completed % max(1, save_every) == 0:
                        should_save = True

                # Avoid deadlock: save_state does not acquire `lock` but keep pattern consistent.
                if should_save:
                    save_state()

            pbar.close()

        save_state()

    # Build enriched per-paper outputs.
    enriched_done = 0
    for rel_key, fp in file_payloads.items():
        src_path: Path = fp["path"]
        out_path: Path = fp["out_path"]

        if out_path.exists() and not overwrite:
            # Mark as done for progress (resume-friendly).
            progress["files"][rel_key] = {
                "status": "done",
                "source": str(src_path),
                "output": str(out_path),
                "updated_at": utc_now_iso(),
                "reason": "output exists",
            }
            continue

        data = fp["data"]
        citations = data.get("citations", [])
        enriched: list[dict[str, Any]] = []

        for c in citations:
            if not isinstance(c, dict):
                continue
            q = build_query(c)
            s2 = None
            if q and q.key in entries:
                s2 = entries[q.key]

            item = dict(c)
            item["s2"] = s2
            enriched.append(item)

        out_obj = {
            "meta": {
                "source": str(src_path),
                "generated_at": utc_now_iso(),
                "endpoint": endpoint,
                "fields": fields,
            },
            "n_citations": data.get("n_citations", len(enriched)),
            "citations": enriched,
        }

        # Attach resolved corpusId for the *target/source paper* (best-effort).
        target_key = fp.get("target_query_key")
        target_corpus_id = None
        if isinstance(target_key, str) and target_key and target_key in entries:
            e = entries.get(target_key)
            if isinstance(e, dict) and e.get("status") == "ok":
                p = e.get("paper")
                if isinstance(p, dict):
                    target_corpus_id = p.get("corpusId")
        out_obj["meta"]["target_query_key"] = target_key
        out_obj["meta"]["target_corpusId"] = target_corpus_id
        out_obj["meta"]["target_query_method"] = fp.get("target_query_method")
        out_obj["meta"]["target_query_doi"] = fp.get("target_query_doi")
        out_obj["meta"]["target_query_title"] = fp.get("target_query_title")

        try:
            atomic_write_json(out_path, out_obj)
            progress["files"][rel_key] = {
                "status": "done",
                "source": str(src_path),
                "output": str(out_path),
                "updated_at": utc_now_iso(),
            }
            enriched_done += 1
        except Exception as e:
            progress["files"][rel_key] = {
                "status": "failed",
                "source": str(src_path),
                "output": str(out_path),
                "updated_at": utc_now_iso(),
                "error": str(e),
            }

    save_state()

    done_files = sum(1 for v in progress["files"].values() if v.get("status") == "done")
    failed_files = sum(1 for v in progress["files"].values() if v.get("status") == "failed")
    print("\n" + "=" * 60)
    print("Complete!")
    print(f"Files done: {done_files}")
    print(f"Files failed: {failed_files}")
    print(f"Cache: {cache_path}")
    print(f"Progress: {progress_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Augment GROBID citations using S2 Asta MCP API.")
    parser.add_argument("--input", default="data/chi22_grobid", help="Directory containing *.citations.json")
    parser.add_argument(
        "--output",
        default="data/chi22_grobid_s2",
        help="Directory to write outputs",
    )
    parser.add_argument(
        "--endpoint",
        default=ASTA_MCP_ENDPOINT_DEFAULT,
        help="Asta MCP endpoint (default: https://asta-tools.allen.ai/mcp/v1)",
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("ASTA_API_KEY") or os.getenv("S2_API_KEY") or "",
        help="Asta API key (or set ASTA_API_KEY env var)",
    )
    parser.add_argument("--rps", type=float, default=8.0, help="Global requests-per-second limit (default 10)")
    parser.add_argument("--concurrency", type=int, default=8, help="Worker threads (default 8)")
    parser.add_argument("--timeout", type=int, default=60, help="HTTP timeout seconds (default 60)")
    parser.add_argument(
        "--fields",
        default="title,abstract,authors,year,venue,journal,publicationDate,url,isOpenAccess,fieldsOfStudy,tldr,corpusId,externalIds",
        help="Comma-separated fields to request from Asta",
    )
    parser.add_argument("--retries", type=int, default=3, help="Retries per lookup (default 3)")
    parser.add_argument("--base-backoff", type=float, default=1.0, help="Base backoff seconds for retries")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing *.citations.s2.json outputs")
    parser.add_argument(
        "--retry-cache-errors",
        action="store_true",
        help="Retry lookups already in cache with status=error",
    )
    parser.add_argument("--save-every", type=int, default=50, help="Save cache/progress every N lookups (default 50)")

    args = parser.parse_args()

    if not args.api_key:
        raise SystemExit(
            "Missing API key. Provide --api-key or set ASTA_API_KEY in your environment/.env."
        )

    augment_directory(
        input_dir=Path(args.input),
        output_dir=Path(args.output),
        endpoint=args.endpoint,
        api_key=args.api_key,
        rps=max(0.1, args.rps),
        concurrency=max(1, args.concurrency),
        timeout_s=max(1, args.timeout),
        fields=args.fields,
        max_retries=max(1, args.retries),
        base_backoff_s=max(0.1, args.base_backoff),
        overwrite=bool(args.overwrite),
        retry_cache_errors=bool(args.retry_cache_errors),
        save_every=max(1, args.save_every),
    )

