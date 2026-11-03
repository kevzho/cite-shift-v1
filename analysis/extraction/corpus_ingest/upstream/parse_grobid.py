"""
Parse a folder of paper PDFs with a running GROBID server.

Example (roughly equivalent to):
  curl -v \
    --form input=@./data/chi22_pdfs/thefile.pdf \
    --form consolidateHeader=1 \
    --form consolidateCitations=1 \
    --form includeRawCitations=1 \
    --form segmentSentences=1 \
    http://localhost:8070/api/processFulltextDocument

Usage:
  uv run python parse_grobid.py
  uv run python parse_grobid.py --input data/chi22_pdfs --output data/chi22_grobid --concurrency 4

Notes:
- Concurrency is configurable via --concurrency.
- Resume/failure handling is via a progress JSON file in the output directory.
"""

from __future__ import annotations

import json
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

import requests
from tqdm import tqdm


TEI_NS = {"tei": "http://www.tei-c.org/ns/1.0"}


def _is_permanent_grobid_error(status: int | None, body: str) -> bool:
    """A non-200 that retrying will never fix, so we should stop early instead of
    burning the backoff budget. 4xx are client/content errors; HTTP 500 with GROBID's
    '[NO_BLOCKS] ... empty content' means the PDF has no extractable text (scanned/
    corrupt). Everything else (503 busy, 502/504, generic 500, 429) is treated as
    transient and retried."""
    if status is None:
        return False
    if 400 <= status < 500:
        return True
    if status == 500 and ("NO_BLOCKS" in body or "empty content" in body):
        return True
    return False


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp_path.replace(path)


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(text, encoding="utf-8")
    tmp_path.replace(path)


def list_pdfs(input_dir: Path) -> list[Path]:
    pdfs = sorted(p for p in input_dir.glob("*.pdf") if p.is_file())
    return pdfs


def extract_citations_summary(tei_xml: str) -> dict[str, Any]:
    """
    Extract a lightweight citations summary from GROBID TEI XML.

    This intentionally keeps output small: count + a list of basic fields.
    """
    try:
        root = ET.fromstring(tei_xml)
    except ET.ParseError:
        return {"n_citations": 0, "citations": [], "parse_error": True}

    bibl_structs = root.findall(".//tei:listBibl/tei:biblStruct", TEI_NS)
    citations: list[dict[str, Any]] = []

    for bs in bibl_structs:
        title_el = bs.find(".//tei:analytic/tei:title[@level='a']", TEI_NS) or bs.find(
            ".//tei:monogr/tei:title", TEI_NS
        )
        title = (title_el.text or "").strip() if title_el is not None else ""

        doi_el = bs.find(".//tei:idno[@type='DOI']", TEI_NS)
        doi = (doi_el.text or "").strip() if doi_el is not None else ""

        year_el = bs.find(".//tei:imprint/tei:date[@type='published']", TEI_NS) or bs.find(
            ".//tei:imprint/tei:date", TEI_NS
        )
        year = (year_el.get("when") or (year_el.text or "")).strip() if year_el is not None else ""

        citations.append(
            {
                "title": title or None,
                "doi": doi or None,
                "year": year or None,
            }
        )

    return {"n_citations": len(citations), "citations": citations}


@dataclass(frozen=True)
class Job:
    pdf_path: Path
    rel_key: str
    tei_path: Path
    citations_path: Path


def post_to_grobid(
    *,
    grobid_url: str,
    pdf_path: Path,
    form_options: dict[str, str],
    timeout_s: int,
    max_retries: int,
    base_backoff_s: float,
) -> tuple[bool, dict[str, Any]]:
    """
    Returns (success, result).
    On success, result contains {"tei_xml": str}.
    On failure, result contains {"error": str, "http_status": int|None}.
    """
    last_http_status: int | None = None
    last_err: str | None = None

    for attempt_idx in range(1, max_retries + 1):
        try:
            with pdf_path.open("rb") as f:
                resp = requests.post(
                    grobid_url,
                    files={"input": (pdf_path.name, f, "application/pdf")},
                    data=form_options,
                    timeout=timeout_s,
                )
            last_http_status = resp.status_code

            if resp.status_code == 200 and resp.text.strip():
                return True, {"tei_xml": resp.text, "http_status": resp.status_code}

            # Non-200: retryable in many cases (503 busy, 5xx, 429) but not for permanent
            # content errors (4xx, 500 NO_BLOCKS) — those just waste the backoff budget.
            last_err = f"HTTP {resp.status_code}: {resp.text[:500].strip()}"
            if _is_permanent_grobid_error(resp.status_code, resp.text):
                return False, {"error": last_err, "http_status": resp.status_code}
        except requests.exceptions.Timeout:
            last_err = "Request timeout"
        except requests.exceptions.ConnectionError as e:
            last_err = f"Connection error: {e}"
        except Exception as e:
            last_err = f"Unexpected error: {e}"

        if attempt_idx < max_retries:
            jitter = random.random() * 0.25
            sleep_s = base_backoff_s * (2 ** (attempt_idx - 1)) + jitter
            time.sleep(sleep_s)

    return False, {"error": last_err or "Unknown error", "http_status": last_http_status}


def run(
    *,
    input_dir: Path,
    output_dir: Path,
    progress_path: Path,
    grobid_url: str,
    concurrency: int,
    timeout_s: int,
    max_retries: int,
    base_backoff_s: float,
    write_citations_json: bool,
    retry_failed: bool,
    max_attempts_per_file: int,
    start_index: int,
    max_files: int | None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    progress: dict[str, Any] = {
        "meta": {
            "created_at": utc_now_iso(),
            "updated_at": utc_now_iso(),
            "input_dir": str(input_dir),
            "output_dir": str(output_dir),
            "grobid_url": grobid_url,
        },
        "files": {},
    }
    if progress_path.exists():
        try:
            progress = json.loads(progress_path.read_text(encoding="utf-8"))
            progress.setdefault("meta", {})
            progress.setdefault("files", {})
        except Exception:
            # If progress file is corrupt, do not overwrite it automatically.
            raise RuntimeError(f"Failed to read progress file: {progress_path}")

    form_options = {
        "consolidateHeader": "1",
        "consolidateCitations": "1",
        "includeRawCitations": "1",
        "segmentSentences": "1",
    }
    progress["meta"]["grobid_form_options"] = form_options

    pdfs = list_pdfs(input_dir)
    if start_index:
        pdfs = pdfs[start_index:]
    if max_files is not None:
        pdfs = pdfs[:max_files]

    jobs: list[Job] = []
    for pdf_path in pdfs:
        rel_key = str(pdf_path.relative_to(input_dir))
        stem = pdf_path.stem
        tei_path = output_dir / f"{stem}.tei.xml"
        citations_path = output_dir / f"{stem}.citations.json"
        jobs.append(Job(pdf_path=pdf_path, rel_key=rel_key, tei_path=tei_path, citations_path=citations_path))

    lock = threading.Lock()

    def should_process(job: Job) -> bool:
        # If output already exists, treat as done (and update progress later).
        if job.tei_path.exists() and job.tei_path.stat().st_size > 0:
            return False

        info = progress["files"].get(job.rel_key)
        if not info:
            return True

        status = info.get("status")
        attempts = int(info.get("attempts", 0) or 0)
        if status == "done":
            return False
        if status == "failed" and not retry_failed:
            return False
        if attempts >= max_attempts_per_file:
            return False
        return True

    jobs_to_run = [j for j in jobs if should_process(j)]

    # Opportunistically mark already-existing outputs as done (resume safety)
    should_save = False
    with lock:
        for j in jobs:
            if j.tei_path.exists() and j.tei_path.stat().st_size > 0:
                info = progress["files"].get(j.rel_key, {})
                info.update(
                    {
                        "pdf_path": str(j.pdf_path),
                        "status": "done",
                        "output_tei_path": str(j.tei_path),
                        "updated_at": utc_now_iso(),
                    }
                )
                progress["files"][j.rel_key] = info
                should_save = True
        progress["meta"]["updated_at"] = utc_now_iso()

    if should_save:
        atomic_write_json(progress_path, progress)

    if not jobs_to_run:
        print("Nothing to parse (all TEI outputs already exist or are marked done).")
        return

    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Progress file: {progress_path}")
    print(f"GROBID URL: {grobid_url}")
    print(f"Files total: {len(jobs)}; to process: {len(jobs_to_run)}")
    print(f"Concurrency: {concurrency}")

    def process_one(job: Job) -> dict[str, Any]:
        ok, result = post_to_grobid(
            grobid_url=grobid_url,
            pdf_path=job.pdf_path,
            form_options=form_options,
            timeout_s=timeout_s,
            max_retries=max_retries,
            base_backoff_s=base_backoff_s,
        )
        if not ok:
            return {
                "rel_key": job.rel_key,
                "pdf_path": str(job.pdf_path),
                "success": False,
                "error": result.get("error"),
                "http_status": result.get("http_status"),
            }

        tei_xml: str = result["tei_xml"]
        atomic_write_text(job.tei_path, tei_xml)

        citations_summary: dict[str, Any] | None = None
        if write_citations_json:
            citations_summary = extract_citations_summary(tei_xml)
            atomic_write_json(job.citations_path, citations_summary)

        return {
            "rel_key": job.rel_key,
            "pdf_path": str(job.pdf_path),
            "success": True,
            "http_status": result.get("http_status"),
            "output_tei_path": str(job.tei_path),
            "output_citations_path": str(job.citations_path) if write_citations_json else None,
            "n_citations": (citations_summary or {}).get("n_citations") if citations_summary else None,
        }

    completed_count = 0
    try:
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = {executor.submit(process_one, job): job for job in jobs_to_run}
            pbar = tqdm(as_completed(futures), total=len(futures), desc="Parsing PDFs (GROBID)")

            for future in pbar:
                res = future.result()
                rel_key = res["rel_key"]
                should_save = False

                with lock:
                    info = progress["files"].get(rel_key, {})
                    attempts = int(info.get("attempts", 0) or 0) + 1
                    info.update(
                        {
                            "pdf_path": res.get("pdf_path"),
                            "attempts": attempts,
                            "last_http_status": res.get("http_status"),
                            "updated_at": utc_now_iso(),
                        }
                    )

                    if res["success"]:
                        info.update(
                            {
                                "status": "done",
                                "last_error": None,
                                "output_tei_path": res.get("output_tei_path"),
                                "output_citations_path": res.get("output_citations_path"),
                                "n_citations": res.get("n_citations"),
                            }
                        )
                    else:
                        info.update(
                            {
                                "status": "failed",
                                "last_error": res.get("error"),
                            }
                        )

                    progress["files"][rel_key] = info
                    progress["meta"]["updated_at"] = utc_now_iso()

                    completed_count += 1
                    if completed_count % 10 == 0:
                        should_save = True

                    done_count = sum(1 for v in progress["files"].values() if v.get("status") == "done")
                    failed_count = sum(1 for v in progress["files"].values() if v.get("status") == "failed")
                    pbar.set_postfix({"done": done_count, "failed": failed_count})

                # Avoid deadlock: atomic_write_json acquires no lock, but we keep the
                # pattern consistent with existing project memory guidance.
                if should_save:
                    atomic_write_json(progress_path, progress)

            pbar.close()
    except KeyboardInterrupt:
        print("\nInterrupted. Saving progress...")
    finally:
        atomic_write_json(progress_path, progress)

    done_count = sum(1 for v in progress["files"].values() if v.get("status") == "done")
    failed_count = sum(1 for v in progress["files"].values() if v.get("status") == "failed")
    print("\n" + "=" * 60)
    print("Complete!")
    print(f"Done: {done_count}")
    print(f"Failed: {failed_count}")
    print(f"Progress file: {progress_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Parse a folder of PDFs via GROBID (processFulltextDocument).")
    parser.add_argument("--input", default="data/chi22_pdfs", help="Folder containing PDF files")
    parser.add_argument("--output", default="data/chi22_grobid", help="Output folder for TEI XML and JSON")
    parser.add_argument(
        "--progress",
        default=None,
        help="Progress JSON path (default: <output>/grobid_progress.json)",
    )
    parser.add_argument(
        "--grobid-url",
        default="http://localhost:8070/api/processFulltextDocument",
        help="GROBID endpoint URL",
    )
    parser.add_argument("--concurrency", type=int, default=2, help="Number of concurrent requests")
    parser.add_argument("--timeout", type=int, default=180, help="Request timeout seconds")
    parser.add_argument("--retries", type=int, default=3, help="Retries per request (HTTP/timeout/connection)")
    parser.add_argument("--base-backoff", type=float, default=1.0, help="Base backoff seconds for retries")
    parser.add_argument(
        "--max-attempts-per-file",
        type=int,
        default=5,
        help="Max total attempts per file across runs (tracked in progress file)",
    )
    parser.add_argument(
        "--no-retry-failed",
        action="store_true",
        default=True,
        help="Do not retry files previously marked failed",
    )
    parser.add_argument(
        "--no-citations-json",
        action="store_true",
        help="Do not write <stem>.citations.json summaries",
    )
    parser.add_argument("--start", type=int, default=0, help="Start index into sorted PDFs list")
    parser.add_argument("--max", type=int, default=None, help="Max number of PDFs to process")

    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    progress_path = Path(args.progress) if args.progress else (output_dir / "grobid_progress.json")

    if not input_dir.exists():
        raise SystemExit(f"Input directory does not exist: {input_dir}")

    run(
        input_dir=input_dir,
        output_dir=output_dir,
        progress_path=progress_path,
        grobid_url=args.grobid_url,
        concurrency=max(1, args.concurrency),
        timeout_s=max(1, args.timeout),
        max_retries=max(1, args.retries),
        base_backoff_s=max(0.1, args.base_backoff),
        write_citations_json=not args.no_citations_json,
        retry_failed=not args.no_retry_failed,
        max_attempts_per_file=max(1, args.max_attempts_per_file),
        start_index=max(0, args.start),
        max_files=args.max,
    )
