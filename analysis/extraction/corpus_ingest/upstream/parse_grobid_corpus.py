"""
Multi-venue GROBID parsing orchestrator for the hci_v1 PDF corpus.

Parses every PDF under an input root organized as <input_root>/<venue_dir>/<corpusid>.pdf
(as produced by pdf_crawler.py) into mirrored TEI XML:
    <output_root>/<venue_dir>/<corpusid>.tei.xml
    <output_root>/<venue_dir>/<corpusid>.citations.json   (lightweight summary)

Designed for a long, interruption-tolerant run over ~10K-67K PDFs:

  * RESUME is filesystem-as-truth — a PDF is "done" iff its .tei.xml exists and is
    non-empty. Writes are atomic (tmp -> os.replace), so a SIGKILL mid-write leaves no
    half files and the next run simply re-does anything without a TEI. This is strictly
    more robust than a rewritten progress JSON (which can corrupt on a mid-write crash).
  * FAILURES are recorded in an APPEND-ONLY JSONL event log (never rewritten -> no
    corruption window). Its only job is to cap retries of permanently-bad PDFs: a paper
    with >= --max-attempts recorded failures is skipped unless --retry-failed.
  * One GLOBAL thread pool spans all venues so the GROBID server stays saturated across
    venue boundaries. SIGINT/SIGTERM trigger graceful shutdown (stop claiming new work,
    let in-flight finish, flush, exit 0).

Subcommands:
  run      parse the corpus (resumable; the long run)
  pilot    parse a small sample across venues, measure throughput, project a full ETA
  status   per-venue done/remaining/failed counts from the filesystem + event log

Examples:
  uv run python parse_grobid_corpus.py pilot --n 40 --concurrency 8
  uv run python parse_grobid_corpus.py run --concurrency 10
  uv run python parse_grobid_corpus.py run --venues a_chi,b_dis --concurrency 8
  uv run python parse_grobid_corpus.py status
"""

from __future__ import annotations

import argparse
import json
import signal
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from tqdm import tqdm

# Reuse the tested primitives from the single-folder parser; parse_grobid.py has no
# import-time side effects (everything is under functions / __main__).
from parse_grobid import (
    atomic_write_json,
    atomic_write_text,
    extract_citations_summary,
    post_to_grobid,
    utc_now_iso,
)

DEFAULT_INPUT_ROOT = Path("data/pdfs")
DEFAULT_OUTPUT_ROOT = Path("data/pdfs_grobid")
DEFAULT_GROBID_URL = "http://localhost:8070/api/processFulltextDocument"
EVENTS_NAME = "_grobid_events.jsonl"  # append-only failure/skip log at the output root


@dataclass(frozen=True)
class Job:
    key: str          # "<venue>/<corpusid>" — unique, used in the event log
    venue: str
    corpusid: str
    pdf_path: Path
    tei_path: Path
    citations_path: Path


# --- discovery + resume -------------------------------------------------------
def discover_venue_dirs(input_root: Path, venues: list[str] | None) -> list[Path]:
    """Subdirectories of input_root that contain at least one PDF (skips probe/empty dirs)."""
    dirs = []
    for d in sorted(p for p in input_root.iterdir() if p.is_dir()):
        if venues and d.name not in venues:
            continue
        if any(d.glob("*.pdf")):
            dirs.append(d)
    return dirs


def build_jobs(input_root: Path, output_root: Path, venues: list[str] | None) -> list[Job]:
    jobs: list[Job] = []
    for vdir in discover_venue_dirs(input_root, venues):
        venue = vdir.name
        out_vdir = output_root / venue
        for pdf in sorted(vdir.glob("*.pdf")):
            cid = pdf.stem
            jobs.append(
                Job(
                    key=f"{venue}/{cid}",
                    venue=venue,
                    corpusid=cid,
                    pdf_path=pdf,
                    tei_path=out_vdir / f"{cid}.tei.xml",
                    citations_path=out_vdir / f"{cid}.citations.json",
                )
            )
    return jobs


def is_done(job: Job) -> bool:
    """Filesystem-as-truth: a non-empty TEI output means the PDF is parsed."""
    return job.tei_path.exists() and job.tei_path.stat().st_size > 0


def load_failure_counts(events_path: Path) -> dict[str, int]:
    """Tally recorded failures per job key from the append-only event log.

    Tolerant of a torn final line (possible if a crash interrupted an append)."""
    counts: dict[str, int] = {}
    if not events_path.exists():
        return counts
    with events_path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue  # skip a torn/partial line
            if ev.get("event") == "fail" and ev.get("key"):
                counts[ev["key"]] = counts.get(ev["key"], 0) + 1
    return counts


class EventLog:
    """Append-only JSONL writer. Appends are serialized + flushed so a crash at worst
    loses the last (unflushed) line rather than corrupting earlier ones."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = path.open("a", encoding="utf-8")
        self._lock = threading.Lock()

    def write(self, **fields) -> None:
        fields.setdefault("ts", utc_now_iso())
        line = json.dumps(fields, ensure_ascii=False)
        with self._lock:
            self._fh.write(line + "\n")
            self._fh.flush()

    def close(self) -> None:
        with self._lock:
            self._fh.close()


# --- core run -----------------------------------------------------------------
def _process_one(
    job: Job,
    *,
    grobid_url: str,
    form_options: dict[str, str],
    timeout_s: int,
    max_retries: int,
    base_backoff_s: float,
    write_citations_json: bool,
    stop_event: threading.Event,
) -> dict:
    # If shutdown was requested before this task started, skip fast (don't claim work).
    if stop_event.is_set():
        return {"key": job.key, "status": "skipped"}

    t0 = time.monotonic()
    ok, result = post_to_grobid(
        grobid_url=grobid_url,
        pdf_path=job.pdf_path,
        form_options=form_options,
        timeout_s=timeout_s,
        max_retries=max_retries,
        base_backoff_s=base_backoff_s,
    )
    latency = time.monotonic() - t0

    if not ok:
        return {
            "key": job.key, "status": "fail", "latency_s": latency,
            "error": result.get("error"), "http_status": result.get("http_status"),
        }

    atomic_write_text(job.tei_path, result["tei_xml"])
    n_citations = None
    if write_citations_json:
        summary = extract_citations_summary(result["tei_xml"])
        atomic_write_json(job.citations_path, summary)
        n_citations = summary.get("n_citations")

    return {"key": job.key, "status": "done", "latency_s": latency, "n_citations": n_citations}


def execute(
    jobs: list[Job],
    *,
    grobid_url: str,
    form_options: dict[str, str],
    concurrency: int,
    timeout_s: int,
    max_retries: int,
    base_backoff_s: float,
    write_citations_json: bool,
    events: EventLog,
    desc: str,
) -> dict:
    """Run jobs through a single global pool with graceful shutdown. Returns timing stats."""
    stop_event = threading.Event()
    prev_handlers: dict[int, object] = {}

    def _handle(signum, _frame):
        if not stop_event.is_set():
            print(f"\n[signal {signum}] graceful shutdown: finishing in-flight, not claiming new work...")
            stop_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        prev_handlers[sig] = signal.getsignal(sig)
        signal.signal(sig, _handle)

    counters = {"done": 0, "fail": 0, "skipped": 0}
    latencies: list[float] = []
    lock = threading.Lock()
    wall_t0 = time.monotonic()

    try:
        with ThreadPoolExecutor(max_workers=concurrency) as ex:
            futures = [
                ex.submit(
                    _process_one, job,
                    grobid_url=grobid_url, form_options=form_options, timeout_s=timeout_s,
                    max_retries=max_retries, base_backoff_s=base_backoff_s,
                    write_citations_json=write_citations_json, stop_event=stop_event,
                )
                for job in jobs
            ]
            pbar = tqdm(as_completed(futures), total=len(futures), desc=desc)
            for fut in pbar:
                res = fut.result()
                status = res["status"]
                if status == "fail":
                    events.write(event="fail", key=res["key"],
                                 http_status=res.get("http_status"), error=res.get("error"))
                with lock:
                    counters[status] += 1
                    if "latency_s" in res and status == "done":
                        latencies.append(res["latency_s"])
                    pbar.set_postfix(counters)
            pbar.close()
    finally:
        for sig, h in prev_handlers.items():
            signal.signal(sig, h)  # restore handlers

    wall = time.monotonic() - wall_t0
    n_done = counters["done"]
    return {
        "counters": counters,
        "wall_s": wall,
        "throughput_pdf_per_s": (n_done / wall) if wall > 0 and n_done else 0.0,
        "mean_latency_s": (sum(latencies) / len(latencies)) if latencies else 0.0,
        "interrupted": stop_event.is_set(),
    }


def form_options_from_args(args) -> dict[str, str]:
    return {
        "consolidateHeader": "1" if args.consolidate_header else "0",
        "consolidateCitations": "1" if args.consolidate_citations else "0",
        "includeRawCitations": "1",
        "segmentSentences": "1",
    }


def partition_jobs(jobs: list[Job], failure_counts: dict[str, int], max_attempts: int,
                   retry_failed: bool) -> tuple[list[Job], int, int]:
    """Split into (to_run, n_done, n_capped)."""
    todo, n_done, n_capped = [], 0, 0
    for j in jobs:
        if is_done(j):
            n_done += 1
            continue
        if not retry_failed and failure_counts.get(j.key, 0) >= max_attempts:
            n_capped += 1
            continue
        todo.append(j)
    return todo, n_done, n_capped


def sample_across_venues(jobs: list[Job], n: int) -> list[Job]:
    """Round-robin pick across venues so a pilot reflects the corpus mix, not one venue."""
    by_venue: dict[str, list[Job]] = {}
    for j in jobs:
        by_venue.setdefault(j.venue, []).append(j)
    order = sorted(by_venue)
    picked: list[Job] = []
    idx = 0
    while len(picked) < n and any(by_venue[v] for v in order):
        v = order[idx % len(order)]
        if by_venue[v]:
            picked.append(by_venue[v].pop(0))
        idx += 1
    return picked


def _fmt_dur(seconds: float) -> str:
    s = int(seconds)
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m"
    if m:
        return f"{m}m{sec:02d}s"
    return f"{sec}s"


# --- subcommands --------------------------------------------------------------
def cmd_status(args) -> None:
    input_root, output_root = Path(args.input_root), Path(args.output_root)
    venues = [v.strip() for v in args.venues.split(",")] if args.venues else None
    jobs = build_jobs(input_root, output_root, venues)
    failure_counts = load_failure_counts(output_root / EVENTS_NAME)

    per_venue: dict[str, dict[str, int]] = {}
    for j in jobs:
        d = per_venue.setdefault(j.venue, {"total": 0, "done": 0, "failed": 0})
        d["total"] += 1
        if is_done(j):
            d["done"] += 1
        elif failure_counts.get(j.key, 0):
            d["failed"] += 1

    tot = {"total": 0, "done": 0, "failed": 0}
    print(f"{'venue':<22}{'done':>8}{'total':>8}{'failed':>8}")
    print("-" * 46)
    for v in sorted(per_venue):
        d = per_venue[v]
        for k in tot:
            tot[k] += d[k]
        print(f"{v:<22}{d['done']:>8}{d['total']:>8}{d['failed']:>8}")
    print("-" * 46)
    print(f"{'TOTAL':<22}{tot['done']:>8}{tot['total']:>8}{tot['failed']:>8}")
    remaining = tot["total"] - tot["done"]
    print(f"\nremaining: {remaining}  ({100*tot['done']/tot['total']:.1f}% done)" if tot["total"] else "no PDFs found")


def cmd_pilot(args) -> None:
    input_root, output_root = Path(args.input_root), Path(args.output_root)
    venues = [v.strip() for v in args.venues.split(",")] if args.venues else None
    jobs = build_jobs(input_root, output_root, venues)
    failure_counts = load_failure_counts(output_root / EVENTS_NAME)
    todo, n_done, n_capped = partition_jobs(jobs, failure_counts, args.max_attempts, args.retry_failed)
    total_remaining = len(todo)

    sample = sample_across_venues(todo, args.n)
    if not sample:
        print("Nothing to pilot — all discovered PDFs already parsed (or capped).")
        return

    form_options = form_options_from_args(args)
    print(f"PILOT: {len(sample)} PDFs @ concurrency={args.concurrency} | "
          f"consolidateHeader={form_options['consolidateHeader']} "
          f"consolidateCitations={form_options['consolidateCitations']}")
    print(f"corpus: {len(jobs)} total | {n_done} already done | {total_remaining} remaining"
          + (f" | {n_capped} capped" if n_capped else ""))

    events = EventLog(output_root / EVENTS_NAME)
    try:
        stats = execute(
            sample, grobid_url=args.grobid_url, form_options=form_options,
            concurrency=args.concurrency, timeout_s=args.timeout, max_retries=args.retries,
            base_backoff_s=args.base_backoff, write_citations_json=not args.no_citations_json,
            events=events, desc="pilot",
        )
    finally:
        events.close()

    c = stats["counters"]
    tput = stats["throughput_pdf_per_s"]
    print("\n" + "=" * 60)
    print(f"PILOT RESULT: done={c['done']} fail={c['fail']} skipped={c['skipped']} "
          f"in {_fmt_dur(stats['wall_s'])} ({stats['wall_s']:.1f}s)")
    print(f"  mean server latency/pdf : {stats['mean_latency_s']:.1f}s")
    print(f"  throughput @ c={args.concurrency:<3}     : {tput:.2f} pdf/s "
          f"({tput*3600:.0f} pdf/hr)" if tput else "  throughput: n/a (no successes)")
    if tput > 0:
        eta = total_remaining / tput
        print(f"  projected for {total_remaining} remaining: ~{_fmt_dur(eta)} at this concurrency")
        print(f"  (full corpus is {len(jobs)} PDFs; {n_done} already done)")
    if form_options["consolidateCitations"] == "1":
        print("\n  NOTE: consolidateCitations=1 makes per-reference CrossRef/biblio-glutton")
        print("  lookups — usually the dominant cost. Re-run with --consolidate-citations 0")
        print("  to compare; the separate augment_citations_s2.py pass resolves refs against S2 later.")


def cmd_run(args) -> None:
    input_root, output_root = Path(args.input_root), Path(args.output_root)
    venues = [v.strip() for v in args.venues.split(",")] if args.venues else None
    output_root.mkdir(parents=True, exist_ok=True)

    jobs = build_jobs(input_root, output_root, venues)
    if not jobs:
        raise SystemExit(f"No PDFs found under {input_root}" + (f" for venues={venues}" if venues else ""))
    failure_counts = load_failure_counts(output_root / EVENTS_NAME)
    todo, n_done, n_capped = partition_jobs(jobs, failure_counts, args.max_attempts, args.retry_failed)
    if args.max is not None:
        todo = todo[: args.max]

    form_options = form_options_from_args(args)
    print(f"input : {input_root}\noutput: {output_root}\nGROBID: {args.grobid_url}")
    print(f"venues: {len(set(j.venue for j in jobs))} | PDFs total: {len(jobs)} | "
          f"already done: {n_done} | to run: {len(todo)}"
          + (f" | capped (>= {args.max_attempts} fails): {n_capped}" if n_capped else ""))
    print(f"concurrency: {args.concurrency} | consolidateHeader={form_options['consolidateHeader']} "
          f"consolidateCitations={form_options['consolidateCitations']}")
    if not todo:
        print("Nothing to do — all PDFs already parsed (or capped). Use --retry-failed to re-attempt capped.")
        return

    events = EventLog(output_root / EVENTS_NAME)
    try:
        stats = execute(
            todo, grobid_url=args.grobid_url, form_options=form_options,
            concurrency=args.concurrency, timeout_s=args.timeout, max_retries=args.retries,
            base_backoff_s=args.base_backoff, write_citations_json=not args.no_citations_json,
            events=events, desc="grobid",
        )
    finally:
        events.close()

    c = stats["counters"]
    print("\n" + "=" * 60)
    print(("Interrupted (graceful) " if stats["interrupted"] else "Complete ")
          + f"in {_fmt_dur(stats['wall_s'])}")
    print(f"done={c['done']} fail={c['fail']} skipped={c['skipped']} | "
          f"throughput {stats['throughput_pdf_per_s']:.2f} pdf/s")
    print(f"failures logged to: {output_root / EVENTS_NAME}")
    print("Re-run the same command to resume (finished TEIs are skipped automatically).")


# --- CLI ----------------------------------------------------------------------
def add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--input-root", default=str(DEFAULT_INPUT_ROOT),
                   help="Root containing <venue>/<corpusid>.pdf dirs")
    p.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT),
                   help="Root for mirrored <venue>/<corpusid>.tei.xml output")
    p.add_argument("--venues", default=None, help="Comma-separated venue dir names (default: all)")


def add_grobid(p: argparse.ArgumentParser) -> None:
    p.add_argument("--grobid-url", default=DEFAULT_GROBID_URL)
    # Matched to the server's default `concurrency: 10`. Measured optimum on the -full
    # (deep-learning) image here: throughput plateaus at c=10 (~0.71 pdf/s); pushing higher
    # only adds queueing latency + 503 churn (the server pool, not the client, is the ceiling).
    # Patient 503 retries mop up transient saturation. Raise this only if you raise the server's
    # own concurrency (and have GPU/CPU headroom for it).
    p.add_argument("--concurrency", type=int, default=10, help="Concurrent GROBID requests")
    p.add_argument("--timeout", type=int, default=300, help="Per-request timeout seconds")
    p.add_argument("--retries", type=int, default=6,
                   help="In-call retries (HTTP 503/5xx/timeout) with exponential backoff")
    p.add_argument("--base-backoff", type=float, default=2.0)
    p.add_argument("--max-attempts", type=int, default=3,
                   help="Skip a PDF after this many recorded failures (unless --retry-failed)")
    p.add_argument("--retry-failed", action="store_true", help="Re-attempt previously-capped failures")
    p.add_argument("--no-citations-json", action="store_true",
                   help="Do not write <corpusid>.citations.json summaries")
    # Consolidation toggles. Default OFF: both are redundant for this pipeline (the
    # inventory already supplies authoritative header metadata per corpusid, and
    # augment_citations_s2.py resolves references against S2 later) AND consolidateCitations=1
    # makes per-reference CrossRef lookups that drop throughput ~20x (measured ~41s/pdf vs ~2s).
    # Set to 1 only if you specifically want GROBID-side CrossRef enrichment.
    p.add_argument("--consolidate-header", type=int, choices=(0, 1), default=0)
    p.add_argument("--consolidate-citations", type=int, choices=(0, 1), default=0)


def main() -> None:
    ap = argparse.ArgumentParser(description="Multi-venue GROBID parsing orchestrator (resumable).")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser("run", help="Parse the corpus (resumable long run)")
    add_common(p_run); add_grobid(p_run)
    p_run.add_argument("--max", type=int, default=None, help="Cap number of PDFs this run (debug)")
    p_run.set_defaults(func=cmd_run)

    p_pilot = sub.add_parser("pilot", help="Sample N PDFs, measure throughput, project ETA")
    add_common(p_pilot); add_grobid(p_pilot)
    p_pilot.add_argument("--n", type=int, default=40, help="Number of PDFs to sample")
    p_pilot.set_defaults(func=cmd_pilot)

    p_status = sub.add_parser("status", help="Per-venue done/remaining/failed from filesystem")
    add_common(p_status)
    p_status.set_defaults(func=cmd_status)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
