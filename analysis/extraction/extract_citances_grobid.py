#!/usr/bin/env python3
"""Extract in-text citances from GROBID TEI and link them to bibliography entries.

DATA GENERATION ONLY. Produces the citance corpus; computes no study quantity.
Output schema mirrors the canonical corpus: one row per (citing paper, cited
reference) edge, carrying the list of sentences in which the reference is cited.

Usage:
    python3 extract_citances.py <tei_root> <out.jsonl> [--limit N]
"""
from __future__ import annotations
import json, re, sys
from pathlib import Path
import xml.etree.ElementTree as ET

TEI = "{http://www.tei-c.org/ns/1.0}"
XMLID = "{http://www.w3.org/XML/1998/namespace}id"


def text_of(el) -> str:
    return re.sub(r"\s+", " ", "".join(el.itertext())).strip()


def bibliography(root) -> dict[str, dict]:
    out = {}
    for bs in root.iter(f"{TEI}biblStruct"):
        bid = bs.get(XMLID)
        if not bid:
            continue
        title = next((text_of(t) for t in bs.iter(f"{TEI}title")
                      if t.get("level") in ("a", "m") and text_of(t)), "")
        doi = next((text_of(i) for i in bs.iter(f"{TEI}idno")
                    if (i.get("type") or "").upper() == "DOI"), "")
        year = ""
        for d in bs.iter(f"{TEI}date"):
            year = d.get("when", "")[:4] or text_of(d)[:4]
            if year:
                break
        out[bid] = {"title": title, "doi": doi, "year": year}
    return out


def extract(path: Path) -> list[dict]:
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError:
        return []
    bib = bibliography(root)
    body = root.find(f"{TEI}text/{TEI}body")
    if body is None:
        return []

    edges: dict[str, dict] = {}
    for div in body.iter(f"{TEI}div"):
        head = div.find(f"{TEI}head")
        section = text_of(head) if head is not None else ""
        for s in div.iter(f"{TEI}s"):
            refs = [r for r in s.iter(f"{TEI}ref")
                    if r.get("type") == "bibr" and r.get("target")]
            if not refs:
                continue
            sent = text_of(s)
            if len(sent) < 20:
                continue
            for r in refs:
                tid = r.get("target", "").lstrip("#")
                if tid not in bib:
                    continue
                e = edges.setdefault(tid, {
                    "cited_ref_id": tid, **bib[tid],
                    "citances": [], "sections": [],
                })
                if sent not in e["citances"]:
                    e["citances"].append(sent)
                    e["sections"].append(section)

    pid = path.name.replace(".tei.xml", "")
    venue = path.parent.name
    rows = []
    for tid, e in edges.items():
        rows.append({
            "citing_paper_id": pid,
            "venue": venue,
            "cited_ref_id": tid,
            "cited_title": e["title"],
            "cited_doi": e["doi"],
            "cited_year": e["year"],
            "n_citances": len(e["citances"]),
            "citances": e["citances"],
            "sections": e["sections"],
        })
    return rows


def main() -> None:
    root = Path(sys.argv[1])
    out = Path(sys.argv[2])
    limit = None
    if "--limit" in sys.argv:
        limit = int(sys.argv[sys.argv.index("--limit") + 1])

    files = sorted(root.rglob("*.tei.xml"))
    if limit:
        files = files[:limit]

    n_rows = n_cit = 0
    papers = 0
    with out.open("w", encoding="utf-8") as fh:
        for i, f in enumerate(files, 1):
            rows = extract(f)
            if rows:
                papers += 1
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
                n_rows += 1
                n_cit += r["n_citances"]
            if i % 500 == 0:
                print(f"  {i}/{len(files)} files | {n_rows:,} edges | {n_cit:,} citances",
                      flush=True)

    print(f"\nDONE  files={len(files)}  papers_with_edges={papers}")
    print(f"      edges={n_rows:,}  citances={n_cit:,}")
    print(f"      mean citances/edge={n_cit/max(n_rows,1):.2f}")
    print(f"      out={out}")


if __name__ == "__main__":
    main()