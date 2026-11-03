#!/usr/bin/env python3
"""Extract in-text citances from JATS XML and link them to ref-list entries.

JATS counterpart of ../../chi_chiea_grobid_2026-09-15/code/extract_citances.py
(the CHI / CHI EA GROBID-TEI extractor). DATA GENERATION ONLY: computes no study
quantity, reads no label.

Output: one JSON line per (citing paper, cited reference) edge with EXACTLY the
CHI fields, in the CHI order:

    citing_paper_id, venue, cited_ref_id, cited_title, cited_doi, cited_year,
    n_citances, citances, sections

Semantics kept from the CHI extractor
  * edge = citing paper x ref-list entry that is cited at least once in the body;
    edges are emitted in order of first citation;
  * citances = unique body sentences containing >= 1 in-text citation of that ref,
    in document order, marker text left in the sentence (the CHI side keeps "[12]");
  * sentences shorter than 20 characters (marker text included) are dropped;
  * sections[i] = heading of the innermost titled section holding citances[i]
    (GROBID emits flat <div>s, one per heading, so CHI sections are the deepest
    heading); "" when the paragraph has no titled ancestor;
  * body only: abstract (front) and back matter (acknowledgements, funding, data
    and code availability, author contributions, competing interests, appendices,
    supplementary, reference list) are excluded, as GROBID moves those out of
    <body> (div type=acknowledgement/availability/funding/annex in <back>);
  * figure and table captions, table cells and footnotes are excluded (GROBID puts
    them in <figure>/<note> outside the body divs, which the CHI extractor skips);
  * cited_title = analytic title (article/chapter) else monograph title (book,
    report, software; never a journal name) -- GROBID title level a, else m;
  * cited_doi / cited_year as strings, "" when absent.

Unavoidable differences (see PILOT_NOTES.md, section 7)
  * citing_paper_id is the PMCID (file stem). CHI uses the Semantic Scholar corpus
    id (also its file stem). The PMCID -> DOI / S2 corpus id map is written to the
    --papers sidecar, never into the edge file.
  * cited_ref_id is the JATS ref @id (e.g. "CR12"), CHI's is GROBID's "b11".
  * JATS has no sentence tags, so sentences are split by rule here (GROBID
    segments with its own model). Citation markers that follow the full stop stay
    with the preceding sentence, as GROBID does.
  * Superscript markers are rendered as their digits preceded by one space
    ("infection 4."), CHI markers are bracketed ("[4]").
  * A range written <xref>2</xref>-<xref>5</xref> is expanded to refs 2..5 by
    ref-list order (GROBID 0.9 also expands superscript ranges in Nature PDFs; on
    bracketed CHI ranges such as "[21-24, 45]" it emits an untargeted <ref>, which
    the CHI extractor drops).
  * Inline MathML is rendered as its text; TeX-only inline formulas as "[formula]".
  * Titles of unstructured references come from the Europe PMC google-scholar
    link, else from a rule-based parse of the citation string.

Usage:
    python3 extract_citances_jats.py <jats_dir> <out.jsonl> [--manifest manifest.csv]
        [--papers papers_sidecar.csv] [--frame pilot_frame.csv] [--limit N]

Stdlib only. Deterministic: files are processed in sorted order and no
set/dict iteration order depends on hashing.
"""
from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path
from urllib.parse import unquote
import xml.etree.ElementTree as ET

XLINK_HREF = "{http://www.w3.org/1999/xlink}href"
MML = "{http://www.w3.org/1998/Math/MathML}"
MARK = ""  # private-use char: position of one bibr <xref> in the working string

# Inline elements whose content is not running text.
SKIP_INLINE = {"fig", "fig-group", "table-wrap", "table-wrap-group", "table", "disp-formula",
               "disp-formula-group", "supplementary-material", "fn", "fn-group", "ref-list",
               "graphic", "media", "inline-graphic", "chem-struct-wrap", "array", "code",
               "preformat", "alternatives", "label"}
# Block elements that are never walked for paragraphs.
SKIP_BLOCK = SKIP_INLINE | {"title", "ack", "app", "app-group", "back", "glossary",
                            "notes", "sec-meta", "boxed-text", "caption", "statement"}
BACK_SEC_TYPES = {"ack", "fn-group", "ref-list", "supplementary-material", "associated-data",
                  "contrib-info", "app", "extended-data", "display-objects", "data-availability",
                  "COI-statement", "author-contributions", "funding", "supplementary"}
BACK_TITLE = re.compile(
    r"^\s*(acknowledg|author(s'|s’|s)? contributions?|contributions$|competing (financial )?interests?"
    r"|conflicts? of interest|declaration of (competing )?interests?|data availability"
    r"|code availability|data and code availability|code and data availability"
    r"|materials? availability|availability of data|peer review|funding|online content"
    r"|additional information|supplementary|supporting information|source data"
    r"|footnotes|references$|contributor information|extended data|reporting summary"
    r"|ethics declarations|associated data|electronic supplementary|change history"
    r"|rights and permissions|inclusion (and|&) ethics|abbreviations$|appendix)", re.I)

ABBR = {"al", "fig", "figs", "e.g", "i.e", "eg", "ie", "vs", "ref", "refs", "approx", "ca", "cf",
        "eq", "eqs", "nos", "suppl", "supp", "sect", "resp", "dr", "st", "mr", "ms", "prof",
        "inc", "ltd", "co", "jr", "sr", "vol", "pp", "ext", "sp", "spp", "var", "subsp", "tab",
        "chr", "no", "nr", "ed", "eds", "ser", "viz"}
DASH = "–—‒−-"
DASH_RE = "–—‒−\\-"   # for use inside a regex character class


def norm_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def text_of(el) -> str:
    return norm_ws("".join(el.itertext())) if el is not None else ""


# ----------------------------------------------------------------- reference list
DOI_RE = re.compile(r"\b(10\.\d{4,9}/[^\s\"<>]+)")
GS_KEYS = "author|journal|volume|issue|publication_year|pages|pmid|doi|publisher|isbn|title"
AUTH_NATURE = re.compile(
    r"^(?:.*?\bet al\.?|.*?&\s*[^,&]+?,\s*(?:[A-Z][a-z]?\.\s?-?\s?)+|[^,]+?,\s*(?:[A-Z][a-z]?\.\s?-?\s?)+)\s*")
AUTH_INIT_DOT = re.compile(
    r"^(?:.*?\bet al\.?|.*?&\s*[^,&.]+?\s(?:[A-Z][a-z]?\.\s?-?)+|[^,.]+?\s(?:[A-Z][a-z]?\.\s?-?)+)\s*")
AUTH_VANCOUVER = re.compile(r"^(?:.*?\bet al|[^.]*?\b[A-Z]{1,4})\.\s+")


def clean_doi(d: str) -> str:
    d = d.strip()
    d = re.sub(r"^(?:https?://)?(?:dx\.)?doi\.org/", "", d, flags=re.I)
    d = re.sub(r"^doi:\s*", "", d, flags=re.I)
    return d.rstrip(".,;")


def gs_param(href: str, key: str) -> str:
    m = re.search(r"(?:^|[?&])" + key + r"=(.*?)(?=&(?:" + GS_KEYS + r")=|&?$)", href)
    return unquote(m.group(1)).strip() if m else ""


def title_from_string(node) -> str:
    """Rule-based title for an unstructured reference (Nature or Vancouver style)."""
    pre, ital = [node.text or ""], ""
    for c in node:
        if c.tag == "italic":
            ital = text_of(c)
            break
        pre.append("".join(c.itertext()))
        pre.append(c.tail or "")
    s = norm_ws("".join(pre))
    s = re.sub(r"\s*doi:\s*\S+.*$", "", s, flags=re.I)
    auth = (AUTH_NATURE if re.match(r"[^,]+,\s*[A-Z][a-z]?\.", s)       # "Surname, I. & Surname, I."
            else AUTH_INIT_DOT if re.match(r"[^,.]+?\s[A-Z][a-z]?\.", s)  # "Surname I., Surname I."
            else None)
    m = (auth or AUTH_VANCOUVER).match(s)                  # else Vancouver: "Surname AB, ... ."
    rest = s[m.end():] if m else s
    rest = re.split(r"\.\s+In\s", rest)[0]
    title = re.split(r"(?<=[.?!])\s+(?=[A-Z])", rest, maxsplit=1)[0].rstrip(".").strip()
    if not title and ital:                                 # "Author. <italic>Book</italic> (Publ., yr)"
        title = ital
    return title


def ref_list(root) -> tuple[dict[str, dict], list[str]]:
    refs: dict[str, dict] = {}
    order: list[str] = []
    for ref in root.iter("ref"):           # Europe PMC may place <ref-list> in <body>
        rid = ref.get("id")
        if not rid or rid in refs:
            continue
        cit = ref.find("citation-alternatives/element-citation")
        if cit is None:
            cit = ref.find("element-citation")
        if cit is None:
            cit = ref.find("citation-alternatives/mixed-citation")
        if cit is None:
            cit = ref.find("mixed-citation")
        if cit is None:
            cit = ref
        ptype = (cit.get("publication-type") or "").lower()
        title = ""
        for tag in ("article-title", "chapter-title", "data-title", "part-title"):
            t = cit.find(f".//{tag}")
            if text_of(t):
                title = text_of(t)
                break
        if not title and ptype not in ("journal", "") and text_of(cit.find("source")):
            title = text_of(cit.find("source"))          # book / report / software: level m
        doi = next((text_of(p) for p in ref.iter("pub-id")
                    if (p.get("pub-id-type") or "").lower() == "doi"), "")
        gs = ""
        for e in ref.iter("ext-link"):
            kind, href = (e.get("ext-link-type") or "").lower(), e.get(XLINK_HREF) or ""
            if not doi and kind == "doi":
                doi = href or text_of(e)
            elif not doi and "doi.org/10." in href:
                doi = href
            elif kind == "google-scholar" and not gs:
                gs = href
        year = text_of(cit.find(".//year"))[:4]
        if gs:
            title = title or gs_param(gs, "title")
            year = year or gs_param(gs, "publication_year")[:4]
            doi = doi or gs_param(gs, "doi")
        cs = cit.find("named-content[@content-type='citation-string']")
        raw_node = cs if cs is not None else cit
        raw = text_of(raw_node)
        if not doi:
            m = DOI_RE.search(raw)
            doi = m.group(1) if m else ""
        if not title and cit.find(".//source") is None and cit.find(".//article-title") is None:
            title = title_from_string(raw_node)
        if not re.fullmatch(r"\d{4}", year or ""):
            bare = re.sub(r"(?:doi:\s*)?\b10\.\d{4,9}/\S+|https?://\S+", " ", raw, flags=re.I)
            ys = re.findall(r"\((1[5-9]\d\d|20\d\d)[a-z]?\)", bare) or \
                re.findall(r"\b(1[5-9]\d\d|20\d\d)[a-z]?\b", bare)
            year = ys[-1] if ys else ""
        refs[rid] = {"title": title, "doi": clean_doi(doi), "year": year}
        order.append(rid)
    return refs, order


# ------------------------------------------------------------- body paragraphs
def is_back_sec(sec) -> bool:
    if (sec.get("sec-type") or "") in BACK_SEC_TYPES:
        return True
    return bool(BACK_TITLE.match(text_of(sec.find("title"))))


def paragraphs(node, section: str):
    """Yield (section heading, <p>) in document order."""
    for c in node:
        tag = c.tag
        if tag == "sec":
            if is_back_sec(c):
                continue
            t = text_of(c.find("title"))
            yield from paragraphs(c, t or section)
        elif tag == "p":
            yield section, c
        elif isinstance(tag, str) and tag not in SKIP_BLOCK:
            yield from paragraphs(c, section)


def math_text(el) -> str:
    m = el.find(f".//{MML}math")
    return f" {norm_ws(''.join(m.itertext()))} " if m is not None else " [formula] "


def render(el, parts: list[str], rids: list[list[str]]) -> None:
    """Append the running text of element `el` (not its tail)."""
    tag = el.tag
    if not isinstance(tag, str) or tag in SKIP_INLINE:
        parts.append(" ")
        return
    if tag == "inline-formula":
        parts.append(math_text(el))
        return
    if tag == "xref" and el.get("ref-type") == "bibr":
        rids.append((el.get("rid") or "").split())
        prev = "".join(parts[-3:])
        if prev and not prev[-1].isspace() and prev[-1] not in ",;(" + DASH:
            parts.append(", " if prev[-1].isdigit() and prev.rfind(MARK) > len(prev) - 6 else " ")
        parts.append(MARK + text_of(el))
        return
    if el.text:
        parts.append(el.text)
    for c in el:
        render(c, parts, rids)
        if c.tail:
            parts.append(c.tail)


SPLIT = re.compile(
    r"([.!?])([”\"'’]*)"                                  # terminal punctuation (+ closing quote)
    r"((?:[\s,;" + DASH_RE + r"\[\]()]*" + MARK + r"[^\s.!?" + MARK + r"]*)*)"   # trailing markers
    r"(\s+(?![a-z,;:.)\]}%])|(?=[A-Z][a-z]))")               # next sentence starts: not lowercase ASCII


def sentences(s: str) -> list[tuple[int, int]]:
    """Rule-based sentence spans. Markers after the full stop stay left (as in GROBID)."""
    spans, start = [], 0
    for m in SPLIT.finditer(s):
        pre = s[start:m.start(1)]
        w = re.findall(r"([A-Za-z][A-Za-z.]*)$", pre)
        if w and w[0].lower().rstrip(".") in ABBR:
            continue
        if re.search(r"(?:^|[\s(])[A-Z]$", pre):           # initials: "K. pneumoniae", "J. Smith"
            continue
        if m.group(4) == "" and not m.group(3):            # "x.Y" with no space: only after a marker
            continue
        spans.append((start, m.end(3)))
        start = m.end(4) if m.group(4) else m.end(3)
    spans.append((start, len(s)))
    return spans


def paragraph_marks(s: str, rids: list[list[str]], refs: dict, order: list[str],
                    pos: dict[str, int]) -> list[tuple[int, list[str]]]:
    """(offset, [ref ids]) per MARK, with a-b ranges expanded by ref-list order."""
    marks: list[tuple[int, list[str], int]] = []
    for at, rl in zip([m.start() for m in re.finditer(MARK, s)], rids):
        ids = [r for r in rl if r in refs]
        if marks and ids and marks[-1][1]:
            if re.fullmatch(r"\s*[" + DASH_RE + r"]\s*", s[marks[-1][2]:at]):
                a, b = pos[marks[-1][1][-1]], pos[ids[0]]
                if b > a + 1:
                    ids = order[a + 1:b] + ids
        lab = re.match(r"[^\s" + MARK + r",;" + DASH_RE + r")\]]*", s[at + 1:]).group(0)
        marks.append((at, ids, at + 1 + len(lab)))
    return [(at, ids) for at, ids, _ in marks]


def extract(path: Path, venue: str = "") -> list[dict]:
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError:
        return []
    refs, order = ref_list(root)
    pos = {r: i for i, r in enumerate(order)}
    body = root.find("body")
    if body is None:
        return []
    edges: dict[str, dict] = {}
    for section, p in paragraphs(body, ""):
        parts, rids = [], []
        render(p, parts, rids)
        s = "".join(parts)
        if MARK not in s:
            continue
        s = norm_ws(s)
        marks = paragraph_marks(s, rids, refs, order, pos)
        for a0, a1 in sentences(s):
            inside = [ids for at, ids in marks if a0 <= at < a1]
            if not inside:
                continue
            sent = norm_ws(s[a0:a1].replace(MARK, ""))
            if len(sent) < 20:
                continue
            for ids in inside:
                for rid in ids:
                    e = edges.setdefault(rid, {"citances": [], "sections": []})
                    if sent not in e["citances"]:
                        e["citances"].append(sent)
                        e["sections"].append(section)
    pid = path.stem
    return [{
        "citing_paper_id": pid,
        "venue": venue,
        "cited_ref_id": rid,
        "cited_title": refs[rid]["title"],
        "cited_doi": refs[rid]["doi"],
        "cited_year": refs[rid]["year"],
        "n_citances": len(e["citances"]),
        "citances": e["citances"],
        "sections": e["sections"],
    } for rid, e in edges.items()]


def arg(name: str) -> str | None:
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else None


def main() -> None:
    src, out = Path(sys.argv[1]), Path(sys.argv[2])
    meta: dict[str, dict] = {}
    if arg("--manifest"):
        with open(arg("--manifest"), newline="") as fh:
            meta = {r["pmcid"]: r for r in csv.DictReader(fh)}
    s2: dict[str, str] = {}
    if arg("--frame"):
        with open(arg("--frame"), newline="") as fh:
            for r in csv.DictReader(fh):
                if r.get("citing_corpusid"):
                    s2[r["doi"].lower()] = str(int(float(r["citing_corpusid"])))
    files = sorted(src.glob("PMC*.xml"))
    if arg("--limit"):
        files = files[:int(arg("--limit"))]
    n_rows = n_cit = papers = 0
    side = []
    with out.open("w", encoding="utf-8") as fh:
        for f in files:
            m = meta.get(f.stem, {})
            rows = extract(f, m.get("venue_key", ""))
            papers += bool(rows)
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
                n_rows += 1
                n_cit += r["n_citances"]
            side.append({"citing_paper_id": f.stem, "citing_doi": m.get("doi", ""),
                         "s2_corpusid": s2.get(m.get("doi", "").lower(), ""),
                         "venue": m.get("venue_key", ""), "year": m.get("year", ""),
                         "stratum": m.get("stratum", ""), "licence": m.get("licence", ""),
                         "n_edges": len(rows)})
    if arg("--papers"):
        with open(arg("--papers"), "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(side[0]) if side else ["citing_paper_id"])
            w.writeheader()
            w.writerows(side)
    print(f"DONE  files={len(files)}  papers_with_edges={papers}")
    print(f"      edges={n_rows:,}  citances={n_cit:,}")
    print(f"      mean citances/edge={n_cit / max(n_rows, 1):.2f}")
    print(f"      out={out}")


if __name__ == "__main__":
    main()
