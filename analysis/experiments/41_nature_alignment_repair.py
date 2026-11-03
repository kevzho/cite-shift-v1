#!/usr/bin/env python3
"""Exploratory Nature reference alignment diagnostic; never overwrites registered audit.

Candidate evidence comes from the complete PDF-derived TEI bibliography, rather
than only bibliography entries with an extracted citance. Original mapped pairs
are frozen. New pairs require exact normalized title or exact title containment
in raw reference plus author/year support, and must be unique in both directions.
Candidate-only recovery rates are development diagnostics, not fresh validation.
"""
from __future__ import annotations

import csv
import hashlib
import json
import random
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from xml.etree import ElementTree as ET

REPO = Path(__file__).resolve().parents[2]
ARM = REPO / "external_data/nature_arm_2026-10-04"
OUT = REPO / "external_data/nature_repair_2026-10-25"
NS = {"t": "http://www.tei-c.org/ns/1.0"}
XMLID = "{http://www.w3.org/XML/1998/namespace}id"


def norm(s):
    s = unicodedata.normalize("NFKD", s or "")
    return re.sub(r"[^a-z0-9]", "", s.lower())


def tokens(s):
    return set(re.findall(r"[a-z]{3,}", (s or "").lower()))


def text(el):
    return " ".join(el.itertext()).strip() if el is not None else ""


def records(path):
    with path.open() as f:
        return [json.loads(line) for line in f]


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def run():
    OUT.mkdir(exist_ok=True)
    if (OUT / "diagnostic.json").exists():
        raise SystemExit("Output exists; preserve it and choose a new dated directory.")
    audit = list(csv.DictReader((ARM / "parity/parity_gate_edges.csv").open()))
    jats = {(e["citing_paper_id"], e["cited_ref_id"]): e
            for e in records(ARM / "parity/jats_citances_200.jsonl")}
    grob = {(e["citing_paper_id"], e["cited_ref_id"]): e
            for e in records(ARM / "citances/nature_arm_citances.jsonl")}
    audit_pids = sorted({r["pmcid"] for r in audit})
    original_targets = defaultdict(list)
    for r in audit:
        if r["located"] != "none":
            original_targets[(r["pmcid"], r["grobid_ref"])].append(r["jats_ref"])
    tei_files = {p.name.removesuffix(".tei.xml"): p
                 for p in (ARM / "pdfs_grobid").rglob("*.tei.xml")}
    bibs, jmeta, body_counts = {}, {}, {}
    for pid in audit_pids:
        jt = ET.parse(ARM / "jats" / f"{pid}.xml")
        jmeta[pid] = {}
        for ref in jt.findall(".//ref-list/ref"):
            surname = text(ref.find(".//surname"))
            year = text(ref.find(".//year"))
            jmeta[pid][ref.get("id")] = {"surname": surname, "year": year,
                "raw": text(ref), "label": text(ref.find("label"))}
        tt = ET.parse(tei_files[pid])
        counts = Counter()
        for el in tt.findall(".//t:ref[@type='bibr']", NS):
            for target in (el.get("target") or "").split():
                counts[target.removeprefix("#")] += 1
        body_counts[pid] = counts
        bb = []
        for ref in tt.findall(".//t:listBibl/t:biblStruct", NS):
            titles = [text(x) for x in ref.findall(".//t:title", NS) if text(x)]
            raw = text(ref.find("t:note[@type='raw_reference']", NS))
            surname = text(ref.find(".//t:surname", NS))
            date = ref.find(".//t:date", NS)
            year = (date.get("when") or text(date))[:4] if date is not None else ""
            bb.append({"id": ref.get(XMLID), "titles": titles, "raw": raw,
                       "surname": surname, "year": year})
        bibs[pid] = bb
    inspections = []
    proposals = []
    for row in audit:
        if row["located"] != "none":
            continue
        pid, rid = row["pmcid"], row["jats_ref"]
        e = jats[(pid, rid)]
        jm = jmeta[pid].get(rid, {})
        title = norm(e["cited_title"])
        possible = []
        for b in bibs[pid]:
            raw = norm(b["raw"])
            exact = len(title) >= 12 and title in {norm(t) for t in b["titles"]}
            contained = len(title) >= 30 and title in raw
            author = bool(jm.get("surname")) and norm(jm["surname"]) == norm(b["surname"])
            year = bool(jm.get("year")) and jm["year"] == b["year"]
            # Fallback repairs field-boundary errors without relaxing fuzzy-title threshold.
            supported_raw = contained and author and year
            if exact or supported_raw:
                possible.append((b, "exact_title" if exact else "raw_title_author_year"))
        status = "no_conservative_bibliography_candidate"
        b = None
        rule = ""
        if len(possible) > 1:
            status = "ambiguous_multiple_bibliography_candidates"
        elif possible:
            b, rule = possible[0]
            if original_targets.get((pid, b["id"])):
                status = "candidate_conflicts_with_original_pair"
            elif (pid, b["id"]) not in grob:
                status = "bibliography_present_no_extracted_edge"
            else:
                status = "new_alignment_candidate"
                proposals.append((row, b, rule))
        inspections.append({"pmcid": pid, "jats_ref": rid, "status": status,
            "jats_title": e["cited_title"], "jats_doi": e["cited_doi"],
            "jats_author": jm.get("surname", ""), "jats_year": jm.get("year", ""),
            "jats_label": jm.get("label", ""), "jats_raw": jm.get("raw", ""),
            "candidate_count": len(possible), "grobid_ref": b["id"] if b else "",
            "rule": rule, "grobid_titles": " | ".join(b["titles"]) if b else "",
            "grobid_raw": b["raw"] if b else "",
            "tei_marker_count": body_counts[pid][b["id"]] if b else "",
            "candidate_has_extracted_edge": int((pid,b["id"]) in grob) if b else "",
            "original_pair_conflicts": " | ".join(original_targets.get((pid,b["id"]), [])) if b else "",
            "candidate_refs": " | ".join(x["id"] for x, _ in possible),
            "manual_match_correct": "", "manual_notes": ""})
    proposed_targets = Counter((r["pmcid"], b["id"]) for r, b, _ in proposals)
    accepted = []
    for r, b, rule in proposals:
        if proposed_targets[(r["pmcid"], b["id"])] != 1:
            for x in inspections:
                if (x["pmcid"], x["jats_ref"]) == (r["pmcid"], r["jats_ref"]):
                    x["status"] = "candidate_conflicts_with_another_new_pair"
            continue
        pid, rid = r["pmcid"], r["jats_ref"]
        j, g = jats[(pid, rid)], grob[(pid, b["id"])]
        rec = [any(len(tokens(c) & tokens(x)) / max(len(tokens(c)), 1) >= .8
                   for x in g["citances"]) for c in j["citances"]]
        accepted.append({"pmcid": pid, "jats_ref": rid, "grobid_ref": b["id"],
                         "rule": rule, "n_jats_citances": len(rec),
                         "n_recovered": sum(rec), "fully_matched": int(all(rec)),
                         "requires_validation": True})
    n_c = sum(int(r["n_jats_citances"]) for r in audit)
    rec_c = sum(int(r["n_recovered"]) for r in audit)
    full = sum(int(r["fully_matched"]) for r in audit)
    status_counts = Counter(x["status"] for x in inspections)
    missing_edge_marker_counts = Counter(
        "tei_has_markers" if x["tei_marker_count"] else "tei_no_markers"
        for x in inspections if x["status"] == "bibliography_present_no_extracted_edge")
    original_collisions = [{"pmcid": p, "grobid_ref": g, "jats_refs": " | ".join(js)}
                           for (p,g),js in original_targets.items() if len(js)>1]
    meta = {"status": "exploratory development diagnostic; NOT fresh validation",
        "registered_RQ4b": "original failed gate and stopping rule remain unchanged",
        "audited_papers": len(audit_pids), "original_unlocated_edges": len(inspections),
        "diagnostic_categories": status_counts,
        "bibliography_present_no_edge_marker_diagnosis": missing_edge_marker_counts,
        "original_nonunique_target_groups": len(original_collisions),
        "unique_new_alignment_candidates": len(accepted),
        "original_citance_recovery": rec_c/n_c, "original_edge_match": full/len(audit),
        "candidate_only_citance_recovery_unvalidated": (rec_c+sum(x["n_recovered"] for x in accepted))/n_c,
        "candidate_only_edge_match_unvalidated": (full+sum(x["fully_matched"] for x in accepted))/len(audit),
        "new_candidate_recovered_citances": sum(x["n_recovered"] for x in accepted),
        "new_candidate_fully_matched_edges": sum(x["fully_matched"] for x in accepted),
        "source_sha256": {str(p.relative_to(REPO)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [ARM/'parity/parity_gate_edges.csv', ARM/'parity/jats_citances_200.jsonl',
                      ARM/'citances/nature_arm_citances.jsonl', Path(__file__)]}}
    write_csv(OUT/'unmatched_reference_inspection.csv', inspections)
    write_csv(OUT/'unique_new_alignment_candidates.csv', accepted)
    write_csv(OUT/'original_pair_collisions.csv', original_collisions)
    # Sample labels for manual inspection are intentionally blank.
    rng = random.Random(20261025)
    review = []
    for status in sorted(status_counts):
        pool = [x for x in inspections if x['status']==status]
        review.extend(rng.sample(pool,min(20,len(pool))))
    write_csv(OUT/'stratified_development_review.csv',review)
    manifest = list(csv.DictReader((ARM/'frame/manifest.csv').open()))
    holdout = [r for r in manifest if r['status'].startswith('ok') and r['pmcid'] not in set(audit_pids)]
    byj = defaultdict(list)
    for r in holdout:
        byj[r['journal']].append(r)
    # Plan a 50-paper fresh diagnostic with coverage of each available journal.
    selected = []
    for j in sorted(byj):
        selected.extend(rng.sample(sorted(byj[j],key=lambda x:x['pmcid']), min(3,len(byj[j]))))
    used = {x['pmcid'] for x in selected}
    remaining = [x for x in holdout if x['pmcid'] not in used]
    selected.extend(rng.sample(sorted(remaining,key=lambda x:x['pmcid']),50-len(selected)))
    write_csv(OUT/'fresh_50_paper_validation_plan.csv',[
        {"pmcid":r['pmcid'],"journal":r['journal'],"year":r['year'],"doi":r['doi'],
         "status":"planned_only_not_downloaded_or_evaluated"} for r in selected])
    meta['fresh_validation_plan'] = {"papers":50,"seed":20261025,
        "sampling":"up to 3 per journal then random remainder; descriptive, not population-weighted",
        "excluded_all_original_200_papers": True,
        "caveat":"freeze method before fetching/evaluating these papers; do not retune on this holdout"}
    (OUT/'diagnostic.json').write_text(json.dumps(meta,indent=2)+'\n')
    report = ["# Exploratory Nature alignment diagnostic — October 25, 2026", "",
        "This is a development diagnostic on the already examined 200 papers. New candidates are not manually or independently validated. The original RQ4(b) failed gate remains unchanged.", "",
        "## What was inspected", "", f"All {len(inspections):,} originally unlocated references were compared with complete TEI bibliographies, not only citance-bearing extracted edges.", ""]
    report.extend(f"- {k}: {v:,}" for k,v in status_counts.items())
    report += ["", "## Tentative alignment-only effect", "",
        f"- Unique new pairs with existing extracted contexts: {len(accepted)}.",
        f"- Original recovery: {rec_c/n_c:.4%}; candidate-only recovery: {meta['candidate_only_citance_recovery_unvalidated']:.4%} (unvalidated).",
        f"- Original complete-edge match: {full/len(audit):.4%}; candidate-only complete-edge match: {meta['candidate_only_edge_match_unvalidated']:.4%} (unvalidated).",
        f"- Original mappings contain {len(original_collisions)} nonunique target groups; these were reported, not silently changed.", "",
        "Bibliography entries with no extracted edge are not counted as recovered contexts. Some can exist in the bibliography without a recognized body citation, so matching improvements alone cannot restore them.", "",
        "## Next required checks", "",
        "1. Review stratified_development_review.csv against XML and TEI/PDF; enter judgments rather than treating candidate rules as proof.",
        "2. Inspect missing-edge cases with and without TEI markers to identify parser linking versus downstream extraction loss.",
        "3. Freeze a revised method, then retrieve publisher XML for the preselected 50 fresh papers. The file is a plan only; no fresh validation has run.",
        "4. Check false matches and one-to-one mapping as well as recovery. Preserve failures and ambiguous cases.",
        "5. Marker handling is a separate issue: this script changes no classifier text or marker exclusions. No new MultiCite uptake statistic is produced.",
        "6. Any follow-up field comparison remains explicitly exploratory; do not relabel the original registered arm as passed.", ""]
    (OUT/'README.md').write_text('\n'.join(report))
    print(json.dumps(meta,indent=2))


if __name__ == '__main__':
    run()
