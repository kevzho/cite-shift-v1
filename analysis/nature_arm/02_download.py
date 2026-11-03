#!/usr/bin/env python3
"""Download the Nature arm PDFs (all 1,200) and the 200-paper JATS parity subsample.

Route: PMC Cloud Service (AWS Open Data bucket `pmc-oa-opendata`), as recommended by the 27 Sep pilot.
Per candidate, in frame order (primary ranks first, then reserve, within each journal x year cell):
  1. list `metadata/<PMCID>.` and fetch the newest version's metadata JSON (cached in logs/meta/)
  2. require is_pmc_openaccess, not retracted, a pdf_url
  3. fetch the PDF, verify the %PDF header and the md5 in pdf_url, save pdfs/<PMCID>.pdf
A cell stops once `quota` PDFs succeed; a failed primary is replaced by the next reserve in that cell.
Resume-safe: an existing valid pdfs/<PMCID>.pdf (md5 match) and cached metadata are reused.

Parity subsample: 200 of the downloaded papers, stratified by journal (largest remainder), seed
20261005; their JATS from the metadata xml_url is saved to jats/<PMCID>.xml (must contain <article).
Writes frame/manifest.csv and frame/licence_counts.csv. Sends only PMCIDs.
"""
import csv, hashlib, json, math, random, re, sys, time
from collections import Counter, defaultdict
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).parent))
from common import ROOT, S3, get, utc

PARITY_N, PARITY_SEED = 200, 20261005
META = ROOT / "logs/meta"


def s3_http(u):
    return (u or "").split("?")[0].replace("s3://pmc-oa-opendata/", S3 + "/")


def md5_of(u):
    m = re.search(r"md5=([0-9a-f]{32})", u or "")
    return m.group(1) if m else None


def metadata(pmcid):
    cache = META / f"{pmcid}.json"
    if cache.exists():
        return json.loads(cache.read_text())
    s, b, _ = get(f"{S3}/?list-type=2&prefix=metadata/{pmcid}.&max-keys=50", delay=0.2)
    keys = re.findall(r"<Key>(metadata/[^<]+\.json)</Key>", b.decode("utf-8", "replace")) if s == 200 else []
    if not keys:
        return {"_error": f"no_metadata(status={s})"}
    keys.sort(key=lambda k: int(re.search(r"\.(\d+)\.json$", k).group(1)))
    s, b, _ = get(f"{S3}/{keys[-1]}", delay=0.2)
    if s != 200:
        return {"_error": f"metadata_fetch_{s}"}
    md = json.loads(b)
    md["_key"] = keys[-1]
    cache.write_text(json.dumps(md))
    return md


def fetch_pdf(pmcid, md):
    p = ROOT / "pdfs" / f"{pmcid}.pdf"
    want = md5_of(md.get("pdf_url"))
    if p.exists():
        data = p.read_bytes()
        if data[:4] == b"%PDF" and (not want or hashlib.md5(data).hexdigest() == want):
            return data, "ok_cached"
    s, data, _ = get(s3_http(md.get("pdf_url")), delay=0.2, timeout=300)
    if s != 200:
        return None, f"pdf_http_{s}"
    if data[:4] != b"%PDF":
        return None, "pdf_bad_header"
    if want and hashlib.md5(data).hexdigest() != want:
        return None, "pdf_md5_mismatch"
    p.write_bytes(data)
    return data, "ok"


frame = pd.read_csv(ROOT / "frame/frame.csv")
alloc = pd.read_csv(ROOT / "frame/allocation_1200.csv")
quota = {(r.journal, r.year): r.quota for r in alloc.itertuples()}
man, t0 = [], time.time()
for (j, y), cell in frame.sort_values(["year", "journal", "rank"]).groupby(["journal", "year"], sort=False):
    got = 0
    for r in cell.sort_values("rank").itertuples():
        rec = {"pmcid": r.pmcid, "doi": r.doi, "journal": j, "year": y, "rank": r.rank, "role": r.role,
               "license": "", "pdf_path": "", "pdf_sha256": "", "pdf_bytes": "", "jats_path": "",
               "jats_sha256": "", "status": "", "retrieved_utc": ""}
        if got >= quota[(j, y)]:
            rec["status"] = "unused_reserve"
            man.append(rec)
            continue
        md = metadata(r.pmcid)
        rec["license"] = md.get("license_code") or ""
        if "_error" in md:
            rec["status"] = md["_error"]
        elif not md.get("is_pmc_openaccess"):
            rec["status"] = "not_pmc_openaccess"
        elif md.get("is_retracted"):
            rec["status"] = "retracted"
        elif not md.get("pdf_url"):
            rec["status"] = "no_pdf_url"
        else:
            data, st = fetch_pdf(r.pmcid, md)
            rec["status"] = st
            if data is not None:
                rec.update(pdf_path=f"pdfs/{r.pmcid}.pdf", pdf_sha256=hashlib.sha256(data).hexdigest(),
                           pdf_bytes=len(data), retrieved_utc=utc())
                got += 1
        man.append(rec)
    print(f"{utc()} {j:10s} {y} {got}/{quota[(j, y)]}  elapsed {time.time()-t0:.0f}s", flush=True)

# parity subsample, stratified by journal among downloaded papers
ok = [m for m in man if m["status"].startswith("ok")]
byj = defaultdict(list)
for m in ok:
    byj[m["journal"]].append(m)
exact = {jn: PARITY_N * len(v) / len(ok) for jn, v in byj.items()}
take = {jn: int(math.floor(e)) for jn, e in exact.items()}
for jn in sorted(exact, key=lambda k: (-(exact[k] - take[k]), k))[:PARITY_N - sum(take.values())]:
    take[jn] += 1
rng = random.Random(PARITY_SEED)
for jn in sorted(byj):
    for m in rng.sample(sorted(byj[jn], key=lambda x: x["pmcid"]), take[jn]):
        md = metadata(m["pmcid"])
        s, b, _ = get(s3_http(md.get("xml_url")), delay=0.2)
        if s == 200 and b"<article" in b[:3000]:
            (ROOT / "jats" / f"{m['pmcid']}.xml").write_bytes(b)
            m["jats_path"], m["jats_sha256"] = f"jats/{m['pmcid']}.xml", hashlib.sha256(b).hexdigest()
        else:
            m["jats_path"] = f"FAILED_{s}"

pd.DataFrame(man).to_csv(ROOT / "frame/manifest.csv", index=False)
lic = Counter(m["license"] or "(none)" for m in ok)
pd.DataFrame(sorted(lic.items(), key=lambda x: -x[1]), columns=["license", "n"]).to_csv(
    ROOT / "frame/licence_counts.csv", index=False)
print(json.dumps({"finished_utc": utc(), "seconds": round(time.time() - t0),
                  "downloaded": len(ok), "statuses": Counter(m["status"] for m in man),
                  "parity_jats_ok": sum(bool(m["jats_path"]) and not m["jats_path"].startswith("FAILED") for m in ok),
                  "licences": lic}, indent=1, default=str))
