"""
SLR Phase 1 — Automated Paper Fetcher
Domain: Deep Learning for Angiographic Sequence Processing
Sources: PubMed, Semantic Scholar, arXiv
Output:  slr_stage1_screening.csv  (ready for title/abstract screening)

Usage:
    # Run Semantic Scholar only (default):
    python slr_fetch.py

    # Run all sources (PubMed + Semantic Scholar + arXiv):
    python slr_fetch.py all

    pip install biopython requests

Optional: set your NCBI API key as an env var for higher PubMed rate limits:
    export NCBI_API_KEY="your_key_here"
"""

import csv
import time
import os
import re
import sys
import json
import requests
from Bio import Entrez

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

ENTREZ_EMAIL   = "nikhan@ucdavis.edu"
NCBI_API_KEY   = os.getenv("NCBI_API_KEY", "b62cdd0f606eecea1e4ec3fb9b348b8b9809")
OUTPUT_FILE    = "/data/Deep_Angiography/AngioVision/slr/results/slr_stage1_screening.csv"
STATS_FILE     = "/data/Deep_Angiography/AngioVision/slr/results/slr_fetching_stats.csv"
MAX_PER_SOURCE = 2000

# CLI: pass "ss" to run Semantic Scholar only; default runs all sources
RUN_ALL = not (len(sys.argv) > 1 and sys.argv[1].strip().lower() == "ss")

# ── Queries ────────────────────────────────────────────────────────────────

PUBMED_QUERY = (
    "(angiograph*[tiab] OR fluoroscop*[tiab] OR "
    "\"digital subtraction angiography\"[tiab] OR DSA[tiab]) "
    "AND (sequence*[tiab] OR video[tiab] OR frame*[tiab] OR "
    "time-series[tiab] OR temporal[tiab]) "
    "AND (processing[tiab] OR enhancement[tiab] OR denois*[tiab] OR "
    "subtraction[tiab] OR registration[tiab] OR motion[tiab] OR "
    "segmentation[tiab] OR detection[tiab] OR classification[tiab] OR "
    "localization[tiab] OR diagnosis[tiab]) "
    "AND (\"deep learning\"[tiab] OR \"machine learning\"[tiab] OR "
    "CNN[tiab] OR transformer[tiab] OR \"artificial intelligence\"[tiab] OR AI[tiab]) "
    "AND (\"2000\"[pdat] : \"3000\"[pdat]) "
    "AND english[lang]"
)

SS_QUERIES = [
    "deep learning digital subtraction angiography sequence",
    "CNN fluoroscopy image enhancement temporal",
    "transformer angiographic vessel segmentation",
    "machine learning DSA motion correction",
    "deep learning fluoroscopic sequence processing",
    "artificial intelligence angiography diagnosis detection",
    "deep learning DSA sequence classification localization",
]

# arXiv query disabled — arXiv results excluded from this SLR
# ARXIV_QUERY = (
#     "ti_abs:(angiograph* OR fluoroscop* OR DSA) AND "
#     "ti_abs:(deep learning OR CNN OR transformer OR machine learning OR artificial intelligence OR AI) AND "
#     "ti_abs:(segmentation OR enhancement OR detection OR registration OR diagnosis OR localization OR classification)"
# )

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def clean(text):
    if not text:
        return ""
    return re.sub(r"\s+", " ", str(text)).strip()


def dedup(records):
    """Deduplicate by DOI (case-insensitive), then by normalised title."""
    seen_doi   = {}
    seen_title = {}
    out = []
    for r in records:
        doi       = r.get("doi", "").strip().lower()
        title     = r.get("title", "").strip().lower()
        title_key = re.sub(r"[^a-z0-9 ]", "", title)

        if doi and doi in seen_doi:
            seen_doi[doi]["source"] += f", {r['source']}"
            continue
        if title_key and title_key in seen_title:
            seen_title[title_key]["source"] += f", {r['source']}"
            continue

        out.append(r)
        if doi:
            seen_doi[doi] = r
        if title_key:
            seen_title[title_key] = r
    return out


CSV_FIELDS = [
    "record_id",
    "source",
    "title",
    "authors",
    "year",
    "journal_venue",
    "doi",
    "url",
    "abstract",
    "screen_decision",
    "screen_reason",
    "notes",
]


def load_existing_csv(path):
    """Load existing CSV records; returns [] if file doesn't exist."""
    if not os.path.exists(path):
        print(f"[Reconcile] No existing file found at {path} — starting fresh.")
        return []
    records = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append(dict(row))
    print(f"[Reconcile] Loaded {len(records)} existing records from {path}")
    return records


# ─────────────────────────────────────────────
# SOURCE 1 — PubMed
# ─────────────────────────────────────────────

def fetch_pubmed(query, max_results=MAX_PER_SOURCE):
    print(f"\n[PubMed] Searching … (max {max_results})")
    Entrez.email   = ENTREZ_EMAIL
    Entrez.api_key = NCBI_API_KEY if NCBI_API_KEY else None

    handle = Entrez.esearch(db="pubmed", term=query,
                            retmax=max_results, usehistory="y")
    search = Entrez.read(handle)
    handle.close()
    total = int(search["Count"])
    pmids = search["IdList"]
    print(f"  → {total} total hits; fetching {len(pmids)} records …")

    if not pmids:
        return []

    records = []
    batch   = 200
    for start in range(0, len(pmids), batch):
        chunk  = pmids[start:start + batch]
        handle = Entrez.efetch(db="pubmed", id=",".join(chunk),
                               rettype="xml", retmode="xml")
        data   = Entrez.read(handle)
        handle.close()
        time.sleep(0.4)

        for article in data["PubmedArticle"]:
            try:
                med = article["MedlineCitation"]
                art = med["Article"]
                title = clean(art.get("ArticleTitle", ""))
                auth_list = art.get("AuthorList", [])
                authors = "; ".join(
                    f"{a.get('LastName', '')} {a.get('ForeName', '')}".strip()
                    for a in auth_list if "LastName" in a
                )
                pub_date = art.get("Journal", {}).get(
                    "JournalIssue", {}).get("PubDate", {})
                year    = pub_date.get("Year", pub_date.get("MedlineDate", "")[:4])
                journal = clean(art.get("Journal", {}).get("Title", ""))
                abstract_obj   = art.get("Abstract", {})
                abstract_texts = abstract_obj.get("AbstractText", [])
                if isinstance(abstract_texts, list):
                    abstract = " ".join(str(t) for t in abstract_texts)
                else:
                    abstract = str(abstract_texts)
                abstract = clean(abstract)
                ids  = article.get("PubmedData", {}).get("ArticleIdList", [])
                doi  = next((str(i) for i in ids if str(i).startswith("10.")), "")
                pmid = str(med.get("PMID", ""))
                url  = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else ""
                records.append({
                    "source": "PubMed", "title": title, "authors": authors,
                    "year": str(year), "journal_venue": journal, "doi": doi,
                    "url": url, "abstract": abstract,
                })
            except Exception as e:
                print(f"  [warn] parse error: {e}")

    print(f"  ✓ {len(records)} PubMed records parsed")
    return records


# ─────────────────────────────────────────────
# SOURCE 2 — Semantic Scholar (with backoff)
# ─────────────────────────────────────────────

SS_BASE    = "https://api.semanticscholar.org/graph/v1/paper/search"
SS_FIELDS  = "title,authors,year,venue,externalIds,abstract,openAccessPdf,url"
SS_API_KEY = os.getenv("S2_API_KEY", "s2k-zzVnmfb9aBVopLLnx3K82JPbEbO3FQr4UNGqEbSs")  # register free at semanticscholar.org/product/api


def _ss_get_with_backoff(params, max_retries=8):
    """GET with exponential back-off on 429 / 5xx errors."""
    delay   = 10  # initial wait in seconds (longer without an API key)
    headers = {"x-api-key": SS_API_KEY} if SS_API_KEY else {}
    for attempt in range(max_retries):
        try:
            resp = requests.get(SS_BASE, params=params, headers=headers, timeout=30)
            if resp.status_code == 200:
                return resp
            if resp.status_code == 429:
                wait = delay * (2 ** attempt)
                print(f"  [429] Rate-limited. Waiting {wait}s before retry {attempt+1}/{max_retries} …")
                time.sleep(wait)
                continue
            resp.raise_for_status()
        except requests.exceptions.Timeout:
            wait = delay * (2 ** attempt)
            print(f"  [timeout] Waiting {wait}s before retry {attempt+1}/{max_retries} …")
            time.sleep(wait)
        except Exception as e:
            raise e
    raise Exception(f"SS request failed after {max_retries} retries")


def fetch_semantic_scholar(queries, max_per_query=500):
    print(f"\n[Semantic Scholar] Searching {len(queries)} queries …")
    if SS_API_KEY:
        print(f"  ✓ Using API key — higher rate limits active")
    else:
        print(f"  ⚠ No S2_API_KEY set — using unauthenticated tier (slow). ")
        print(f"    Register free at: semanticscholar.org/product/api")
        print(f"    Then: export S2_API_KEY=\"your_key_here\"")
    all_records = []
    seen_ss_ids = set()

    for q in queries:
        offset  = 0
        limit   = min(500, max_per_query)
        fetched = 0
        print(f"  Query: '{q}'")

        while fetched < max_per_query:
            params = {
                "query":  q,
                "fields": SS_FIELDS,
                "limit":  limit,
                "offset": offset,
            }
            try:
                resp = _ss_get_with_backoff(params)
                data = resp.json()
            except Exception as e:
                print(f"  [warn] SS request failed: {e}")
                break

            papers = data.get("data", [])
            if not papers:
                break

            for p in papers:
                sid = p.get("paperId", "")
                if sid in seen_ss_ids:
                    continue
                seen_ss_ids.add(sid)

                ext      = p.get("externalIds") or {}
                doi      = ext.get("DOI", "")
                authors  = "; ".join(
                    a.get("name", "") for a in (p.get("authors") or [])
                )
                pdf_url  = (p.get("openAccessPdf") or {}).get("url", "")
                url      = p.get("url", pdf_url)
                all_records.append({
                    "source":        "SemanticScholar",
                    "title":         clean(p.get("title", "")),
                    "authors":       authors,
                    "year":          str(p.get("year", "")),
                    "journal_venue": clean(p.get("venue", "")),
                    "doi":           doi,
                    "url":           url,
                    "abstract":      clean(p.get("abstract", "")),
                })

            fetched += len(papers)
            offset  += len(papers)
            if len(papers) < limit:
                break

            # Polite inter-request delay to avoid 429s (longer without API key)
            time.sleep(2 if SS_API_KEY else 10)

        print(f"    → {fetched} fetched (total unique so far: {len(all_records)})")
        # Extra pause between different query strings
        time.sleep(3 if SS_API_KEY else 15)

    print(f"  ✓ {len(all_records)} Semantic Scholar records")
    return all_records


# ─────────────────────────────────────────────
# SOURCE 3 — arXiv (DISABLED)
# ─────────────────────────────────────────────

# arXiv fetch function disabled — arXiv excluded from this SLR
# ARXIV_BASE = "https://export.arxiv.org/api/query"
#
# def fetch_arxiv(query, max_results=MAX_PER_SOURCE):
#     print(f"\n[arXiv] Searching … (max {max_results})")
#     records = []
#     batch   = 200
#     start   = 0
#
#     while start < max_results:
#         size   = min(batch, max_results - start)
#         params = {
#             "search_query": query,
#             "start":        start,
#             "max_results":  size,
#             "sortBy":       "relevance",
#             "sortOrder":    "descending",
#         }
#         try:
#             resp = requests.get(ARXIV_BASE, params=params, timeout=60)
#             resp.raise_for_status()
#             xml = resp.text
#         except Exception as e:
#             print(f"  [warn] arXiv request failed: {e}")
#             break
#
#         entries = re.findall(r"<entry>(.*?)</entry>", xml, re.DOTALL)
#         if not entries:
#             break
#
#         for entry in entries:
#             def tag(t, _e=entry):
#                 m = re.search(rf"<{t}[^>]*>(.*?)</{t}>", _e, re.DOTALL)
#                 return clean(m.group(1)) if m else ""
#
#             title     = tag("title")
#             abstract  = tag("summary")
#             published = tag("published")[:4]
#             doi_m     = re.search(r'<arxiv:doi[^>]*>(.*?)</arxiv:doi>', entry)
#             doi       = clean(doi_m.group(1)) if doi_m else ""
#             link_m    = re.search(r'<id>(.*?)</id>', entry)
#             url       = clean(link_m.group(1)) if link_m else ""
#             authors   = "; ".join(re.findall(r"<name>(.*?)</name>", entry))
#             journal_m = re.search(
#                 r'<arxiv:journal_ref[^>]*>(.*?)</arxiv:journal_ref>', entry)
#             journal   = clean(journal_m.group(1)) if journal_m else "arXiv preprint"
#
#             records.append({
#                 "source": "arXiv", "title": title, "authors": authors,
#                 "year": published, "journal_venue": journal, "doi": doi,
#                 "url": url, "abstract": abstract,
#             })
#
#         start += len(entries)
#         if len(entries) < size:
#             break
#         time.sleep(3)
#
#     print(f"  ✓ {len(records)} arXiv records parsed")
#     return records


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def write_stats(pre_dedup_counts, post_dedup_counts, run_timestamp):
    """Write per-source before/after dedup counts to the stats CSV."""
    import datetime
    stats_fields = ["run_timestamp", "source", "before_dedup", "after_dedup", "duplicates_removed"]
    all_sources  = sorted(set(list(pre_dedup_counts) + list(post_dedup_counts)))

    rows = []
    for src in all_sources:
        before = pre_dedup_counts.get(src, 0)
        after  = post_dedup_counts.get(src, 0)
        rows.append({
            "run_timestamp":     run_timestamp,
            "source":            src,
            "before_dedup":      before,
            "after_dedup":       after,
            "duplicates_removed": before - after,
        })

    # Totals row
    total_before = sum(r["before_dedup"]      for r in rows)
    total_after  = sum(r["after_dedup"]        for r in rows)
    rows.append({
        "run_timestamp":     run_timestamp,
        "source":            "TOTAL",
        "before_dedup":      total_before,
        "after_dedup":       total_after,
        "duplicates_removed": total_before - total_after,
    })

    os.makedirs(os.path.dirname(STATS_FILE), exist_ok=True)
    file_exists = os.path.exists(STATS_FILE)
    with open(STATS_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=stats_fields)
        if not file_exists:
            writer.writeheader()
        writer.writerows(rows)

    print(f"✅ Stats  → {STATS_FILE}")
    print(f"{'Source':<20} {'Before':>10} {'After':>10} {'Removed':>10}")
    print("-" * 54)
    for r in rows:
        print(f"{r['source']:<20} {r['before_dedup']:>10} {r['after_dedup']:>10} {r['duplicates_removed']:>10}")


def main():
    import datetime
    run_timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print("=" * 60)
    mode = "ALL SOURCES" if RUN_ALL else "SEMANTIC SCHOLAR ONLY"
    print(f"AngioVision SLR — Phase 1: Automated Paper Collection [{mode}]")
    print("=" * 60)

    new_records      = []
    pre_dedup_counts = {}   # source → raw count fetched

    if RUN_ALL:
        try:
            pubmed_recs = fetch_pubmed(PUBMED_QUERY)
            pre_dedup_counts["PubMed"] = len(pubmed_recs)
            new_records += pubmed_recs
        except Exception as e:
            print(f"[ERROR] PubMed failed: {e}")
            pre_dedup_counts["PubMed"] = 0

    try:
        ss_recs = fetch_semantic_scholar(SS_QUERIES, max_per_query=500)
        pre_dedup_counts["SemanticScholar"] = len(ss_recs)
        new_records += ss_recs
    except Exception as e:
        print(f"[ERROR] Semantic Scholar failed: {e}")
        pre_dedup_counts["SemanticScholar"] = 0

    # arXiv fetch disabled
    # if RUN_ALL:
    #     try:
    #         new_records += fetch_arxiv(ARXIV_QUERY)
    #     except Exception as e:
    #         print(f"[ERROR] arXiv failed: {e}")

    # ── Reconcile with existing CSV ──────────────────────────────────────
    existing = load_existing_csv(OUTPUT_FILE)

    # Strip record_id and screening columns from existing so dedup works
    # on content fields only; we'll re-assign IDs at the end.
    existing_content = []
    existing_screening = {}  # title_key → {screen_decision, screen_reason, notes}
    for row in existing:
        title_key = re.sub(r"[^a-z0-9 ]", "", row.get("title", "").lower())
        doi       = row.get("doi", "").strip().lower()
        existing_screening[title_key] = {
            "screen_decision": row.get("screen_decision", ""),
            "screen_reason":   row.get("screen_reason", ""),
            "notes":           row.get("notes", ""),
        }
        if doi:
            existing_screening[doi] = existing_screening[title_key]
        existing_content.append({k: row[k] for k in CSV_FIELDS
                                  if k in row and k not in
                                  ("record_id", "screen_decision", "screen_reason", "notes")})

    combined = existing_content + new_records
    print(f"\n[Reconcile] {len(existing_content)} existing + "
          f"{len(new_records)} new = {len(combined)} total before dedup")

    unique = dedup(combined)
    print(f"  ✓ {len(unique)} unique records after deduplication")

    # ── Count surviving records per source (post-dedup) ──────────────────
    post_dedup_counts = {}
    for rec in unique:
        # A record may list multiple sources (e.g. "PubMed, SemanticScholar")
        for src in rec.get("source", "").split(","):
            src = src.strip()
            if src:
                post_dedup_counts[src] = post_dedup_counts.get(src, 0) + 1

    # Re-attach any existing screening decisions
    for rec in unique:
        title_key = re.sub(r"[^a-z0-9 ]", "", rec.get("title", "").lower())
        doi       = rec.get("doi", "").strip().lower()
        screening = (existing_screening.get(doi)
                     or existing_screening.get(title_key)
                     or {})
        rec["screen_decision"] = screening.get("screen_decision", "")
        rec["screen_reason"]   = screening.get("screen_reason", "")
        rec["notes"]           = screening.get("notes", "")

    # ── Write screening CSV ───────────────────────────────────────────────
    unique.sort(key=lambda r: (r.get("year", "0000") or "0000"), reverse=True)

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for i, rec in enumerate(unique, 1):
            rec["record_id"] = f"R{i:04d}"
            writer.writerow(rec)

    print(f"\n✅ Saved → {OUTPUT_FILE}  ({len(unique)} records)")

    # ── Write stats CSV ───────────────────────────────────────────────────
    write_stats(pre_dedup_counts, post_dedup_counts, run_timestamp)

    print("\nNext steps:")
    print("  • Open the CSV in Excel / LibreOffice Calc")
    print("  • Fill 'screen_decision': Include / Exclude / Maybe")
    print("  • Fill 'screen_reason' when excluding (maps to E1–E5 criteria)")
    print("  • Records with 'Maybe' go to full-text review (Stage 2)")


if __name__ == "__main__":
    main()