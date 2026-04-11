"""
SLR Phase 1 — Automated Paper Fetcher
Domain: Deep Learning for Angiographic Sequence Processing
Sources: PubMed, Semantic Scholar, arXiv
Output:  slr_stage1_screening.csv  (ready for title/abstract screening)

Usage:
    pip install biopython requests
    python slr_fetch.py

Optional: set your NCBI API key as an env var for higher PubMed rate limits:
    export NCBI_API_KEY="your_key_here"
    (Get a free key at: https://www.ncbi.nlm.nih.gov/account/)
"""

import csv
import time
import os
import re
import json
import requests
from Bio import Entrez

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

ENTREZ_EMAIL   = "nikhan@ucdavis.edu"          # required by NCBI
NCBI_API_KEY   = os.getenv("NCBI_API_KEY", "b62cdd0f606eecea1e4ec3fb9b348b8b9809")
OUTPUT_FILE    = "slr_stage1_screening.csv"
MAX_PER_SOURCE = 2000                            # max records to pull per source

# ── Boolean query (adapted from your SLR protocol) ─────────────────────────

PUBMED_QUERY = (
    "(angiograph*[tiab] OR fluoroscop*[tiab] OR "
    "\"digital subtraction angiography\"[tiab] OR DSA[tiab]) "
    "AND (sequence*[tiab] OR video[tiab] OR frame*[tiab] OR "
    "time-series[tiab] OR temporal[tiab]) "
    "AND (processing[tiab] OR enhancement[tiab] OR denois*[tiab] OR "
    "subtraction[tiab] OR registration[tiab] OR motion[tiab] OR "
    "segmentation[tiab] OR detection[tiab] OR classification[tiab] OR "
    "localization[tiab]) "
    "AND (\"deep learning\"[tiab] OR \"machine learning\"[tiab] OR "
    "CNN[tiab] OR transformer[tiab]) "
    "AND (\"2000\"[pdat] : \"3000\"[pdat]) "
    "AND english[lang]"
)

# Semantic Scholar / arXiv keyword list (API uses text search, not Boolean)
SS_QUERIES = [
    "deep learning digital subtraction angiography sequence",
    "CNN fluoroscopy image enhancement temporal",
    "transformer angiographic vessel segmentation",
    "machine learning DSA motion correction",
    "deep learning fluoroscopic sequence processing",
]

ARXIV_QUERY = (
    "ti_abs:(angiograph* OR fluoroscop* OR DSA) AND "
    "ti_abs:(deep learning OR CNN OR transformer OR machine learning) AND "
    "ti_abs:(segmentation OR enhancement OR detection OR registration)"
)

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def clean(text):
    """Strip newlines and extra whitespace."""
    if not text:
        return ""
    return re.sub(r"\s+", " ", str(text)).strip()


def dedup(records):
    """Deduplicate by DOI (case-insensitive), then by title similarity."""
    seen_doi   = {}
    seen_title = {}
    out = []
    for r in records:
        doi   = r.get("doi", "").strip().lower()
        title = r.get("title", "").strip().lower()
        # normalise title for fuzzy-ish dedup (remove punctuation, lower)
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
    "record_id",    # sequential ID for screening tool
    "source",       # PubMed | SemanticScholar | arXiv
    "title",
    "authors",
    "year",
    "journal_venue",
    "doi",
    "url",
    "abstract",
    # Stage-1 screening columns (fill in manually or via LLM)
    "screen_decision",   # Include / Exclude / Maybe
    "screen_reason",
    "notes",
]

# ─────────────────────────────────────────────
# SOURCE 1 — PubMed (via Biopython Entrez)
# ─────────────────────────────────────────────

def fetch_pubmed(query, max_results=MAX_PER_SOURCE):
    print(f"\n[PubMed] Searching … (max {max_results})")
    Entrez.email   = ENTREZ_EMAIL
    Entrez.api_key = NCBI_API_KEY if NCBI_API_KEY else None

    # Step 1: get PMIDs
    handle = Entrez.esearch(db="pubmed", term=query,
                            retmax=max_results, usehistory="y")
    search = Entrez.read(handle)
    handle.close()
    total  = int(search["Count"])
    pmids  = search["IdList"]
    print(f"  → {total} total hits; fetching {len(pmids)} records …")

    if not pmids:
        return []

    # Step 2: fetch full records in batches
    records = []
    batch   = 200
    for start in range(0, len(pmids), batch):
        chunk = pmids[start:start + batch]
        handle = Entrez.efetch(db="pubmed", id=",".join(chunk),
                               rettype="xml", retmode="xml")
        data   = Entrez.read(handle)
        handle.close()
        time.sleep(0.4)  # be polite to NCBI

        for article in data["PubmedArticle"]:
            try:
                med = article["MedlineCitation"]
                art = med["Article"]

                title = clean(art.get("ArticleTitle", ""))

                # authors
                auth_list = art.get("AuthorList", [])
                authors   = "; ".join(
                    f"{a.get('LastName', '')} {a.get('ForeName', '')}".strip()
                    for a in auth_list
                    if "LastName" in a
                )

                # year
                pub_date = art.get("Journal", {}).get(
                    "JournalIssue", {}).get("PubDate", {})
                year = pub_date.get("Year", pub_date.get("MedlineDate", "")[:4])

                # journal
                journal = clean(
                    art.get("Journal", {}).get("Title", "")
                )

                # abstract
                abstract_obj = art.get("Abstract", {})
                abstract_texts = abstract_obj.get("AbstractText", [])
                if isinstance(abstract_texts, list):
                    abstract = " ".join(str(t) for t in abstract_texts)
                else:
                    abstract = str(abstract_texts)
                abstract = clean(abstract)

                # DOI
                ids  = article.get("PubmedData", {}).get(
                    "ArticleIdList", [])
                doi  = next((str(i) for i in ids
                             if str(i).startswith("10.")), "")
                pmid = str(med.get("PMID", ""))
                url  = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else ""

                records.append({
                    "source":        "PubMed",
                    "title":         title,
                    "authors":       authors,
                    "year":          str(year),
                    "journal_venue": journal,
                    "doi":           doi,
                    "url":           url,
                    "abstract":      abstract,
                })
            except Exception as e:
                print(f"  [warn] parse error: {e}")

    print(f"  ✓ {len(records)} PubMed records parsed")
    return records


# ─────────────────────────────────────────────
# SOURCE 2 — Semantic Scholar (free API)
# ─────────────────────────────────────────────

SS_BASE  = "https://api.semanticscholar.org/graph/v1/paper/search"
SS_FIELDS = "title,authors,year,venue,externalIds,abstract,openAccessPdf,url"

def fetch_semantic_scholar(queries, max_per_query=100):
    print(f"\n[Semantic Scholar] Searching {len(queries)} queries …")
    all_records = []
    seen_ss_ids = set()

    for q in queries:
        offset = 0
        limit  = min(100, max_per_query)
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
                resp = requests.get(SS_BASE, params=params, timeout=20)
                resp.raise_for_status()
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

                ext  = p.get("externalIds") or {}
                doi  = ext.get("DOI", "")
                arxiv_id = ext.get("ArXiv", "")

                authors = "; ".join(
                    a.get("name", "") for a in (p.get("authors") or [])
                )
                pdf_url = (p.get("openAccessPdf") or {}).get("url", "")
                url     = p.get("url", pdf_url)

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
            time.sleep(1.1)  # SS rate limit ~1 req/s

        print(f"    → {fetched} fetched so far (total unique: {len(all_records)})")

    print(f"  ✓ {len(all_records)} Semantic Scholar records")
    return all_records


# ─────────────────────────────────────────────
# SOURCE 3 — arXiv (REST API)
# ─────────────────────────────────────────────

ARXIV_BASE = "https://export.arxiv.org/api/query"

def fetch_arxiv(query, max_results=MAX_PER_SOURCE):
    print(f"\n[arXiv] Searching … (max {max_results})")
    records = []
    batch   = 200
    start   = 0

    while start < max_results:
        size = min(batch, max_results - start)
        params = {
            "search_query": query,
            "start":        start,
            "max_results":  size,
            "sortBy":       "relevance",
            "sortOrder":    "descending",
        }
        try:
            resp = requests.get(ARXIV_BASE, params=params, timeout=30)
            resp.raise_for_status()
            xml = resp.text
        except Exception as e:
            print(f"  [warn] arXiv request failed: {e}")
            break

        # Simple XML parsing (no lxml needed)
        entries = re.findall(r"<entry>(.*?)</entry>", xml, re.DOTALL)
        if not entries:
            break

        for entry in entries:
            def tag(t):
                m = re.search(rf"<{t}[^>]*>(.*?)</{t}>", entry, re.DOTALL)
                return clean(m.group(1)) if m else ""

            title    = tag("title")
            abstract = tag("summary")
            published = tag("published")[:4]  # year only
            doi_m    = re.search(r'<arxiv:doi[^>]*>(.*?)</arxiv:doi>', entry)
            doi      = clean(doi_m.group(1)) if doi_m else ""
            link_m   = re.search(r'<id>(.*?)</id>', entry)
            url      = clean(link_m.group(1)) if link_m else ""
            authors  = "; ".join(
                re.findall(r"<name>(.*?)</name>", entry)
            )
            journal_m = re.search(
                r'<arxiv:journal_ref[^>]*>(.*?)</arxiv:journal_ref>', entry)
            journal  = clean(journal_m.group(1)) if journal_m else "arXiv preprint"

            records.append({
                "source":        "arXiv",
                "title":         title,
                "authors":       authors,
                "year":          published,
                "journal_venue": journal,
                "doi":           doi,
                "url":           url,
                "abstract":      abstract,
            })

        start += len(entries)
        if len(entries) < size:
            break
        time.sleep(3)  # arXiv asks for 3s between requests

    print(f"  ✓ {len(records)} arXiv records parsed")
    return records


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    print("=" * 60)
    print("AngioVision SLR — Phase 1: Automated Paper Collection")
    print("=" * 60)

    all_records = []

    # --- PubMed ---
    try:
        all_records += fetch_pubmed(PUBMED_QUERY)
    except Exception as e:
        print(f"[ERROR] PubMed failed: {e}")

    # --- Semantic Scholar ---
    try:
        all_records += fetch_semantic_scholar(SS_QUERIES, max_per_query=100)
    except Exception as e:
        print(f"[ERROR] Semantic Scholar failed: {e}")

    # --- arXiv ---
    try:
        all_records += fetch_arxiv(ARXIV_QUERY)
    except Exception as e:
        print(f"[ERROR] arXiv failed: {e}")

    # --- Deduplicate ---
    print(f"\n[Dedup] {len(all_records)} total → deduplicating …")
    unique = dedup(all_records)
    print(f"  ✓ {len(unique)} unique records after deduplication")

    # --- Write CSV ---
    unique.sort(key=lambda r: (r.get("year", "0000") or "0000"), reverse=True)

    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for i, rec in enumerate(unique, 1):
            rec["record_id"]      = f"R{i:04d}"
            rec["screen_decision"] = ""
            rec["screen_reason"]   = ""
            rec["notes"]           = ""
            writer.writerow(rec)

    print(f"\n✅ Saved → {OUTPUT_FILE}  ({len(unique)} records)")
    print("\nNext steps:")
    print("  • Open the CSV in Excel / LibreOffice Calc")
    print("  • Fill 'screen_decision' column: Include / Exclude / Maybe")
    print("  • Fill 'screen_reason' when excluding (maps to E1–E5 criteria)")
    print("  • Records with 'Maybe' go to full-text review (Stage 2)")


if __name__ == "__main__":
    main()