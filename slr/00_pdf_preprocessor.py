#!/usr/bin/env python3
"""
preprocess_articles.py
======================
Recursively converts ALL PDF articles under the papers/ folder into
cleaned, numbered Markdown files (1.md, 2.md, 3.md …).

Core engine: pymupdf4llm.to_markdown()
---------------------------------------
pymupdf4llm (built on PyMuPDF / MuPDF) handles:
  - Single-column and multi-column layout detection automatically
  - Table extraction with proper Markdown output
  - Header/bold detection and Markdown heading promotion
  - Ligature and Unicode normalization
  - Much cleaner text extraction than pdfplumber for complex journals

This script adds on top:
  - Recursive PDF discovery across all sub-folders
  - Stable sequential output naming (1.md, 2.md, … via _index.json)
  - Incremental processing (skip already-done files)
  - Postprocessing layer:
      - (cid:N) garbage removal (broken font encodings)
      - Spaced-out running header stripping (C r i t i c a l …)
      - Header/footer boilerplate removal
      - Trailing section truncation (References, Acknowledgments, etc.)
      - Pipe-flood cleanup (spurious table markup from paragraph text)
      - Year extraction from text when PDF metadata is absent
  - Encrypted / corrupted PDF handling (graceful skip)
  - OCR fallback via pymupdf4llm's built-in OCR support (optional)

Input  : /Users/nafiz43/Documents/GitHub/OVC-Analysis/code/data/papers
           └── AI/
           └── biomarkers/
           └── epidemiology, prognosis, treatment/
           └── genomics, transcriptomics, proteomics/
           └── others/

Output : /Users/nafiz43/Documents/GitHub/OVC-Analysis/code/data/actionable-articles-preprocessed
           └── 1.md, 2.md, 3.md …  (flat, no sub-dirs)
           └── _index.json          (maps number → original PDF path)

Dependencies:
    pip install pymupdf4llm pypdf

Usage:
    python preprocess_articles.py             # normal incremental run
    python preprocess_articles.py --no-skip   # force reprocess everything
    python preprocess_articles.py --debug     # verbose logging
"""

import argparse
import json
import logging
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Optional, List, Tuple

import pymupdf4llm
import pymupdf                          # bundled with pymupdf4llm
from pypdf import PdfReader
from pypdf.errors import PdfReadError

# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# =========================================================================== #
# Regex constants
# =========================================================================== #

# ---------------------------------------------------------------------------
# Trailing section stripping
# ---------------------------------------------------------------------------
_STRIP_SECTION_RE = re.compile(
    r"^\s*"
    r"(?:(?:[0-9]+|[IVXivx]+)[\.\s]\s*)?"
    r"("
    r"acknowledg(?:e?ment|e?ments)"
    r"|references?(?:\s+cited)?"
    r"|literature\s+cited"
    r"|bibliography"
    r"|works\s+cited"
    r"|citations?"
    r"|reference\s+list"
    r"|funding(?:\s+sources?)?"
    r"|conflict(?:s)?\s+of\s+interest(?:s)?"
    r"|disclosure(?:s)?"
    r"|author\s+contribution(?:s)?"
    r"|competing\s+interest(?:s)?"
    r"|supplementary(?:\s+(?:material|data|information))?"
    r"|supporting\s+information"
    r"|appendix"
    r"|data\s+availability(?:\s+statement)?"
    r"|ethics\s+(?:statement|approval|declaration)"
    r"|declaration(?:s)?\s+of\s+(?:interest|competing)"
    r")"
    r"(?:\s+and\s+\w+(?:\s+\w+)*)?"
    r"\s*:?\s*$",
    flags=re.IGNORECASE,
)

# Catches ## References, **Acknowledgments**, etc.
_INLINE_SECTION_RE = re.compile(
    r"^\s*(?:#{1,6}\s*|\*{1,2})?(?:[0-9]+|[IVXivx]+)?[\.\s]*"
    r"(?P<keyword>"
    r"acknowledg(?:e?ment|e?ments)"
    r"|references?(?:\s+cited)?"
    r"|literature\s+cited"
    r"|bibliography|works\s+cited|citations?|reference\s+list"
    r"|funding(?:\s+sources?)?"
    r"|conflict(?:s)?\s+of\s+interest(?:s)?"
    r"|disclosure(?:s)?"
    r"|author\s+contribution(?:s)?"
    r"|competing\s+interest(?:s)?"
    r"|supplementary(?:\s+(?:material|data|information))?"
    r"|supporting\s+information"
    r"|appendix"
    r"|data\s+availability(?:\s+statement)?"
    r"|ethics\s+(?:statement|approval|declaration)"
    r"|declaration(?:s)?\s+of\s+(?:interest|competing)"
    r")"
    r"(?:\s+and\s+[\w\s]+)?"
    r"(?:\*{1,2})?\s*:?\s*$",
    flags=re.IGNORECASE,
)

# Bare single-word heading leftover after noise stripping
_FUZZY_SECTION_RE = re.compile(
    r"^(references?|bibliography|acknowledgements?|acknowledgments?"
    r"|funding|appendix|disclosures?)$",
    flags=re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Header / footer noise
# ---------------------------------------------------------------------------
_YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")

_HEADER_FOOTER_RE = re.compile(
    r"^(\s*("
    r"page\s+\d+"
    r"|\d+\s*/\s*\d+"
    r"|©.*"
    r"|received:.*"
    r"|accepted:.*"
    r"|available\s+online.*"
    r"|doi\s*:.*"
    r"|downloaded\s+from.*"
    r"|published\s+by.*"
    r"|all\s+rights\s+reserved.*"
    r"|www\.\S+"
    r")\s*)$",
    flags=re.IGNORECASE,
)

_PAGE_NUMBER_RE = re.compile(r"^\s*\d{1,4}\s*$")

# Spaced-out running header: "C r i t i c a l  R e v i e w s …"
_SPACED_HEADER_RE = re.compile(
    r"^(?:[A-Za-z0-9]{1,2}\s+){6,}[A-Za-z0-9]{1,2}$"
)

# ---------------------------------------------------------------------------
# Content cleanup
# ---------------------------------------------------------------------------
# Unresolved PDF character ID references (broken font encodings)
_CID_RE = re.compile(r"\(cid:\d+\)")

# Pipe-flood detection: lines where >40% of non-space chars are pipes
_MAX_PIPE_RATIO = 0.40

_LIGATURES = {
    "\ufb00": "ff", "\ufb01": "fi", "\ufb02": "fl",
    "\ufb03": "ffi", "\ufb04": "ffl", "\ufb05": "st", "\ufb06": "st",
}


# =========================================================================== #
# Postprocessing helpers
# =========================================================================== #

def normalize_text(text: str) -> str:
    """Fix ligatures, normalize unicode, strip null bytes."""
    for lig, rep in _LIGATURES.items():
        text = text.replace(lig, rep)
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\x00", "")
    return text


def remove_cid_references(text: str) -> str:
    """
    Strip unresolved PDF character ID references like (cid:21).
    Appears when the PDF uses a custom font encoding (common in Bentham
    Science and some Elsevier journals). Applied first so downstream
    steps see clean text.
    """
    text = _CID_RE.sub("", text)
    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped and re.search(r"[A-Za-z0-9]", stripped):
            lines.append(line)
        elif not stripped:
            lines.append("")
    return "\n".join(lines)


def remove_header_footer_noise(text: str) -> str:
    """
    Strip:
    - Pure page numbers
    - Journal boilerplate (DOI, received/accepted dates, etc.)
    - Spaced-out running headers ("C r i t i c a l  R e v i e w s …")
    - Standalone footnote / URL lines
    """
    cleaned = []
    for line in text.splitlines():
        stripped = line.strip()

        if _PAGE_NUMBER_RE.match(line):
            continue
        if _HEADER_FOOTER_RE.match(line):
            continue
        if len(stripped) > 10 and _SPACED_HEADER_RE.match(stripped):
            continue
        if re.match(r"^\s*(\d+\s+)?https?://\S+\s*$", line):
            continue

        cleaned.append(line)
    return "\n".join(cleaned)


def remove_pipe_flood(text: str) -> str:
    """
    Remove lines that are overwhelmingly pipe characters — spurious Markdown
    table rows generated when paragraph text was mis-detected as a table.
    Lines with >40% pipes are de-piped back into plain prose.
    Pure Markdown separator rows (| --- | --- |) are dropped when orphaned.
    """
    _sep_re = re.compile(r"^\|[\s\-\|]+\|$")
    cleaned = []
    prev_was_table_header = False

    for line in text.splitlines():
        stripped = line.strip()

        if _sep_re.match(stripped):
            if prev_was_table_header:
                cleaned.append(line)
            prev_was_table_header = False
            continue

        if "|" in stripped:
            non_space = stripped.replace(" ", "")
            if non_space:
                pipe_ratio = non_space.count("|") / len(non_space)
                if pipe_ratio > _MAX_PIPE_RATIO:
                    prose = re.sub(r"\|+", " ", stripped).strip()
                    prose = re.sub(r"\s{2,}", " ", prose)
                    if prose and re.search(r"[A-Za-z0-9]", prose):
                        cleaned.append(prose)
                    prev_was_table_header = False
                    continue

        cleaned.append(line)
        prev_was_table_header = bool(
            stripped and stripped.startswith("|") and stripped.endswith("|")
        )

    return "\n".join(cleaned)


def truncate_at_trailing_sections(text: str) -> str:
    """
    Remove everything from the first trailing section heading onward.
    Handles plain / ALL-CAPS / numbered / markdown-heading / bold variants,
    plus fuzzy bare-keyword match for headings left alone after noise stripping.
    """
    lines = text.splitlines()
    cut_at: Optional[int] = None

    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue
        if (
            _STRIP_SECTION_RE.match(line)
            or _INLINE_SECTION_RE.match(line)
            or _FUZZY_SECTION_RE.match(stripped)
        ):
            log.debug(f"Trailing section '{stripped[:60]}' at line {i}")
            cut_at = i
            break

    return "\n".join(lines[:cut_at]) if cut_at is not None else text


def clean_excessive_whitespace(text: str) -> str:
    return re.sub(r"\n{3,}", "\n\n", text)


def postprocess(text: str) -> str:
    """Full postprocessing pipeline applied after pymupdf4llm extraction."""
    text = normalize_text(text)
    text = remove_cid_references(text)
    text = remove_header_footer_noise(text)
    text = remove_pipe_flood(text)
    text = truncate_at_trailing_sections(text)
    text = clean_excessive_whitespace(text)
    return text.strip()


# =========================================================================== #
# Metadata extraction
# =========================================================================== #

def get_metadata(pdf_path: Path) -> Tuple[str, str]:
    """Extract title and publication year from PDF metadata."""
    title = ""
    year  = ""
    try:
        reader = PdfReader(str(pdf_path))
        meta   = reader.metadata or {}

        raw_title = str(meta.get("/Title") or "").strip()
        if raw_title and len(raw_title) > 3:
            title = raw_title

        for key in ("/CreationDate", "/ModDate"):
            val = str(meta.get(key) or "")
            m = _YEAR_RE.search(val)
            if m:
                year = m.group(0)
                break
    except Exception:
        pass

    if not title:
        title = pdf_path.stem.replace("_", " ").replace("-", " ").title()

    return title, year


def extract_year_from_text(text: str) -> str:
    """
    Return the most-frequent modern year (>=2000) in the first 50 lines
    instead of the first year seen (which is often a citation year).
    """
    head = "\n".join(text.splitlines()[:50])
    all_years = _YEAR_RE.findall(head)
    if not all_years:
        return ""
    modern = [y for y in all_years if int(y) >= 2000]
    pool   = modern if modern else all_years
    return Counter(pool).most_common(1)[0][0]


# =========================================================================== #
# PDF type checks
# =========================================================================== #

def is_encrypted(pdf_path: Path) -> bool:
    try:
        return PdfReader(str(pdf_path)).is_encrypted
    except Exception:
        return False


# =========================================================================== #
# Core: PDF → Markdown  (pymupdf4llm)
# =========================================================================== #

def pdf_to_markdown(pdf_path: Path) -> Optional[str]:
    """
    Extract a PDF to clean Markdown using pymupdf4llm.to_markdown().

    pymupdf4llm handles automatically:
      - Multi-column layout detection and correct reading order
      - Table extraction → proper Markdown tables
      - Heading detection from font-size analysis
      - Bold / italic preservation
      - Ligature normalization
      - Much better (cid:N) resolution than pdfplumber

    This function wraps it with:
      - Encryption check
      - Metadata extraction for the header block
      - Postprocessing (CID cleanup, noise removal, section truncation)
    """
    if is_encrypted(pdf_path):
        log.warning(f"Skipping encrypted PDF: {pdf_path.name}")
        return None

    title, year = get_metadata(pdf_path)

    try:
        # pymupdf4llm.to_markdown accepts a file path string directly
        raw_md: str = pymupdf4llm.to_markdown(
            str(pdf_path),
            # Use strict line-based table detection to avoid treating
            # two-column prose as tables
            table_strategy="lines_strict",
            # Don't write image files — we only care about text
            write_images=False,
            ignore_images=True,
            ignore_graphics=True,
            # Don't add page separators (they interrupt paragraph flow)
            page_separators=False,
            # Show font sizes that are too small (footnotes etc.)
            fontsize_limit=3,
            # Don't show progress bar in batch mode
            show_progress=False,
        )
    except Exception as e:
        log.error(f"pymupdf4llm failed on {pdf_path.name}: {e}")
        return None

    if not raw_md or not raw_md.strip():
        log.warning(f"No text extracted from {pdf_path.name}")
        return None

    # Apply postprocessing
    cleaned = postprocess(raw_md)

    if not cleaned.strip():
        log.warning(f"Empty after postprocessing: {pdf_path.name}")
        return None

    # Fill in year from text if metadata didn't provide it
    if not year:
        year = extract_year_from_text(cleaned)

    # Build document header
    header = [f"# {title}"]
    if year:
        header.append(f"\n**Publication Year:** {year}")
    header.append(f"\n**Source File:** `{pdf_path.name}`")
    header.append("\n---\n")

    return "\n".join(header) + "\n" + cleaned


# =========================================================================== #
# Paths
# =========================================================================== #

INPUT_DIR  = Path("/data/Deep_Angiography/Z-SLR/articles")
OUTPUT_DIR = Path("/data/Deep_Angiography/Z-SLR/articles-processed")
INDEX_FILE = OUTPUT_DIR / "_index.json"


# =========================================================================== #
# Index helpers  (stable sequential naming across incremental runs)
# =========================================================================== #

def load_index() -> dict:
    """Load the persistent pdf_path → number mapping."""
    if INDEX_FILE.exists():
        try:
            with INDEX_FILE.open("r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def save_index(index: dict) -> None:
    with INDEX_FILE.open("w", encoding="utf-8") as f:
        json.dump(index, f, indent=2, ensure_ascii=False)


def next_number(index: dict) -> int:
    return max(index.values(), default=0) + 1


# =========================================================================== #
# Batch runner
# =========================================================================== #

def collect_pdfs(root: Path) -> List[Path]:
    """Recursively collect all PDFs, sorted deterministically."""
    return sorted(root.rglob("*.pdf"), key=lambda p: (p.parent.name, p.name))


def process_all(skip_existing: bool = True) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not INPUT_DIR.is_dir():
        log.error(f"Input folder does not exist: {INPUT_DIR}")
        sys.exit(1)

    pdf_files = collect_pdfs(INPUT_DIR)
    if not pdf_files:
        log.warning(f"No PDF files found under: {INPUT_DIR}")
        return

    log.info(f"Found {len(pdf_files)} PDF(s) under {INPUT_DIR}")

    index = load_index()
    success, skipped, failed = 0, 0, 0

    for pdf_path in pdf_files:
        rel_key = str(pdf_path.relative_to(INPUT_DIR))

        if rel_key not in index:
            index[rel_key] = next_number(index)
            save_index(index)

        number   = index[rel_key]
        out_path = OUTPUT_DIR / f"{number}.md"

        if skip_existing and out_path.exists() and out_path.stat().st_size > 0:
            log.info(f"[SKIP]  {rel_key}  →  {out_path.name}")
            skipped += 1
            continue

        log.info(f"[PROC]  {rel_key}  →  {out_path.name}")
        try:
            md = pdf_to_markdown(pdf_path)
            if md is None:
                log.error(f"[FAIL]  {rel_key}")
                failed += 1
                continue
            out_path.write_text(md, encoding="utf-8")
            log.info(f"[OK]    → {out_path.name}  ({len(md.split()):,} words)")
            success += 1
        except Exception as e:
            log.error(f"[FAIL]  {rel_key}: {e}")
            failed += 1

    log.info(f"\nDone.  Success={success}  Skipped={skipped}  Failed={failed}")
    log.info(f"Index: {INDEX_FILE}")


# =========================================================================== #
# Entry point
# =========================================================================== #

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preprocess all PDF articles under papers/ → Markdown (pymupdf4llm)."
    )
    parser.add_argument(
        "--no-skip", action="store_true",
        help="Re-process PDFs even if output already exists."
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Enable debug-level logging."
    )
    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    process_all(skip_existing=not args.no_skip)


if __name__ == "__main__":
    main()
