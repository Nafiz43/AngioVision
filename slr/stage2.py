"""
Stage 2 — Full-Text Structured Data Extraction
AngioVision Systematic Review (PRISMA)

Backend: ollama Python library (no separate `ollama serve` required)
         Model:  GPT-OSS:20b   (`ollama pull GPT-OSS:20b`)

Processes ALL .md files found in --fulltext_dir.
No cross-check with Stage 1 — articles in the folder are assumed included.

Markdown format expected (based on actual article files):
    # <Title>
    **Publication Year:** <year>
    **Source File:** `<filename>`

Title, year, and source file are parsed directly from the markdown header
before the LLM call — so they are never null even if the model misbehaves.

Usage:
    python3 stage2_extraction.py
    python3 stage2_extraction.py --fulltext_dir /path/to/articles --output stage2_results.jsonl
    python3 stage2_extraction.py --summary-only

Output:
    stage2_results.jsonl        — one JSON record per article
    stage2_results_summary.csv  — flattened CSV for quick inspection
"""

# ── Auto-install dependencies ─────────────────────────────────────────────────
import subprocess
import sys

def _ensure(package, import_name=None):
    name = import_name or package
    try:
        __import__(name)
    except ImportError:
        print(f"[bootstrap] Installing '{package}'...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package, "-q"])

_ensure("ollama")
_ensure("pandas")
_ensure("tqdm")
_ensure("tabulate")

# ── Imports ───────────────────────────────────────────────────────────────────
import os
import re
import json
import time
import argparse
import logging
import pandas as pd
import ollama
from pathlib import Path
from tqdm import tqdm
from tabulate import tabulate

# ── Config ────────────────────────────────────────────────────────────────────
MODEL              = "gpt-oss:20b"
TEMPERATURE        = 0
MAX_TOKENS         = 1024
RETRY_LIMIT        = 3
RETRY_DELAY        = 5
REPAIR_LIMIT       = 2
MAX_FULLTEXT_CHARS = 80_000

DEFAULT_FULLTEXT_DIR = "/data/Deep_Angiography/Z-SLR/articles-processed"
DEFAULT_OUTPUT       = "stage2_results.jsonl"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("stage2.log")],
)
log = logging.getLogger(__name__)

# ── System prompt ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are a systematic review data extraction assistant. You will be given the full text
of a research article in Markdown format. Extract the following structured information precisely.

If a field cannot be determined from the text, return null for that field.
Do not infer or hallucinate values not present in the paper.

RESPOND ONLY with a valid JSON object in the schema below. No preamble, no markdown fences.

{
  "study_identity": {
    "authors": ["Last FM", ...],
    "journal_or_venue": "...",
    "doi": "...",
    "publication_type": "journal" | "conference" | "preprint" | "other"
  },
  "imaging": {
    "modality": "DSA" | "fluoroscopy" | "DSA+fluoroscopy" | "other",
    "anatomy": "coronary" | "cerebral" | "peripheral" | "abdominal" | "multi" | "other",
    "frame_rate_fps": null,
    "sequence_length_frames": null
  },
  "dataset": {
    "source": "public" | "private" | "phantom" | "mixed",
    "dataset_name": null,
    "n_patients": null,
    "n_sequences": null,
    "n_frames": null,
    "train_test_split": "..."
  },
  "task": {
    "primary_task": "denoising" | "enhancement" | "subtraction" | "registration" |
                    "motion_correction" | "segmentation" | "detection" |
                    "classification" | "localization" | "outcome_prediction" | "other",
    "secondary_tasks": [],
    "task_description": "..."
  },
  "method": {
    "architecture_family": "CNN" | "U-Net" | "GAN" | "Transformer" | "RNN" |
                           "Mamba" | "hybrid" | "classical" | "other",
    "architecture_name": "...",
    "input_type": "2D_frame" | "3D_sequence" | "optical_flow" | "mixed",
    "temporal_modelling": true | false,
    "pretrained_backbone": null,
    "training_supervision": "fully_supervised" | "self_supervised" | "weakly_supervised" | "unsupervised"
  },
  "evaluation": {
    "metrics": ["PSNR", "SSIM", "Dice", ...],
    "best_metric_value": "...",
    "comparators": ["method_name", ...],
    "validation_design": "held_out_test" | "cross_validation" | "prospective" | "other"
  },
  "limitations": "...",
  "future_work_stated": "...",
  "open_source_code": true | false,
  "open_source_data": true | false
}"""

REPAIR_SYSTEM = """You are a JSON repair assistant. You will be given a broken or empty response
that was supposed to be a valid JSON extraction record for a research paper.
Return ONLY the corrected JSON object. No preamble, no markdown fences, no explanation."""


# ── Markdown header parser ────────────────────────────────────────────────────

def parse_md_header(text):
    """
    Extract title, year, and source_file directly from the markdown header.

    Expected format:
        # A multimodal generative AI copilot for human pathology
        **Publication Year:** 2024
        **Source File:** `Copilot-Nature.pdf`

    Returns dict with keys: title, year, source_file (all may be None if not found).
    """
    title       = None
    year        = None
    source_file = None

    for line in text.splitlines()[:20]:   # only scan the first 20 lines
        line = line.strip()

        # Title: first level-1 heading
        if title is None and line.startswith("# "):
            title = line[2:].strip()

        # Publication year
        if year is None:
            m = re.search(r"\*\*Publication Year:\*\*\s*(\d{4})", line)
            if m:
                year = int(m.group(1))

        # Source file
        if source_file is None:
            m = re.search(r"\*\*Source File:\*\*\s*`?([^`\n]+)`?", line)
            if m:
                source_file = m.group(1).strip()

    return {"title": title, "year": year, "source_file": source_file}


# ── JSON helpers ──────────────────────────────────────────────────────────────

def _extract_json(text):
    if not text:
        return None
    cleaned = re.sub(r"^```(?:json)?\s*", "", text.strip(), flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return None


def _repair_json(broken_raw, title):
    user_content = (
        f"The following response was supposed to be a valid JSON extraction record "
        f"but is broken or empty.\n\n"
        f"BROKEN RESPONSE:\n{broken_raw if broken_raw else '(empty)'}\n\n"
        f"PAPER TITLE: {title}\n\n"
        f"Please return the correct JSON object."
    )
    options = ollama.Options(temperature=0, num_predict=MAX_TOKENS)
    for attempt in range(1, REPAIR_LIMIT + 1):
        try:
            response = ollama.chat(
                model=MODEL,
                messages=[
                    {"role": "system", "content": REPAIR_SYSTEM},
                    {"role": "user",   "content": user_content},
                ],
                format="json",
                options=options,
            )
            raw = response.message.content.strip()
            parsed = _extract_json(raw)
            if parsed:
                log.info(f"  JSON repair succeeded on attempt {attempt}.")
                return parsed, raw
            log.warning(f"  Repair attempt {attempt}/{REPAIR_LIMIT}: still invalid JSON.")
        except Exception as e:
            log.warning(f"  Repair attempt {attempt}/{REPAIR_LIMIT}: error — {e}")
        time.sleep(RETRY_DELAY)
    return None, ""


# ── LLM call ──────────────────────────────────────────────────────────────────

def _call_model(fulltext, title):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": f"PAPER TITLE: {title}\n\nFULL TEXT:\n{fulltext}"},
    ]
    options = ollama.Options(temperature=TEMPERATURE, num_predict=MAX_TOKENS)
    last_raw = ""

    for attempt in range(1, RETRY_LIMIT + 1):
        try:
            response = ollama.chat(
                model=MODEL,
                messages=messages,
                format="json",
                options=options,
            )
            raw = response.message.content.strip()

            if not raw:
                log.warning(f"  Attempt {attempt}/{RETRY_LIMIT}: empty response.")
                last_raw = raw
                time.sleep(RETRY_DELAY)
                continue

            parsed = _extract_json(raw)
            if parsed:
                return parsed

            log.warning(f"  Attempt {attempt}/{RETRY_LIMIT}: JSON parse error | raw={raw[:120]}")
            last_raw = raw
            time.sleep(RETRY_DELAY)

        except ollama.ResponseError as e:
            log.warning(f"  Attempt {attempt}/{RETRY_LIMIT}: ollama ResponseError — {e}")
            last_raw = ""
            time.sleep(RETRY_DELAY)

        except Exception as e:
            log.warning(f"  Attempt {attempt}/{RETRY_LIMIT}: Unexpected error — {e}")
            last_raw = ""
            time.sleep(RETRY_DELAY)

    log.warning(f"  All {RETRY_LIMIT} attempts failed. Attempting JSON repair...")
    parsed, _ = _repair_json(last_raw, title)
    if parsed:
        return parsed

    log.error(f"  JSON repair also failed for: {title}")
    return None


# ── Main extraction loop ──────────────────────────────────────────────────────

def extract_all(fulltext_dir, output_jsonl, resume=True):
    md_files = sorted(Path(fulltext_dir).glob("*.md"))

    if not md_files:
        log.error(f"No .md files found in: {fulltext_dir}")
        return

    log.info(f"Found {len(md_files)} .md files in {fulltext_dir}")

    # Resume: track by md filename
    done_files = set()
    if resume and os.path.exists(output_jsonl):
        with open(output_jsonl, encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    if "_source_file" in rec:
                        done_files.add(rec["_source_file"])
                except Exception:
                    pass
        log.info(f"Resuming: {len(done_files)} files already processed.")

    pending = [f for f in md_files if f.name not in done_files]
    log.info(f"Pending: {len(pending)} files to process.")

    with open(output_jsonl, "a", encoding="utf-8") as out_f:
        with tqdm(total=len(pending), desc="Extracting", unit="article",
                  bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]") as pbar:

            for md_path in pending:
                pbar.set_postfix_str(md_path.stem[:60], refresh=True)

                # Load text
                text = md_path.read_text(encoding="utf-8", errors="replace")
                if len(text) > MAX_FULLTEXT_CHARS:
                    log.warning(f"  {md_path.name}: truncated to {MAX_FULLTEXT_CHARS} chars.")
                    text = text[:MAX_FULLTEXT_CHARS]

                # Parse header fields directly — no LLM needed for these
                header = parse_md_header(text)
                title       = header["title"]       or md_path.stem
                year        = header["year"]
                source_file = header["source_file"] or md_path.name

                # LLM extraction of the remaining schema fields
                llm_result = _call_model(text, title)

                if llm_result is None:
                    record = {
                        "title":        title,
                        "year":         year,
                        "_source_file": source_file,
                        "_md_file":     md_path.name,
                        "_error":       "Extraction failed after all retries",
                    }
                else:
                    # Merge: header fields take precedence over whatever the LLM returned
                    record = {
                        "title":        title,
                        "year":         year,
                        "_source_file": source_file,
                        "_md_file":     md_path.name,
                    }
                    # Carry over all LLM-extracted fields except title/year (already set above)
                    for k, v in llm_result.items():
                        if k not in ("title", "year"):
                            record[k] = v

                out_f.write(json.dumps(record) + "\n")
                out_f.flush()

                status = "ERROR" if "_error" in record else title[:45]
                pbar.set_postfix_str(f"{status}… → {'ERR' if '_error' in record else 'OK'}", refresh=True)
                pbar.update(1)

    log.info(f"\nExtraction complete. JSONL written to: {output_jsonl}")
    _write_summary(output_jsonl)


# ── Summary ───────────────────────────────────────────────────────────────────

def _write_summary(jsonl_path):
    records = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
                records.append({
                    "title":               rec.get("title"),
                    "year":                rec.get("year"),
                    "source_file":         rec.get("_source_file"),
                    "md_file":             rec.get("_md_file"),
                    "journal":             rec.get("study_identity", {}).get("journal_or_venue"),
                    "doi":                 rec.get("study_identity", {}).get("doi"),
                    "publication_type":    rec.get("study_identity", {}).get("publication_type"),
                    "modality":            rec.get("imaging", {}).get("modality"),
                    "anatomy":             rec.get("imaging", {}).get("anatomy"),
                    "primary_task":        rec.get("task", {}).get("primary_task"),
                    "architecture_family": rec.get("method", {}).get("architecture_family"),
                    "architecture_name":   rec.get("method", {}).get("architecture_name"),
                    "temporal_modelling":  rec.get("method", {}).get("temporal_modelling"),
                    "validation_design":   rec.get("evaluation", {}).get("validation_design"),
                    "metrics":             json.dumps(rec.get("evaluation", {}).get("metrics", [])),
                    "open_source_code":    rec.get("open_source_code"),
                    "open_source_data":    rec.get("open_source_data"),
                    "error":               rec.get("_error"),
                })
            except Exception:
                pass

    df = pd.DataFrame(records)
    csv_path = jsonl_path.replace(".jsonl", "_summary.csv").replace(".json", "_summary.csv")
    df.to_csv(csv_path, index=False)
    log.info(f"Summary CSV written to: {csv_path}")

    def _tbl(rows, headers):
        print()
        print(tabulate(rows, headers=headers, tablefmt="simple_outline"))
        print()

    total   = len(df)
    success = int(df["error"].isna().sum())
    errors  = int(df["error"].notna().sum())

    print("\n" + "=" * 55)
    print("  STAGE 2 SUMMARY")
    print("=" * 55)

    _tbl(
        [["Total processed", total], ["Extracted OK", success], ["Errors", errors]],
        ["", "Count"]
    )

    for col, label in [
        ("architecture_family", "ARCHITECTURE FAMILY"),
        ("primary_task",        "PRIMARY TASK"),
        ("modality",            "IMAGING MODALITY"),
        ("anatomy",             "ANATOMY"),
        ("temporal_modelling",  "TEMPORAL MODELLING"),
        ("validation_design",   "VALIDATION DESIGN"),
        ("publication_type",    "PUBLICATION TYPE"),
    ]:
        if col not in df.columns or df[col].isna().all():
            continue
        counts = df[col].value_counts(dropna=False)
        print(f"{'─'*55}")
        print(f"  {label}")
        rows = [[str(k), v, f"{100*v/total:.1f}%"] for k, v in counts.items()]
        _tbl(rows, ["Value", "Count", "%"])

    # Year distribution
    year_col = pd.to_numeric(df["year"], errors="coerce").dropna()
    if not year_col.empty:
        print(f"{'─'*55}")
        print(f"  PUBLICATION YEAR RANGE")
        _tbl(
            [["Min", int(year_col.min())],
             ["Max", int(year_col.max())],
             ["Median", int(year_col.median())],
             ["Total with year", len(year_col)]],
            ["", ""]
        )

    print("=" * 55)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    global MODEL

    parser = argparse.ArgumentParser(
        description="Stage 2 — Full-Text Extraction from .md files (ollama / GPT-OSS:20b)"
    )
    parser.add_argument(
        "--fulltext_dir",
        default=DEFAULT_FULLTEXT_DIR,
        help=f"Directory containing .md files (default: {DEFAULT_FULLTEXT_DIR})"
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help=f"Output JSONL path (default: {DEFAULT_OUTPUT})"
    )
    parser.add_argument(
        "--model",
        default=MODEL,
        help=f"Ollama model name (default: {MODEL})"
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Start fresh, ignore existing output"
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Skip extraction, re-print summary from existing JSONL"
    )
    args = parser.parse_args()

    MODEL = args.model

    if args.summary_only:
        if not os.path.exists(args.output):
            print(f"Error: output file '{args.output}' not found.")
            sys.exit(1)
        _write_summary(args.output)
    else:
        extract_all(
            fulltext_dir=args.fulltext_dir,
            output_jsonl=args.output,
            resume=not args.no_resume,
        )


if __name__ == "__main__":
    main()
