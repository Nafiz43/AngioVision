"""
CLAIM 2024 Adherence Extraction
AngioVision Systematic Review (PRISMA)

Processes all .md files in --fulltext_dir.
For each article, the LLM is asked to score each of the 12 CLAIM 2024
thematic clusters (Yes / No / NA) and provide a one-sentence rationale.

The 44 CLAIM 2024 items are mapped to 12 clusters; the LLM receives the
full item list for each cluster so it can ground its answers in the text.

Backend : ollama Python library
Model   : gpt-oss:20b  (set via --model)

Usage:
    python3 claim2024_extraction.py
    python3 claim2024_extraction.py --fulltext_dir /path/to/articles --output claim2024_results.jsonl
    python3 claim2024_extraction.py --summary-only
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
MAX_TOKENS         = 3000
RETRY_LIMIT        = 3
RETRY_DELAY        = 5
REPAIR_LIMIT       = 2
MAX_FULLTEXT_CHARS = 80_000

DEFAULT_FULLTEXT_DIR = "/data/Deep_Angiography/AngioVision/slr/articles-processed"
DEFAULT_OUTPUT       = "results/claim2024_results.jsonl"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("claim2024_extraction.log")],
)
log = logging.getLogger(__name__)

# ── CLAIM 2024 cluster definitions ────────────────────────────────────────────
# 44 items collapsed into 12 thematic clusters.
# Each cluster carries the full item wording so the LLM can ground its answer.

CLAIM2024_CLUSTERS = [
    {
        "cluster_id":   "C01",
        "cluster_label": "AI identification & abstract completeness",
        "section":      "Title/Abstract",
        "items_covered": "1-2",
        "items": [
            "Item 1: AI/ML technique (e.g. 'deep learning', 'transformer') identified in title and/or abstract.",
            "Item 2: Abstract contains a structured summary covering study design, methods, results, and conclusions.",
        ],
    },
    {
        "cluster_id":   "C02",
        "cluster_label": "Background & study objectives",
        "section":      "Introduction",
        "items_covered": "3-5",
        "items": [
            "Item 3: Scientific and/or clinical background clearly stated.",
            "Item 4: Study objectives or research questions explicitly defined.",
            "Item 5: Rationale for using AI/ML for this specific task is provided.",
        ],
    },
    {
        "cluster_id":   "C03",
        "cluster_label": "Study design & data sources",
        "section":      "Methods",
        "items_covered": "6-9",
        "items": [
            "Item 6: Study design described (e.g. prospective, retrospective, cross-sectional).",
            "Item 7: Data sources and acquisition process described.",
            "Item 8: Study setting and dates of data collection stated.",
            "Item 9: Eligibility criteria for inclusion/exclusion of participants or images specified.",
        ],
    },
    {
        "cluster_id":   "C04",
        "cluster_label": "Participants & image acquisition",
        "section":      "Methods",
        "items_covered": "10-13",
        "items": [
            "Item 10: Participant characteristics (age, sex, relevant clinical features) reported.",
            "Item 11: De-identification or privacy-protection measures described.",
            "Item 12: Institutional review board (IRB) approval or ethics waiver mentioned.",
            "Item 13: Image acquisition protocol described (manufacturer, sequence/modality parameters, resolution).",
        ],
    },
    {
        "cluster_id":   "C05",
        "cluster_label": "Ground truth & annotation",
        "section":      "Methods",
        "items_covered": "14-16",
        "items": [
            "Item 14: Definition and method used to obtain the reference standard (ground truth) described.",
            "Item 15: Number of annotators and their clinical expertise reported.",
            "Item 16: Inter-rater agreement or quality of annotations reported.",
        ],
    },
    {
        "cluster_id":   "C06",
        "cluster_label": "Data partitioning & class handling",
        "section":      "Methods",
        "items_covered": "17-21",
        "items": [
            "Item 17: Training, validation, and test splits clearly reported.",
            "Item 18: Method of partitioning (random split, site-based, temporal) described.",
            "Item 19: Data augmentation strategies described.",
            "Item 20: Class imbalance and how it was addressed described.",
            "Item 21: Data leakage prevention measures described.",
        ],
    },
    {
        "cluster_id":   "C07",
        "cluster_label": "Model architecture & training",
        "section":      "Methods",
        "items_covered": "22-26",
        "items": [
            "Item 22: Model architecture described in sufficient detail for replication.",
            "Item 23: Pre-trained weights or transfer learning described (backbone, source dataset).",
            "Item 24: Loss function(s) defined.",
            "Item 25: Optimizer, learning rate, and key hyperparameters reported.",
            "Item 26: Stopping criteria or training procedure described.",
        ],
    },
    {
        "cluster_id":   "C08",
        "cluster_label": "Evaluation design & fairness",
        "section":      "Methods",
        "items_covered": "27-31",
        "items": [
            "Item 27: Performance metrics pre-specified before analysis.",
            "Item 28: Statistical analysis plan described.",
            "Item 29: Confidence intervals or uncertainty estimates reported.",
            "Item 30: Robustness or sensitivity analysis performed.",
            "Item 31: Explainability or interpretability methods applied and described.",
        ],
    },
    {
        "cluster_id":   "C09",
        "cluster_label": "Performance & external validation",
        "section":      "Results",
        "items_covered": "32-36",
        "items": [
            "Item 32: Evaluation on internal (held-out) test data reported.",
            "Item 33: External validation dataset described and results reported.",
            "Item 34: Participant/image flow described (numbers screened, included, excluded).",
            "Item 35: Failure case or error analysis reported.",
            "Item 36: Model performance compared to human expert or prior methods.",
        ],
    },
    {
        "cluster_id":   "C10",
        "cluster_label": "Demographic & subgroup reporting",
        "section":      "Results",
        "items_covered": "37-39",
        "items": [
            "Item 37: Demographic characteristics of the study population reported.",
            "Item 38: Subgroup or stratified performance analysis reported (e.g. by sex, age, site).",
            "Item 39: Fairness or algorithmic bias evaluation reported.",
        ],
    },
    {
        "cluster_id":   "C11",
        "cluster_label": "Limitations, implications & future work",
        "section":      "Discussion",
        "items_covered": "40-42",
        "items": [
            "Item 40: Study limitations identified (methods, data, generalisability, uncertainty).",
            "Item 41: Clinical implications and intended use of the AI model discussed.",
            "Item 42: Reference to full study protocol or additional technical details provided.",
        ],
    },
    {
        "cluster_id":   "C12",
        "cluster_label": "Reproducibility & funding disclosure",
        "section":      "Other Information",
        "items_covered": "43-44",
        "items": [
            "Item 43: Software, trained model, and/or data availability stated (URL or access conditions).",
            "Item 44: Sources of funding and role of funders disclosed; author independence stated.",
        ],
    },
]

# ── System prompt ─────────────────────────────────────────────────────────────
def _build_system_prompt():
    cluster_block = ""
    for c in CLAIM2024_CLUSTERS:
        cluster_block += f'\n  "{c["cluster_id"]}": {{\n'
        cluster_block += f'    // {c["cluster_label"]} (CLAIM 2024 items {c["items_covered"]})\n'
        cluster_block += f'    "adherence": "Yes" | "No" | "NA",\n'
        cluster_block += f'    "rationale": "One sentence citing the specific evidence or absence thereof."\n'
        cluster_block += f'  }},'

    return f"""You are a systematic literature review (SLR) quality assessor specialising in CLAIM 2024 — the Checklist for Artificial Intelligence in Medical Imaging (Tejani et al., Radiology: AI, 2024).

You will be given the full text of a research article in Markdown format. Your task is to assess adherence to each of the 12 CLAIM 2024 thematic clusters listed below and return a structured JSON object.

CLUSTER DEFINITIONS (what each cluster covers):
{json.dumps([{
    "cluster_id": c["cluster_id"],
    "cluster_label": c["cluster_label"],
    "section": c["section"],
    "items": c["items"]
} for c in CLAIM2024_CLUSTERS], indent=2)}

SCORING RULES:
- "Yes"  — the cluster's requirements are clearly and explicitly addressed in the article text.
- "No"   — the cluster's requirements are absent or so incomplete that a reader cannot verify them.
- "NA"   — not applicable to this study design (e.g. no external dataset exists for C09; no annotators needed for a synthetic dataset in C05). Use sparingly and justify.
- Do not infer or hallucinate information. Base scoring strictly on what is written.
- If a cluster has multiple items, score "Yes" only if the MAJORITY of its items are satisfied. Otherwise score "No".
- Keep rationale to one sentence maximum. Cite a specific section, sentence, or absence of evidence.

RESPOND ONLY with a valid JSON object in exactly this structure. No preamble, no markdown fences, no trailing commentary.

{{
  "claim2024_adherence": {{{cluster_block}
  }}
}}"""

SYSTEM_PROMPT = _build_system_prompt()

REPAIR_SYSTEM = """You are a JSON repair assistant. You will be given a broken or empty response
that was supposed to be a valid JSON assessment of CLAIM 2024 adherence.
Return ONLY the corrected JSON object. No preamble, no markdown fences, no explanation."""


# ── Markdown header parser ────────────────────────────────────────────────────
def parse_md_header(text):
    """Extract title, year, source_file from the first 20 lines of the markdown."""
    title       = None
    year        = None
    source_file = None

    for line in text.splitlines()[:20]:
        line = line.strip()
        if title is None and line.startswith("# "):
            title = line[2:].strip()
        if year is None:
            m = re.search(r"\*\*Publication Year:\*\*\s*(\d{4})", line)
            if m:
                year = int(m.group(1))
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
        f"The following response was supposed to be a valid JSON CLAIM 2024 adherence record "
        f"but is broken or empty.\n\n"
        f"BROKEN RESPONSE:\n{broken_raw if broken_raw else '(empty)'}\n\n"
        f"PAPER TITLE: {title}\n\n"
        f"Please return the correct JSON object with keys: claim2024_adherence -> C01 through C12, "
        f"each having 'adherence' (Yes/No/NA) and 'rationale' (one sentence)."
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


# ── LLM call ─────────────────────────────────────────────────────────────────
def _call_model(fulltext, title):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": f"PAPER TITLE: {title}\n\nFULL TEXT:\n{fulltext}"},
    ]
    options  = ollama.Options(temperature=TEMPERATURE, num_predict=MAX_TOKENS)
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

    # Resume: track by _md_file key (the .md filename)
    done_files = set()
    if resume and os.path.exists(output_jsonl):
        with open(output_jsonl, encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    if "_md_file" in rec:
                        done_files.add(rec["_md_file"])
                except Exception:
                    pass
        log.info(f"Resuming: {len(done_files)} files already processed.")

    pending = [f for f in md_files if f.name not in done_files]
    log.info(f"Pending: {len(pending)} files to process.")
    log.info(f"Model: {MODEL}")

    os.makedirs(os.path.dirname(output_jsonl) or ".", exist_ok=True)

    with open(output_jsonl, "a", encoding="utf-8") as out_f:
        with tqdm(
            total=len(pending), desc="CLAIM 2024 extraction", unit="article",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
        ) as pbar:

            for md_path in pending:
                pbar.set_postfix_str(md_path.stem[:60], refresh=True)

                text = md_path.read_text(encoding="utf-8", errors="replace")
                if len(text) > MAX_FULLTEXT_CHARS:
                    log.warning(f"  {md_path.name}: truncated to {MAX_FULLTEXT_CHARS} chars.")
                    text = text[:MAX_FULLTEXT_CHARS]

                header      = parse_md_header(text)
                title       = header["title"]       or md_path.stem
                year        = header["year"]
                source_file = header["source_file"] or md_path.name

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
                    record = {
                        "title":        title,
                        "year":         year,
                        "_source_file": source_file,
                        "_md_file":     md_path.name,
                    }
                    for k, v in llm_result.items():
                        if k not in ("title", "year"):
                            record[k] = v

                out_f.write(json.dumps(record) + "\n")
                out_f.flush()

                status = "ERR" if "_error" in record else "OK"
                pbar.set_postfix_str(f"{title[:40]}… → {status}", refresh=True)
                pbar.update(1)

    log.info(f"\nExtraction complete. JSONL written to: {output_jsonl}")
    _write_summary(output_jsonl)


# ── Summary ───────────────────────────────────────────────────────────────────
def _write_summary(jsonl_path):
    records = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            try:
                rec   = json.loads(line)
                adh   = rec.get("claim2024_adherence", {})
                row   = {
                    "title":       rec.get("title"),
                    "year":        rec.get("year"),
                    "md_file":     rec.get("_md_file"),
                    "error":       rec.get("_error"),
                }
                for c in CLAIM2024_CLUSTERS:
                    cid  = c["cluster_id"]
                    cell = adh.get(cid, {})
                    row[f"{cid}_adherence"] = cell.get("adherence") if isinstance(cell, dict) else None
                    row[f"{cid}_rationale"] = cell.get("rationale") if isinstance(cell, dict) else None
                records.append(row)
            except Exception:
                pass

    df      = pd.DataFrame(records)
    csv_out = jsonl_path.replace(".jsonl", "_summary.csv")
    df.to_csv(csv_out, index=False)
    log.info(f"Summary CSV written to: {csv_out}")

    total   = len(df)
    success = int(df["error"].isna().sum())
    errors  = int(df["error"].notna().sum())

    print("\n" + "=" * 60)
    print("  CLAIM 2024 ADHERENCE EXTRACTION SUMMARY")
    print("=" * 60)
    print(tabulate(
        [["Total processed", total], ["Extracted OK", success], ["Errors", errors]],
        headers=["", "Count"], tablefmt="simple_outline",
    ))

    print(f"\n{'─'*60}")
    print("  CLUSTER ADHERENCE RATES")
    rows = []
    for c in CLAIM2024_CLUSTERS:
        cid = c["cluster_id"]
        col = f"{cid}_adherence"
        if col not in df.columns:
            continue
        yes = int((df[col] == "Yes").sum())
        no  = int((df[col] == "No").sum())
        na  = int((df[col] == "NA").sum())
        denom = yes + no  # exclude NA from proportion
        prop  = f"{100*yes/denom:.1f}%" if denom > 0 else "N/A"
        rows.append([cid, c["cluster_label"][:42], yes, no, na, prop])
    print(tabulate(rows, headers=["ID", "Cluster", "Yes", "No", "NA", "% Yes (excl. NA)"],
                   tablefmt="simple_outline"))
    print("=" * 60)


# ── CLI ───────────────────────────────────────────────────────────────────────
def main():
    global MODEL

    parser = argparse.ArgumentParser(
        description="CLAIM 2024 adherence extraction from .md full-texts (ollama)"
    )
    parser.add_argument("--fulltext_dir", default=DEFAULT_FULLTEXT_DIR,
                        help=f"Directory containing .md files (default: {DEFAULT_FULLTEXT_DIR})")
    parser.add_argument("--output",       default=DEFAULT_OUTPUT,
                        help=f"Output JSONL path (default: {DEFAULT_OUTPUT})")
    parser.add_argument("--model",        default=MODEL,
                        help=f"Ollama model name (default: {MODEL})")
    parser.add_argument("--no-resume",    action="store_true",
                        help="Start fresh, ignore existing output")
    parser.add_argument("--summary-only", action="store_true",
                        help="Skip extraction, re-print summary from existing JSONL")
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