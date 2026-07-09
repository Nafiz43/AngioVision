"""
Stage 2 — Temporal Ablation Analysis
AngioVision Systematic Review (PRISMA)

Backend: ollama Python library
         Model:  gpt-oss:20b  (same as stage2_extraction.py)

Identifies articles that explicitly compared performance:
  (a) WITH  a temporal component (e.g., LSTM, attention over frames, optical flow)
  (b) WITHOUT that temporal component (e.g., single-frame baseline, ablated version)

Only records where BOTH settings are quantitatively reported are flagged
as confirmed_ablation=true.

Markdown format expected:
    # <Title>
    **Publication Year:** <year>
    **Source File:** `<filename>`

Usage:
    python3 stage2_temporal_ablation.py
    python3 stage2_temporal_ablation.py --fulltext_dir /path/to/articles --output results/temporal_ablation.jsonl
    python3 stage2_temporal_ablation.py --summary-only
    python3 stage2_temporal_ablation.py --summary-only --output results/temporal_ablation.jsonl

Output:
    temporal_ablation.jsonl        — one JSON record per article
    temporal_ablation_summary.csv  — flattened CSV, confirmed ablations only
    temporal_ablation_report.txt   — human-readable comparison table
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
MAX_TOKENS         = 8192
RETRY_LIMIT        = 3
RETRY_DELAY        = 5
REPAIR_LIMIT       = 2
MAX_FULLTEXT_CHARS = 80_000

DEFAULT_FULLTEXT_DIR = "/data/Deep_Angiography/AngioVision/slr/articles-processed"
DEFAULT_OUTPUT       = "results/temporal_ablation.jsonl"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("stage2_temporal_ablation.log"),
    ],
)
log = logging.getLogger(__name__)

# ── System prompt ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are a systematic literature review assistant specialising in deep learning for medical image analysis.
Your sole task is to determine whether the given paper explicitly reports a quantitative ablation study that compares performance
WITH and WITHOUT a temporal component.

DEFINITIONS
- Temporal component: any mechanism that exploits information across multiple frames or time steps in a sequence.
  Examples include — but are not limited to — LSTM / GRU / RNN layers, temporal attention, 3D convolutions,
  video transformers (TimeSformer, Video Swin), optical-flow inputs, convolutional LSTMs, multi-frame stacks,
  recurrent decoders, and motion-compensation modules.
- "With temporal" setting: the full model including the temporal component.
- "Without temporal" setting: the same model but with the temporal component removed, replaced by a single-frame
  equivalent, or compared against a 2D/single-frame baseline — with numeric results reported in the paper.

EXTRACTION RULES
1. confirmed_ablation must be true ONLY when the paper contains a table, figure, or explicit text that reports
   numeric performance values for BOTH the with-temporal and without-temporal configurations.
2. If only one setting is reported (e.g., the full temporal model with no ablation), set confirmed_ablation = false.
3. Extract ALL metric rows that directly compare the two settings — capture every reported number.
4. comparison_table should list rows exactly as they appear (one row per metric / experiment variant).
5. temporal_component_names: list the exact names used by the authors (e.g., "LSTM branch", "temporal attention").
6. For metric values, preserve the exact string from the paper (e.g., "0.923", "92.3%", "87.4 ± 1.2").
7. If a field cannot be determined, return null.  For lists, return [].
8. conclusion_on_temporal: a brief verbatim or near-verbatim excerpt (≤2 sentences) where the authors state
   what the temporal component contributes.

RESPOND ONLY with a valid JSON object matching the schema below.
No preamble, no markdown fences, no trailing commentary.

{
  "confirmed_ablation": true | false,

  "temporal_component_present": true | false | null,
  "temporal_component_names": ["...", "..."],
  "temporal_mechanism_description": "...",

  "task": "...",
  "primary_metric": "...",
  "dataset_split_reported": "test" | "validation" | "cross_val" | "other" | null,

  "with_temporal": {
    "label": "...",
    "results": [
      { "metric": "...", "value": "..." }
    ]
  },

  "without_temporal": {
    "label": "...",
    "results": [
      { "metric": "...", "value": "..." }
    ]
  },

  "comparison_table": [
    {
      "variant_name": "...",
      "has_temporal": true | false,
      "metric": "...",
      "value": "..."
    }
  ],

  "performance_delta": [
    {
      "metric": "...",
      "delta": "...",
      "direction": "improvement" | "degradation" | "neutral" | null
    }
  ],

  "ablation_section": "...",
  "conclusion_on_temporal": "...",

  "additional_ablation_factors": ["...", "..."]
}"""

REPAIR_SYSTEM = """You are a JSON repair assistant. You will be given a broken or empty response
that was supposed to be a valid JSON extraction record for a research paper.
Return ONLY the corrected JSON object. No preamble, no markdown fences, no explanation."""


# ── Markdown header parser ────────────────────────────────────────────────────

def parse_md_header(text):
    """Extract title, year, and source_file from the markdown header (first 20 lines)."""
    title = year = source_file = None
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
        f"The following response was supposed to be a valid JSON temporal-ablation record "
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
        {
            "role": "user",
            "content": (
                f"PAPER TITLE: {title}\n\n"
                f"FULL TEXT:\n{fulltext}"
            ),
        },
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

            log.warning(
                f"  Attempt {attempt}/{RETRY_LIMIT}: JSON parse error | raw={raw[:120]}"
            )
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

    log.warning(f"  All {RETRY_LIMIT} attempts failed — attempting JSON repair…")
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
    log.info(f"Model: {MODEL}")

    # Resume: track by _md_file key
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

    # Ensure output directory exists
    Path(output_jsonl).parent.mkdir(parents=True, exist_ok=True)

    with open(output_jsonl, "a", encoding="utf-8") as out_f:
        bar_fmt = "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
        with tqdm(total=len(pending), desc="Analysing", unit="article",
                  bar_format=bar_fmt) as pbar:

            for md_path in pending:
                pbar.set_postfix_str(md_path.stem[:60], refresh=True)

                text = md_path.read_text(encoding="utf-8", errors="replace")
                if len(text) > MAX_FULLTEXT_CHARS:
                    log.warning(
                        f"  {md_path.name}: truncated to {MAX_FULLTEXT_CHARS} chars."
                    )
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
                    record.update(llm_result)

                out_f.write(json.dumps(record) + "\n")
                out_f.flush()

                confirmed = record.get("confirmed_ablation", False)
                status_tag = "✓ ABLATION" if confirmed else "– no ablation"
                if "_error" in record:
                    status_tag = "ERR"
                pbar.set_postfix_str(f"{title[:40]}… {status_tag}", refresh=True)
                pbar.update(1)

    log.info(f"Extraction complete. JSONL → {output_jsonl}")
    _write_summary(output_jsonl)


# ── Summary & report ──────────────────────────────────────────────────────────

def _flatten_record(rec):
    """Flatten one JSONL record to a single CSV row."""

    def _first_val(results_list, metric=None):
        """Return value of first result entry (optionally matching metric name)."""
        if not results_list:
            return None
        if metric:
            for r in results_list:
                if metric.lower() in r.get("metric", "").lower():
                    return r.get("value")
        return results_list[0].get("value") if results_list else None

    with_res  = rec.get("with_temporal",    {}).get("results", [])
    wo_res    = rec.get("without_temporal", {}).get("results", [])
    primary   = rec.get("primary_metric")

    return {
        "title":                         rec.get("title"),
        "year":                          rec.get("year"),
        "source_file":                   rec.get("_source_file"),
        "md_file":                       rec.get("_md_file"),
        "confirmed_ablation":            rec.get("confirmed_ablation"),
        "temporal_component_present":    rec.get("temporal_component_present"),
        "temporal_component_names":      "; ".join(rec.get("temporal_component_names") or []),
        "temporal_mechanism":            rec.get("temporal_mechanism_description"),
        "task":                          rec.get("task"),
        "primary_metric":                primary,
        "dataset_split":                 rec.get("dataset_split_reported"),
        "with_temporal_label":           rec.get("with_temporal",    {}).get("label"),
        "with_temporal_primary_value":   _first_val(with_res,  primary),
        "with_temporal_all_results":     json.dumps(with_res),
        "without_temporal_label":        rec.get("without_temporal", {}).get("label"),
        "without_temporal_primary_value":_first_val(wo_res, primary),
        "without_temporal_all_results":  json.dumps(wo_res),
        "performance_delta":             json.dumps(rec.get("performance_delta", [])),
        "ablation_section":              rec.get("ablation_section"),
        "conclusion_on_temporal":        rec.get("conclusion_on_temporal"),
        "additional_ablation_factors":   "; ".join(rec.get("additional_ablation_factors") or []),
        "error":                         rec.get("_error"),
    }


def _write_summary(jsonl_path):
    records = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            try:
                records.append(_flatten_record(json.loads(line)))
            except Exception:
                pass

    df = pd.DataFrame(records)

    # ── Full CSV (all articles) ───────────────────────────────────────────────
    csv_all = jsonl_path.replace(".jsonl", "_summary.csv").replace(".json", "_summary.csv")
    df.to_csv(csv_all, index=False)
    log.info(f"Full summary CSV  → {csv_all}")

    # ── Confirmed-ablation CSV ────────────────────────────────────────────────
    df_confirmed = df[df["confirmed_ablation"] == True].copy()
    csv_conf = jsonl_path.replace(".jsonl", "_confirmed.csv").replace(".json", "_confirmed.csv")
    df_confirmed.to_csv(csv_conf, index=False)
    log.info(f"Confirmed CSV     → {csv_conf}")

    # ── Human-readable report ─────────────────────────────────────────────────
    report_path = jsonl_path.replace(".jsonl", "_report.txt").replace(".json", "_report.txt")
    _write_text_report(jsonl_path, df_confirmed, report_path)

    # ── Console summary ───────────────────────────────────────────────────────
    _print_console_summary(df, df_confirmed)


def _write_text_report(jsonl_path, df_confirmed, report_path):
    """Write a detailed human-readable report for confirmed ablation articles."""
    lines = []
    sep   = "=" * 80

    lines.append(sep)
    lines.append("  ANGIOVISION SLR — TEMPORAL ABLATION ANALYSIS REPORT")
    lines.append(f"  Source: {jsonl_path}")
    lines.append(f"  Confirmed ablation studies: {len(df_confirmed)}")
    lines.append(sep)

    # Re-load full records for confirmed articles to get comparison_table
    confirmed_md_files = set(df_confirmed["md_file"].dropna())
    full_records = {}
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
                if rec.get("_md_file") in confirmed_md_files:
                    full_records[rec["_md_file"]] = rec
            except Exception:
                pass

    for i, (_, row) in enumerate(df_confirmed.iterrows(), 1):
        md_key = row.get("md_file")
        rec    = full_records.get(md_key, {})

        lines.append("")
        lines.append(f"[{i:02d}] {row['title'] or 'Unknown Title'}")
        lines.append(f"     Year        : {row.get('year', 'N/A')}")
        lines.append(f"     Source File : {row.get('source_file', 'N/A')}")
        lines.append(f"     Task        : {row.get('task', 'N/A')}")
        lines.append(f"     Temporal    : {row.get('temporal_component_names', 'N/A')}")
        lines.append(f"     Mechanism   : {row.get('temporal_mechanism', 'N/A')}")
        lines.append(f"     Split       : {row.get('dataset_split', 'N/A')}")
        lines.append("")

        # Comparison table
        comp_table = rec.get("comparison_table", [])
        if comp_table:
            lines.append("     COMPARISON TABLE:")
            tbl_rows = []
            for entry in comp_table:
                temporal_flag = "✓" if entry.get("has_temporal") else "✗"
                tbl_rows.append([
                    temporal_flag,
                    entry.get("variant_name", ""),
                    entry.get("metric", ""),
                    entry.get("value", ""),
                ])
            lines.append(
                tabulate(
                    tbl_rows,
                    headers=["Temp?", "Variant", "Metric", "Value"],
                    tablefmt="simple_outline",
                )
            )
        else:
            # Fall back to with/without results
            lines.append("     WITH TEMPORAL:")
            with_res = rec.get("with_temporal", {})
            lines.append(f"       Label   : {with_res.get('label', 'N/A')}")
            for r in (with_res.get("results") or []):
                lines.append(f"       {r.get('metric', '')}: {r.get('value', '')}")

            lines.append("     WITHOUT TEMPORAL:")
            wo_res = rec.get("without_temporal", {})
            lines.append(f"       Label   : {wo_res.get('label', 'N/A')}")
            for r in (wo_res.get("results") or []):
                lines.append(f"       {r.get('metric', '')}: {r.get('value', '')}")

        # Delta
        deltas = rec.get("performance_delta", [])
        if deltas:
            lines.append("")
            lines.append("     PERFORMANCE DELTA (with − without):")
            for d in deltas:
                direction = d.get("direction", "")
                arrow = {"improvement": "↑", "degradation": "↓", "neutral": "→"}.get(
                    direction, "?"
                )
                lines.append(
                    f"       {d.get('metric', '')}: {d.get('delta', '')} {arrow}"
                )

        # Conclusion
        conclusion = row.get("conclusion_on_temporal")
        if conclusion:
            lines.append("")
            lines.append("     AUTHORS' CONCLUSION:")
            # Word-wrap at 75 chars
            words, cur = conclusion.split(), ""
            for w in words:
                if len(cur) + len(w) + 1 > 72:
                    lines.append(f"       {cur}")
                    cur = w
                else:
                    cur = f"{cur} {w}".strip()
            if cur:
                lines.append(f"       {cur}")

        lines.append("")
        lines.append("─" * 80)

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    log.info(f"Text report       → {report_path}")


def _print_console_summary(df, df_confirmed):
    total      = len(df)
    errors     = int(df["error"].notna().sum())
    processed  = total - errors
    n_temporal = int(df["temporal_component_present"].eq(True).sum())
    n_conf     = len(df_confirmed)

    def _tbl(rows, headers):
        print()
        print(tabulate(rows, headers=headers, tablefmt="simple_outline"))
        print()

    print("\n" + "=" * 60)
    print("  TEMPORAL ABLATION ANALYSIS — SUMMARY")
    print("=" * 60)

    _tbl(
        [
            ["Total articles",              total],
            ["Successfully processed",      processed],
            ["Extraction errors",           errors],
            ["Uses temporal component",     n_temporal],
            ["Confirmed ablation studies",  n_conf],
            ["Ablation rate (of temporal)", f"{100*n_conf/n_temporal:.1f}%" if n_temporal else "N/A"],
        ],
        ["", "Count"]
    )

    if n_conf > 0:
        print("─" * 60)
        print("  CONFIRMED ABLATION ARTICLES")
        rows = []
        for _, r in df_confirmed.iterrows():
            rows.append([
                r.get("title", "")[:50],
                r.get("year", ""),
                r.get("primary_metric", ""),
                r.get("with_temporal_primary_value",    ""),
                r.get("without_temporal_primary_value", ""),
            ])
        _tbl(rows, ["Title (truncated)", "Year", "Metric", "With Temp", "W/o Temp"])

        # Task distribution
        task_counts = df_confirmed["task"].value_counts(dropna=False)
        if not task_counts.empty:
            print("─" * 60)
            print("  TASK DISTRIBUTION (confirmed ablations)")
            _tbl(
                [[str(k), v] for k, v in task_counts.items()],
                ["Task", "Count"]
            )

        # Temporal mechanism distribution
        mech_counts = (
            df_confirmed["temporal_component_names"]
            .value_counts(dropna=False)
        )
        if not mech_counts.empty:
            print("─" * 60)
            print("  TEMPORAL MECHANISM (confirmed ablations)")
            _tbl(
                [[str(k), v] for k, v in mech_counts.items()],
                ["Component(s)", "Count"]
            )

    print("=" * 60)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    global MODEL

    parser = argparse.ArgumentParser(
        description="Stage 2 Temporal Ablation — identify and compare with/without temporal settings"
    )
    parser.add_argument(
        "--fulltext_dir",
        default=DEFAULT_FULLTEXT_DIR,
        help=f"Directory of .md full-text files (default: {DEFAULT_FULLTEXT_DIR})",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help=f"Output JSONL path (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--model",
        default=MODEL,
        help=f"Ollama model name (default: {MODEL})",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Start fresh — ignore any existing output file",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Skip extraction; re-generate summary from existing JSONL",
    )
    args = parser.parse_args()
    MODEL = args.model

    if args.summary_only:
        if not os.path.exists(args.output):
            print(f"Error: '{args.output}' not found.")
            sys.exit(1)
        # Rebuild CSV + report from existing JSONL
        records = []
        with open(args.output, encoding="utf-8") as f:
            for line in f:
                try:
                    records.append(_flatten_record(json.loads(line)))
                except Exception:
                    pass
        df           = pd.DataFrame(records)
        df_confirmed = df[df["confirmed_ablation"] == True].copy()
        _write_text_report(args.output, df_confirmed,
                           args.output.replace(".jsonl", "_report.txt"))
        _print_console_summary(df, df_confirmed)
    else:
        extract_all(
            fulltext_dir=args.fulltext_dir,
            output_jsonl=args.output,
            resume=not args.no_resume,
        )


if __name__ == "__main__":
    main()