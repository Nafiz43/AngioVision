import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import requests
from tqdm import tqdm

# -----------------------------
# Defaults / Configuration
# -----------------------------
DEFAULT_CSV_PATH = Path("/data/Deep_Angiography/Reports/Report_List_v01_01.csv")
REPORT_COL = "radrpt"

DEFAULT_OLLAMA_URL = "http://localhost:11434/api/chat"
DEFAULT_MODEL_NAME = "qwen2.5vl:32b"
DEFAULT_TIMEOUT_S = 120

QUESTIONS = [
    "Which artery is catheterized?",
    "Is variant anatomy present?",
    "Is there evidence of hemorrhage or contrast extravasation in this sequence?",
    "Is there evidence of arterial or venous dissection?",
    "Is stenosis present in any visualized vessel?",
    "Is an endovascular stent visible in this sequence?",
]

BASE_PROMPT = """ROLE
You are a meticulous clinical information extraction engine for radiology/interventional radiology narrative reports. Your job is to convert unstructured report text into a high-precision, audit-friendly answer to a single targeted question.

WHY THIS MATTERS (INCENTIVE)
Your output will be used to build a research-grade labeled dataset. High precision is more important than guessing.

SOURCE OF TRUTH
Use ONLY the provided report text.

TASK
Answer exactly ONE question:
Question: {QUESTION}

OUTPUT FORMAT (JSON ONLY)
Return:
- answer
- confidence (0–100)
- evidence (≤3 short excerpts)
- notes

CONTEXT:
<<<REPORT_TEXT_START>>>
{REPORT_TEXT}
<<<REPORT_TEXT_END>>>
"""


# -----------------------------
# Helpers
# -----------------------------
def build_prompt(report_text: str, question: str) -> str:
    return BASE_PROMPT.format(QUESTION=question, REPORT_TEXT=report_text)


def ollama_chat(
    prompt: str,
    model: str,
    url: str,
    timeout_s: int,
) -> str:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {"temperature": 0},
    }

    r = requests.post(url, json=payload, timeout=timeout_s)
    r.raise_for_status()
    return r.json().get("message", {}).get("content", "")


def safe_parse_json(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None

    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass

    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except Exception:
            return None
    return None


def ensure_csv_header(path: Path, columns):
    if path.exists() and path.stat().st_size > 0:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(columns=columns).to_csv(path, index=False)


def append_row(path: Path, row: Dict[str, Any], columns):
    ordered = {c: row.get(c) for c in columns}
    pd.DataFrame([ordered]).to_csv(
        path, mode="a", header=False, index=False
    )


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV_PATH)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT_S)
    parser.add_argument("--delay", type=float, default=0.0)

    args = parser.parse_args()

    if not args.csv.exists():
        raise FileNotFoundError(args.csv)

    df = pd.read_csv(args.csv)

    if REPORT_COL not in df.columns:
        raise ValueError(f"Missing column: {REPORT_COL}")

    if args.limit is not None:
        df = df.iloc[: args.limit]

    out_path = args.out or args.csv.with_name(
        f"{args.csv.stem}_extracted_labels.csv"
    )

    out_cols = ["row_index", "question", "answer", "confidence", "evidence", "notes"]
    ensure_csv_header(out_path, out_cols)

    total_tasks = len(df) * len(QUESTIONS)

    print(f"Processing {len(df)} rows → {total_tasks} questions")
    print(f"Output: {out_path}")

    with tqdm(total=total_tasks, desc="Extracting", unit="q") as pbar:
        for idx, row in df.iterrows():
            report = row.get(REPORT_COL)

            if not isinstance(report, str) or not report.strip():
                for q in QUESTIONS:
                    append_row(
                        out_path,
                        {
                            "row_index": idx,
                            "question": q,
                            "answer": "Not stated",
                            "confidence": 0,
                            "evidence": "[]",
                            "notes": "Empty or missing report",
                        },
                        out_cols,
                    )
                    pbar.update(1)
                continue

            for q in QUESTIONS:
                try:
                    prompt = build_prompt(report, q)
                    raw = ollama_chat(
                        prompt,
                        model=args.model,
                        url=DEFAULT_OLLAMA_URL,
                        timeout_s=args.timeout,
                    )
                    parsed = safe_parse_json(raw)

                    if not parsed:
                        raise ValueError("Invalid JSON")

                    append_row(
                        out_path,
                        {
                            "row_index": idx,
                            "question": q,
                            "answer": parsed.get("answer"),
                            "confidence": parsed.get("confidence"),
                            "evidence": json.dumps(parsed.get("evidence", [])),
                            "notes": parsed.get("notes", ""),
                        },
                        out_cols,
                    )

                except Exception as e:
                    append_row(
                        out_path,
                        {
                            "row_index": idx,
                            "question": q,
                            "answer": "Unclear",
                            "confidence": 0,
                            "evidence": "[]",
                            "notes": f"Error: {str(e)[:200]}",
                        },
                        out_cols,
                    )

                pbar.update(1)
                if args.delay > 0:
                    time.sleep(args.delay)

    print("Done ✔ Partial results are always preserved.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted — output CSV already contains completed rows.", file=sys.stderr)
        raise
