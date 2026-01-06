import json
import time
from pathlib import Path

import pandas as pd
import requests

# -----------------------------
# Configuration
# -----------------------------
CSV_PATH = Path("/data/Deep_Angiography/Reports/Report_List_v01_01.csv")
REPORT_COL = "radrpt"

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL_NAME = "llama3.1"  # change to the model you actually have pulled in Ollama
REQUEST_TIMEOUT_S = 120

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
Your output will be used to build a research-grade labeled dataset. High precision is more important than guessing. Incorrect labels harm downstream modeling and clinical validity. You are rewarded for:
- Faithfulness to the report (no hallucinations)
- Clear uncertainty handling
- Citing short supporting evidence from the report

SOURCE OF TRUTH
Use ONLY the provided report text. Do not use outside medical knowledge. If the report does not clearly contain the information, say so.

TASK
Answer exactly ONE question (provided below) using the report as context:
Question: {QUESTION}

STRICT RULES
1) Do not guess. If not stated or unclear, return “Not stated” or “Unclear” (choose the most accurate).
2) If the report contains conflicting statements, report “Conflicting” and briefly describe both.
3) When possible, extract the most specific entity (e.g., “right hepatic artery” vs “hepatic artery”).
4) If the question depends on which “sequence” is being referred to and the report does not segment by sequence, state that limitation explicitly.
5) Keep evidence excerpts short (a few words to one sentence). Do not quote large passages.

OUTPUT FORMAT (JSON ONLY)
Return a single JSON object with these keys:
- "answer": one of ["Yes", "No", "Not stated", "Unclear", "Conflicting"] OR a short free-text entity when the question asks “Which/What” (e.g., artery name). If multiple entities apply, return a list of strings.
- "confidence": integer 0–100 reflecting how directly the report supports the answer.
- "evidence": an array of up to 3 brief verbatim excerpts from the report that support the answer (empty if "Not stated").
- "notes": one short sentence describing any ambiguity, assumptions avoided, or conflicts.

CONTEXT: RADIOLOGY REPORT
<<<REPORT_TEXT_START>>>
{REPORT_TEXT}
<<<REPORT_TEXT_END>>>
"""


def build_prompt(report_text: str, question: str) -> str:
    return BASE_PROMPT.format(QUESTION=question, REPORT_TEXT=report_text)


def ollama_chat(prompt: str, model: str = MODEL_NAME) -> str:
    """
    Calls Ollama /api/chat and returns the assistant message content as text.
    """
    payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "stream": False,
        "options": {
            "temperature": 0,
        }
    }

    r = requests.post(OLLAMA_URL, json=payload, timeout=REQUEST_TIMEOUT_S)
    r.raise_for_status()
    data = r.json()

    # Ollama returns: {"message": {"role": "assistant", "content": "..."} , ...}
    return data.get("message", {}).get("content", "")


def safe_parse_json(text: str):
    """
    Attempts to parse JSON from the model response.
    If the model wraps JSON in extra text, try to extract the first {...} block.
    """
    text = text.strip()

    # Fast path
    try:
        return json.loads(text)
    except Exception:
        pass

    # Try to extract first JSON object
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = text[start:end+1]
        try:
            return json.loads(candidate)
        except Exception:
            return None

    return None


def main():
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"CSV not found: {CSV_PATH}")

    df = pd.read_csv(CSV_PATH)

    if REPORT_COL not in df.columns:
        raise ValueError(f"Column '{REPORT_COL}' not found. Available columns: {list(df.columns)}")

    results = []
    for idx, row in df.iterrows():
        report_text = row.get(REPORT_COL)
        if not isinstance(report_text, str) or not report_text.strip():
            # Keep a placeholder record so you can track missing reports
            for q in QUESTIONS:
                results.append({
                    "row_index": idx,
                    "question": q,
                    "answer": "Not stated",
                    "confidence": 0,
                    "evidence": [],
                    "notes": "Empty or missing report text."
                })
            continue

        for q in QUESTIONS:
            prompt = build_prompt(report_text, q)

            # Optional: small delay to be gentle on local inference
            # time.sleep(0.05)

            raw = ollama_chat(prompt)
            parsed = safe_parse_json(raw)

            if parsed is None:
                # If the model didn't comply, record raw output for debugging
                results.append({
                    "row_index": idx,
                    "question": q,
                    "answer": "Unclear",
                    "confidence": 0,
                    "evidence": [],
                    "notes": f"Model response was not valid JSON. Raw: {raw[:300]}..."
                })
                continue

            results.append({
                "row_index": idx,
                "question": q,
                "answer": parsed.get("answer"),
                "confidence": parsed.get("confidence"),
                "evidence": parsed.get("evidence", []),
                "notes": parsed.get("notes", "")
            })

    out_df = pd.DataFrame(results)
    out_path = CSV_PATH.with_name("Report_List_v01_01_extracted_labels.csv")
    out_df.to_csv(out_path, index=False)

    print(f"Done. Wrote results to: {out_path}")


if __name__ == "__main__":
    main()
