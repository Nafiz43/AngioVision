"""Text normalisation, answer normalisation, JSON parsing and timestamps."""

import json
import re
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import pandas as pd


# ---------- Text normalisation ----------

def normalize_text(x: object) -> str:
    """Lower-case, collapse whitespace, strip."""
    if pd.isna(x):
        return ""
    s = str(x).strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


# ---------- YES / NO answer normalisation ----------

_YES_LIKE = {"yes", "y", "true", "1", "present", "positive"}
_NO_LIKE = {
    "no", "n", "false", "0", "absent", "negative",
    "unclear", "not visible", "cannot say", "can not say",
    "can't say", "cant say", "unable to determine",
    "cannot determine", "can not determine", "not sure",
    "unknown", "indeterminate", "not identifiable",
    "not seen", "not clear", "not possible to determine",
}

_NO_TERMS_LIST = [
    "no",
    "unclear",
    "not visible",
    "cannot say",
    "can not say",
    "can't say",
    "cant say",
    "unable to determine",
    "cannot determine",
    "can not determine",
    "not sure",
    "unknown",
    "indeterminate",
    "not identifiable",
    "not seen",
    "not clear",
    "not possible to determine",
    "absent",
    "negative",
]


def normalize_gt_answer(raw_answer: object) -> str:
    """Map a ground-truth answer to ``YES`` or ``NO``."""
    if pd.isna(raw_answer):
        return "NO"

    ans = normalize_text(raw_answer)

    if ans in _YES_LIKE:
        return "YES"
    if ans in _NO_LIKE:
        return "NO"

    if "yes" in ans or "present" in ans or "positive" in ans:
        return "YES"

    if any(term in ans for term in _NO_TERMS_LIST):
        return "NO"

    return "NO"


def normalize_llm_answer(raw_answer: object) -> str:
    """Map a raw LLM answer string to ``YES`` or ``NO``."""
    if raw_answer is None:
        return "NO"

    ans = normalize_text(raw_answer)

    yes_terms = {"yes", "y"}
    no_terms = {
        "no", "n",
        "unclear", "not visible", "cannot say", "can not say",
        "cant say", "can't say",
        "unable to determine", "not sure", "unknown", "indeterminate",
        "not identifiable", "not seen", "not clear",
        "not possible to determine", "cannot determine", "can not determine",
    }

    if ans in yes_terms:
        return "YES"
    if ans in no_terms:
        return "NO"
    if "yes" in ans:
        return "YES"

    if any(term in ans for term in _NO_TERMS_LIST):
        return "NO"

    return "NO"


# ---------- JSON parsing ----------

def safe_parse_json(text: str) -> Optional[Dict[str, Any]]:
    """Parse JSON robustly, tolerating extra wrapper text from LLMs."""
    if text is None:
        return None
    t = text.strip()
    try:
        return json.loads(t)
    except Exception:
        pass

    start, end = t.find("{"), t.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(t[start : end + 1])
        except Exception:
            pass

    start, end = t.find("["), t.rfind("]")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(t[start : end + 1])
        except Exception:
            pass

    return None


# ---------- Timestamps ----------

def utc_timestamp() -> str:
    """Return the current UTC time as ``YYYY-MM-DDTHH:MM:SSZ``."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def now_ts() -> str:
    """Return the current local time as ``YYYY-MM-DD HH:MM:SS``."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
