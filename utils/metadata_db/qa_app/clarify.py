"""
Pre-flight clarification gate for the NL→SQL query engine.

Before the agent executes anything, `assess_clarification` makes a single
lightweight LLM call that judges whether the user's question is specific enough
to answer confidently. If it is genuinely ambiguous, it returns a structured
clarifying question plus a few likely interpretations (the UI always appends an
"Other" free-text option). Otherwise it returns ``None`` and the normal agent
pipeline runs unchanged.

Design goals:
- Conservative: strongly biased toward proceeding (fail-open). Any parsing or
  LLM error => return None => the pipeline behaves exactly as before.
- Minimal footprint: reuses the existing `llm_call` helper and the configured
  Ollama model; no new dependencies and no changes to the execution pipeline.
"""

import json
import logging
import re
from typing import Any, Dict, List, Optional

from .llm import llm_call
from .prompts import CLARIFY_SYSTEM

log = logging.getLogger(__name__)

MAX_OPTIONS = 4
_OTHER_LABELS = {"other", "others", "other (please specify)", "other (specify)", "something else"}


def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    """Best-effort extraction of a single JSON object from an LLM response."""
    if not text:
        return None
    # 1) Direct parse.
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass
    # 2) First {...} block (handles models that wrap JSON in prose/fences).
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        obj = json.loads(match.group(0))
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"true", "yes", "y", "1"}
    return False


def _sanitize_options(raw: Any) -> List[str]:
    """Normalize the model's option list: strip, drop 'Other'/blanks, dedupe, cap."""
    if isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, (list, tuple)):
        return []
    options: List[str] = []
    for item in raw:
        label = str(item).strip().strip("-•").strip()
        if not label or label.lower() in _OTHER_LABELS:
            continue
        if label not in options:
            options.append(label)
        if len(options) >= MAX_OPTIONS:
            break
    return options


def assess_clarification(question: str) -> Optional[Dict[str, Any]]:
    """
    Decide whether `question` needs clarification before execution.

    Returns:
        None  -> proceed with the normal agent pipeline (clear enough, off-topic,
                 or any failure — fail-open).
        dict  -> {"question": str, "options": List[str], "reason": str} when a
                 single clarifying question is warranted.
    """
    q = (question or "").strip()
    if not q:
        return None

    try:
        raw = llm_call(
            [
                {"role": "system", "content": CLARIFY_SYSTEM},
                {
                    "role": "user",
                    "content": (
                        f'User question:\n"""\n{q}\n"""\n\n'
                        "Respond with ONLY the JSON object."
                    ),
                },
            ],
            think=False,  # fast, no chain-of-thought needed for this classifier
        )
    except Exception as exc:  # LLM unreachable, model error, etc. -> proceed.
        log.warning(f"clarification check failed, proceeding without it: {exc}")
        return None

    data = _extract_json(raw)
    if not isinstance(data, dict):
        log.debug("clarifier returned non-JSON; proceeding without clarification")
        return None

    if not _as_bool(data.get("needs_clarification")):
        return None

    clarifying_question = str(data.get("question", "")).strip()
    if not clarifying_question:
        # Model flagged ambiguity but gave no question — don't block the user.
        return None

    return {
        "question": clarifying_question,
        "options": _sanitize_options(data.get("options")),
        "reason": str(data.get("reason", "")).strip(),
    }
