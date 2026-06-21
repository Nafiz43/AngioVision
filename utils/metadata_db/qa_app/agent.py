"""
Agentic NL→SQL pipeline (smolagents).

A smolagents ToolCallingAgent can call the `sql_query` tool as many times as it
needs — exploring the schema, checking real column values, recovering from SQL
errors, and refining its query — before producing a final natural-language
answer. The agent talks to whichever local Ollama model is selected in the UI,
via Ollama's OpenAI-compatible endpoint.
"""

import re
import json
import queue
import logging
import threading
import traceback
from typing import Any, Callable, Dict, List, Optional

from . import config
from .deps import SMOLAGENTS_OK, Tool, ToolCallingAgent, OpenAIServerModel
from .state import state
from .db import run_sql_query, clean_sql
from .prompts import SCHEMA_CONTEXT, SYNTHESIS_SYSTEM

log = logging.getLogger(__name__)


class SQLQueryTool(Tool):
    """
    A smolagents Tool enabling the agent to query the DICOM SQLite database.

    Allows the agent to execute read-only SELECT statements as many times as needed,
    exploring the schema, checking actual values, and recovering from SQL errors
    before producing a final answer.
    """

    name = "sql_query"
    description = (
        "Run a single read-only SQLite SELECT statement against the DICOM imaging "
        "database (tables: dicom_files, radiology_reports, image_ingestion_status) "
        "and get the matching rows back as JSON. You can call this tool multiple "
        "times in a row: explore first if you're unsure about a column's actual "
        "values (e.g. SELECT DISTINCT modality FROM dicom_files LIMIT 10), then "
        "write the real query. If a query fails, read the SQL ERROR message, fix "
        "the query, and call this tool again — never give up after one failed "
        "attempt."
    )
    inputs = {
        "query": {
            "type": "string",
            "description": "A single SQLite SELECT statement (no trailing semicolon-separated statements).",
        }
    }
    output_type = "string"

    def __init__(
        self,
        on_step_callback: Optional[Callable[[str, Optional[List[Dict[str, Any]]], int], None]] = None,
        max_rows: int = config.MAX_ROWS_FOR_SYNTHESIS,
    ) -> None:
        """
        Args:
            on_step_callback: Optional callback(sql, rows_or_None, row_count, error=None)
                             called after each query execution for streaming events
            max_rows: Maximum rows to return in payload (others truncated but counted)
        """
        super().__init__()
        self.on_step_callback = on_step_callback
        self.max_rows = max_rows

    def forward(self, query: str) -> str:
        """Execute a SQL query and return results as JSON."""
        sql = clean_sql(query)

        try:
            rows = run_sql_query(sql)
        except Exception as exc:
            err = str(exc)
            if self.on_step_callback:
                self.on_step_callback(sql, None, 0, error=err)
            return (
                f"SQL ERROR: {err}\n\n"
                "Fix the query and call sql_query again. Reminders: every column in "
                "dicom_files is stored as TEXT — CAST numeric columns explicitly "
                "(e.g. CAST(frame_count AS INTEGER)); radrpt lives in "
                "radiology_reports, not dicom_files; join the two with "
                "JOIN radiology_reports USING (accession_number)."
            )

        def serialize_value(v: Any) -> Any:
            """Safely serialize a value to JSON-compatible format."""
            if v is None:
                return None
            try:
                json.dumps(v)
                return v
            except Exception:
                return str(v)

        clean_rows = [{k: serialize_value(v) for k, v in row.items()} for row in rows]
        if self.on_step_callback:
            self.on_step_callback(sql, clean_rows, len(clean_rows))

        display = clean_rows[: self.max_rows]
        payload: Dict[str, Any] = {"row_count": len(clean_rows), "rows": display}
        if len(clean_rows) > self.max_rows:
            payload["note"] = f"{len(clean_rows) - self.max_rows} additional rows truncated."
        elif not clean_rows:
            payload["note"] = "Query returned zero rows — consider relaxing filters or checking actual column values."
        return json.dumps(payload, default=str)


def get_smolagents_model(model_name: str) -> "OpenAIServerModel":
    """Build a smolagents model that talks to the local Ollama server via its
    OpenAI-compatible API, so the agent uses whatever model is selected in the UI."""
    if not SMOLAGENTS_OK:
        raise RuntimeError("smolagents is not installed. Run: pip install 'smolagents[openai]'")
    return OpenAIServerModel(
        model_id=model_name,
        api_base=f"{state.ollama_host}/v1",
        api_key="ollama",   # Ollama ignores the key but the OpenAI client requires one
    )


def build_agent_task(question: str, think: bool) -> str:
    """Compose the full task text given to the agent: schema, SQL rules, answer
    style, and the user's question."""
    prefix = "" if think else "/no_think\n"
    return f"""{prefix}=== IMPORTANT INSTRUCTIONS ===

1. OFF-TOPIC FILTER:
   If the user's question is clearly NOT about the DICOM database (e.g., "Hi",
   "Hello", "What's the weather?"), respond politely WITHOUT using sql_query.

2. IMAGE / SEQUENCE REQUESTS:
   When the user asks to "show", "display", or "see" images, sequences, or cases:
   - ALWAYS include source_path in your SELECT — the UI renders thumbnail
     previews automatically from .dcm file paths in the results table.
   - Also include: frame_count, series_description, modality, study_date,
     accession_number so the user gets useful context alongside the images.

3. CLINICAL FINDING WORKFLOWS:
   When the user asks about a clinical procedure or finding (e.g., "TIPS",
   "stenosis", "embolization", "angioplasty"), follow this multi-step approach:

   Step 1 — SEARCH REPORTS: Find matching radiology reports first.
     SELECT accession_number, SUBSTR(radrpt, 1, 200) AS radrpt_excerpt
     FROM radiology_reports
     WHERE LOWER(radrpt) LIKE '%tips%'
     LIMIT 5

   Step 2 — FETCH SEQUENCES: Use the accession numbers from Step 1 to query
     DICOM sequences, including source_path for image display.
     SELECT source_path, frame_count, series_description, modality,
            study_date, accession_number
     FROM dicom_files
     WHERE accession_number IN ('acc1', 'acc2', ...)
       AND parse_error IS NULL
     ORDER BY CAST(frame_count AS INTEGER) DESC
     LIMIT 20

   This two-step approach (reports → accessions → sequences) is the correct
   workflow for any clinical finding query.

==============================

{SCHEMA_CONTEXT}

You are a clinical informatics assistant answering questions about a DICOM
angiography database using the sql_query tool. Some questions need more than
one query to answer correctly — explore the data first if you're unsure about
something (e.g. check distinct values, run a small LIMIT 5 sample), then
refine. Use the tool as many times as you need, and recover from SQL errors by
fixing the query and trying again, before giving your final answer.

{SYNTHESIS_SYSTEM}

Question: {question}

When you are confident in your answer, call final_answer with the plain-text
response described above.
"""


def run_nl_query_agent(question: str, think: bool, model_name: str) -> Any:
    """
    Run a smolagents ToolCallingAgent for NL→SQL query resolution.

    Emits rich SSE-compatible event dictionaries so the frontend can show exactly
    what the agent is doing at every step — which SQL it tried, how many rows came
    back, whether it hit an error, and what it's thinking.

    New events (in addition to the original protocol):
        agent_start   — agent is initializing (model, max_steps)
        agent_step    — one sql_query tool call completed (step#, sql,
                        row_count, error, max_steps)

    Original events (preserved for frontend compatibility):
        sql_done / sql_repaired, exec_start, exec_done,
        synth_start, answer, error
    """
    event_queue: "queue.Queue[Dict[str, Any]]" = queue.Queue()
    call_count = {"n": 0}

    def on_query_step(
        sql: str,
        rows: Optional[List[Dict[str, Any]]],
        row_count: int,
        error: Optional[str] = None,
    ) -> None:
        """Callback invoked after each sql_query tool call."""
        call_count["n"] += 1

        # ── rich step event so the UI can show a step-by-step log ──
        event_queue.put({
            "event":     "agent_step",
            "step":      call_count["n"],
            "max_steps": state.agent_max_steps,
            "sql":       sql,
            "row_count": row_count,
            "error":     error,
        })

        # ── Original events (kept for backward compat) ──────────────────
        event_queue.put({
            "event": "sql_done" if call_count["n"] == 1 else "sql_repaired",
            "sql": sql,
        })
        event_queue.put({"event": "exec_start"})
        event_queue.put({
            "event": "exec_done",
            "rows": rows or [],
            "row_count": row_count,
        })

    result_holder: Dict[str, Optional[str]] = {"answer": None, "error": None}

    def worker() -> None:
        """Background thread running the smolagents agent."""
        try:
            # ── Notify UI that the agent is starting ────────────────────
            event_queue.put({
                "event":     "agent_start",
                "model":     model_name,
                "max_steps": state.agent_max_steps,
            })

            model = get_smolagents_model(model_name)
            tool  = SQLQueryTool(on_step_callback=on_query_step)
            agent = ToolCallingAgent(
                tools=[tool], model=model, max_steps=state.agent_max_steps,
            )
            task = build_agent_task(question, think)

            log.info(
                f"Agent starting: model={model_name}, "
                f"max_steps={state.agent_max_steps}, q={question!r:.80}"
            )

            raw_answer = agent.run(task)
            answer = re.sub(
                r"<think>.*?</think>", "", str(raw_answer), flags=re.DOTALL,
            ).strip()
            result_holder["answer"] = answer

            log.info(
                f"Agent finished: {call_count['n']} tool call(s), "
                f"answer length={len(answer)}"
            )

        except Exception as exc:
            tb = traceback.format_exc()
            log.error(f"Agent execution failed:\n{tb}")
            result_holder["error"] = f"{exc}\n\n--- traceback ---\n{tb}"
        finally:
            event_queue.put({"event": "__agent_done__"})

    threading.Thread(target=worker, daemon=True).start()

    # ── Yield events as the agent emits them ────────────────────────────
    while True:
        item = event_queue.get()
        if item["event"] == "__agent_done__":
            break
        yield item

    # ── Final outcome ───────────────────────────────────────────────────
    if result_holder["error"]:
        yield {
            "event":      "error",
            "message":    f"Agent failed: {result_holder['error']}",
            "tool_calls": call_count["n"],
        }
    else:
        yield {"event": "synth_start"}
        final_answer = (
            result_holder["answer"]
            or "(The agent did not produce an answer.)"
        )
        yield {"event": "answer", "text": final_answer}
