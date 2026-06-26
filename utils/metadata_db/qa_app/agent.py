"""
Agentic NL→SQL pipeline (smolagents).

A smolagents ToolCallingAgent can call the `sql_query` tool as many times as it
needs — exploring the schema, checking real column values, recovering from SQL
errors, and refining its query — before producing a final natural-language
answer. When the user asks for a visual, it can also call the `render_chart`
tool to draw a bar / line / pie chart in the browser (streamed to the frontend
as a `chart` SSE event and rendered by a self-contained canvas renderer). The
agent talks to whichever local Ollama model is selected in the UI, via Ollama's
OpenAI-compatible endpoint.
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
from .clarify import assess_clarification

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


# ── Chart / graph rendering tool ─────────────────────────────────────────────
ALLOWED_CHART_TYPES = {"bar", "horizontal_bar", "line", "pie", "doughnut"}


def _coerce_list(x: Any) -> List[Any]:
    """Best-effort coercion of a tool argument into a Python list.

    Smaller models sometimes pass a JSON string (e.g. "[1, 2, 3]") or a plain
    comma-separated string instead of a real array — accept all of these.
    """
    if x is None:
        return []
    if isinstance(x, list):
        return x
    if isinstance(x, tuple):
        return list(x)
    if isinstance(x, (str, bytes)):
        s = x.decode() if isinstance(x, bytes) else x
        s = s.strip()
        if not s:
            return []
        try:
            parsed = json.loads(s)
            return parsed if isinstance(parsed, list) else [parsed]
        except Exception:
            return [p.strip() for p in s.split(",") if p.strip() != ""]
    return [x]


class ChartTool(Tool):
    """A smolagents Tool that renders a chart in the user's browser.

    The tool does not draw anything itself: it validates a compact chart
    specification and pushes it to the frontend through a callback (an SSE
    ``chart`` event). The browser renders it with a self-contained canvas
    chart renderer. Use it only when the user asks for a visual (plot, graph,
    chart, distribution, trend, breakdown, …); fetch the data with sql_query
    first, then call this with aligned ``labels`` and ``values``.
    """

    name = "render_chart"
    description = (
        "Render a chart/graph in the user's browser to visualize data. Call this "
        "ONLY when the user asks to plot, graph, chart, visualize, or see a "
        "distribution / trend / breakdown / histogram. Workflow: first get the data "
        "with sql_query (usually a GROUP BY giving one category + one number per "
        "row), then call render_chart with `labels` and `values` of EQUAL length and "
        "in the same order. Choose chart_type: 'bar' (counts by category), 'line' "
        "(trend over an ordered x such as year), 'pie' or 'doughnut' (share of a "
        "whole), or 'horizontal_bar' (many or long category names). After it "
        "returns, call final_answer with a short plain-text summary. Example: "
        "render_chart(chart_type='bar', title='Studies per year', "
        "labels=['2009','2010','2011'], values=[120, 156, 143], series_label='Studies')."
    )
    inputs = {
        "chart_type": {
            "type": "string",
            "description": "One of: bar, horizontal_bar, line, pie, doughnut.",
        },
        "title": {
            "type": "string",
            "description": "Short, descriptive chart title.",
        },
        "labels": {
            "type": "array",
            "description": (
                "Category label for each data point (x-axis tick or pie slice), as a "
                "list of strings. Must be the same length as values and in the same order."
            ),
        },
        "values": {
            "type": "array",
            "description": (
                "Numeric value for each label, as a list of numbers. Must be the same "
                "length as labels and in the same order."
            ),
        },
        "series_label": {
            "type": "string",
            "description": "What the numbers represent (axis / legend caption), e.g. 'Studies'.",
            "nullable": True,
        },
    }
    output_type = "string"

    def __init__(
        self,
        on_chart_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        max_points: int = 40,
    ) -> None:
        """
        Args:
            on_chart_callback: callback(spec) invoked with the validated chart spec
                               so the server can stream a `chart` SSE event.
            max_points: hard cap on the number of plotted data points.
        """
        super().__init__()
        self.on_chart_callback = on_chart_callback
        self.max_points = max_points
        self._emitted = False  # render only the first chart per question

    def forward(
        self,
        chart_type: str,
        title: str,
        labels: Any,
        values: Any,
        series_label: Optional[str] = None,
    ) -> str:
        """Validate a chart spec and push it to the frontend."""
        ctype = str(chart_type or "bar").strip().lower().replace("-", "_").replace(" ", "_")
        if ctype in ("hbar", "barh", "horizontalbar", "horizontal"):
            ctype = "horizontal_bar"
        if ctype == "donut":
            ctype = "doughnut"
        if ctype not in ALLOWED_CHART_TYPES:
            ctype = "bar"

        labels_l = _coerce_list(labels)
        values_l = _coerce_list(values)

        norm_vals: List[float] = []
        for v in values_l:
            try:
                norm_vals.append(float(str(v).replace(",", "").strip()))
            except Exception:
                norm_vals.append(0.0)

        n = min(len(labels_l), len(norm_vals))
        if n == 0:
            return (
                "CHART ERROR: labels and values must be non-empty lists of equal "
                "length. Run sql_query to get the data first (one category and one "
                "number per row), then pass labels=[...] and values=[...] of the "
                "same length to render_chart."
            )

        str_labels = [str(x) for x in labels_l[:n]]
        norm_vals = norm_vals[:n]

        truncated = 0
        if n > self.max_points:
            truncated = n - self.max_points
            str_labels = str_labels[: self.max_points]
            norm_vals = norm_vals[: self.max_points]

        spec: Dict[str, Any] = {
            "chart_type":   ctype,
            "title":        str(title or "Chart"),
            "labels":       str_labels,
            "values":       norm_vals,
            "series_label": str(series_label) if series_label else "",
        }
        # One chart per question: smaller models sometimes call render_chart
        # repeatedly in their reasoning loop (often with a second, mislabeled
        # query), which stacked duplicate charts under the same title. Emit the
        # first valid chart only and steer the agent to finish.
        if self._emitted:
            return (
                "A chart has ALREADY been rendered for this question. Do NOT call "
                "render_chart again. Call final_answer now with a 1-2 sentence "
                "plain-text summary of what the chart shows."
            )

        if self.on_chart_callback:
            self.on_chart_callback(spec)
        self._emitted = True

        msg = (
            f"Chart rendered in the UI: a {ctype.replace('_', ' ')} titled "
            f"'{spec['title']}' with {len(str_labels)} data points. Now call "
            f"final_answer with a 1-2 sentence plain-text summary of what the chart shows."
        )
        if truncated:
            msg += f" (Only the first {self.max_points} of {n} points were plotted.)"
        return msg


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

4. VISUALIZATION / CHART REQUESTS:
   If the user asks to "plot", "graph", "chart", "visualize", "draw", or to see
   a "distribution", "trend", "breakdown", "histogram", something "over time",
   or a "bar/line/pie chart":

   Step 1 -- GET DATA: run sql_query to return rows of (category, number) --
     usually a GROUP BY with a COUNT/SUM/AVG and an ORDER BY. Example:
       SELECT SUBSTR(study_date, 1, 4) AS yr,
              COUNT(DISTINCT study_instance_uid) AS n
       FROM dicom_files
       WHERE parse_error IS NULL AND study_date IS NOT NULL
       GROUP BY yr ORDER BY yr

   Step 2 -- DRAW: call render_chart with
       chart_type   : 'bar' (counts by category), 'line' (trend over an ordered
                      x such as year), 'pie'/'doughnut' (share of a whole), or
                      'horizontal_bar' (many or long category names)
       title        : a short descriptive title
       labels       : the category for each row, as a list (e.g. the yr values)
       values       : the matching number for each row, as a list (e.g. the n values)
       series_label : what the numbers represent (e.g. 'Studies')

   labels and values MUST be the same length and in the same order. Keep it to
   ~30 categories or fewer; if there could be more, add ORDER BY <number> DESC
   LIMIT 30 and chart the top ones. After render_chart succeeds, call
   final_answer with a 1-2 sentence summary. Do NOT call render_chart for
   ordinary questions -- only when the user asks for a visual.

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


def run_nl_query_agent(
    question: str, think: bool, model_name: str, clarify_enabled: bool = True,
) -> Any:
    """
    Run a smolagents ToolCallingAgent for NL→SQL query resolution.

    Emits rich SSE-compatible event dictionaries so the frontend can show exactly
    what the agent is doing at every step — which SQL it tried, how many rows came
    back, whether it hit an error, and what it's thinking.

    New events (in addition to the original protocol):
        agent_start   — agent is initializing (model, max_steps)
        agent_step    — one sql_query tool call completed (step#, sql,
                        row_count, error, max_steps)
        chart         — the agent rendered a chart (spec: {chart_type, title,
                        labels, values, series_label})

    Original events (preserved for frontend compatibility):
        sql_done / sql_repaired, exec_start, exec_done,
        synth_start, answer, error
    """
    # ── Pre-flight clarification gate ────────────────────────────────────────
    # Before running ANY tool, decide whether the request is too ambiguous to
    # answer confidently. If so, emit a single `clarification` event (with
    # selectable options) and stop — the frontend collects the user's choice and
    # re-submits the augmented question with skip_clarify=True, so this runs at
    # most once per question and the execution pipeline below stays untouched.
    if clarify_enabled:
        yield {"event": "clarify_check"}
        clar = assess_clarification(question)
        if clar:
            log.info(
                f"Clarification requested (q={question!r:.80}): {clar['question']!r}"
            )
            yield {
                "event":        "clarification",
                "question":     clar["question"],
                "options":      clar["options"],
                "allow_custom": True,
                "reason":       clar.get("reason", ""),
            }
            return

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

    def on_chart(spec: Dict[str, Any]) -> None:
        """Callback invoked when the agent renders a chart — emits a `chart` event."""
        event_queue.put({"event": "chart", "spec": spec})

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

            model      = get_smolagents_model(model_name)
            sql_tool   = SQLQueryTool(on_step_callback=on_query_step)
            chart_tool = ChartTool(on_chart_callback=on_chart)
            agent = ToolCallingAgent(
                tools=[sql_tool, chart_tool], model=model,
                max_steps=state.agent_max_steps,
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
