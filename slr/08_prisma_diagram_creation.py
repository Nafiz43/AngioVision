"""
PRISMA 2020 Flow Diagram — v5
==============================
Reads counts directly from:
  1. slr_fetching_stats.csv          — identification / deduplication numbers
  2. stage1_results_criteria_summary.csv — per-criterion inclusion/exclusion counts

Usage:
    python prisma_v5.py \
        --fetch  /data/Deep_Angiography/AngioVision/slr/results/slr_fetching_stats.csv \
        --stage1 /data/Deep_Angiography/AngioVision/slr/results/stage1_results_criteria_summary.csv \
        --out    prisma_2020          # output filename (no extension)

Dependencies:
    pip install graphviz pandas
    sudo apt install graphviz   # or: brew install graphviz
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import graphviz


# ══════════════════════════════════════════════════════════════════════════════
# CSV LOADERS
# ══════════════════════════════════════════════════════════════════════════════

def load_fetch_stats(path: str) -> dict:
    """
    Parse slr_fetching_stats.csv.

    Expected columns:
        run_timestamp, source, before_dedup, after_dedup, duplicates_removed

    Returns a dict with keys:
        db_total, after_dedup, duplicates,
        sources: {source_name: before_dedup_count, ...}   (excluding TOTAL row)
    """
    df = pd.read_csv(path, skipinitialspace=True)
    df.columns = df.columns.str.strip()
    df["source"] = df["source"].str.strip()

    total_row = df[df["source"].str.upper() == "TOTAL"].iloc[0]
    source_rows = df[df["source"].str.upper() != "TOTAL"].copy()

    return {
        "db_total":    int(total_row["before_dedup"]),
        "after_dedup": int(total_row["after_dedup"]),
        "duplicates":  int(total_row["duplicates_removed"]),
        # preserve source order from file
        "sources": {
            row["source"]: int(row["before_dedup"])
            for _, row in source_rows.iterrows()
        },
    }


def load_criteria_summary(path: str) -> dict:
    """
    Parse stage1_results_criteria_summary.csv.

    Expected columns:
        criterion, description, type, n_met, n_failed, n_triggered

    Returns:
        inclusion: {criterion: {"n_met": int, "n_failed": int, "desc": str}}
        exclusion: {criterion: {"n_triggered": int, "desc": str}}
        screened:  total records screened (max n_met + n_failed across inclusion rows)
        passed:    min n_met across inclusion criteria (sequential gate)
        excl_total: sum of all n_triggered for exclusion criteria
    """
    df = pd.read_csv(path, skipinitialspace=True)
    df.columns = df.columns.str.strip()
    df["criterion"] = df["criterion"].str.strip()
    df["type"]      = df["type"].str.strip()

    # Strip trailing clause after " — " for short display label
    def short_desc(full_desc: str) -> str:
        full_desc = str(full_desc).strip()
        # Remove leading "Ix — " or "Ex — " prefix added in the description col
        if " \u2014 " in full_desc:
            full_desc = full_desc.split(" \u2014 ", 1)[1]
        if " — " in full_desc:
            full_desc = full_desc.split(" — ", 1)[1]
        return full_desc

    inclusion = {}
    for _, row in df[df["type"] == "inclusion"].iterrows():
        crit = row["criterion"]
        inclusion[crit] = {
            "n_met":    int(row["n_met"])    if pd.notna(row["n_met"])    else 0,
            "n_failed": int(row["n_failed"]) if pd.notna(row["n_failed"]) else 0,
            "desc":     short_desc(row["description"]),
        }

    exclusion = {}
    for _, row in df[df["type"] == "exclusion"].iterrows():
        crit = row["criterion"]
        exclusion[crit] = {
            "n_triggered": int(row["n_triggered"]) if pd.notna(row["n_triggered"]) else 0,
            "desc":        short_desc(row["description"]),
        }

    # Total screened = first inclusion criterion's (n_met + n_failed)
    first_incl = next(iter(inclusion.values()))
    screened = first_incl["n_met"] + first_incl["n_failed"]

    # Records passing all inclusion gates = n_met of the last inclusion criterion
    passed = list(inclusion.values())[-1]["n_met"]

    excl_total = sum(v["n_triggered"] for v in exclusion.values())

    return {
        "inclusion":   inclusion,
        "exclusion":   exclusion,
        "screened":    screened,
        "passed":      passed,
        "excl_total":  excl_total,
    }


# ══════════════════════════════════════════════════════════════════════════════
# LABEL HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def L(lines: list) -> str:
    """Left-aligned multi-line DOT label."""
    return "\\l".join(str(x) for x in lines) + "\\l"


def pad_rows(rows: list[tuple[str, int]], sep: str = "  \u2192  ") -> list[str]:
    """
    Align numbers so counts are right-justified to the same column.
    Uses a monospace font on the node (Courier), so space-padding is reliable.

        rows = [("I1  DSA/fluoroscopy sequence modality", 61),
                ("I2  Addresses a computational processing task", 74)]
        => ["  I1  DSA/fluoroscopy sequence modality          →  61",
            "  I2  Addresses a computational processing task  →  74"]
    """
    if not rows:
        return []
    max_label = max(len(r[0]) for r in rows)
    max_count = max(len(str(r[1])) for r in rows)
    return [
        f"  {label:<{max_label}}{sep}{count:>{max_count}}"
        for label, count in rows
    ]


# ══════════════════════════════════════════════════════════════════════════════
# DIAGRAM BUILDER
# ══════════════════════════════════════════════════════════════════════════════

def build_diagram(fetch: dict, criteria: dict, out_path: str) -> None:

    # ── Derived counts ────────────────────────────────────────────────────────
    db_total    = fetch["db_total"]
    after_dedup = fetch["after_dedup"]
    duplicates  = fetch["duplicates"]
    sources     = fetch["sources"]      # {name: count}

    screened    = criteria["screened"]
    passed      = criteria["passed"]
    excl_total  = criteria["excl_total"]
    inclusion   = criteria["inclusion"]
    exclusion   = criteria["exclusion"]

    # Eligibility phase numbers — not in the CSVs, so derive or set manually
    # sought      = passed (records passing screening = reports sought)
    sought          = passed
    not_retrieved   = 0     # ← update if you have this figure
    assessed        = sought - not_retrieved
    excl_fulltext   = 0     # ← update if you have full-text exclusion count
    included        = assessed - excl_fulltext

    # ── Graphviz setup ────────────────────────────────────────────────────────
    dot = graphviz.Digraph(name="PRISMA_2020", format="pdf")
    dot.attr(
        rankdir="TB",
        splines="ortho",
        nodesep="0.30",
        ranksep="0.38",
        fontname="Helvetica",
        fontsize="13",
        size="8.5,14",
        margin="0.2",
    )
    dot.attr("node",
             shape="box", style="filled,rounded",
             fontname="Helvetica", fontsize="13",
             color="#2C3E50", penwidth="1.6",
             margin="0.22,0.14", width="4.0")
    dot.attr("edge",
             color="#2C3E50", penwidth="1.3", arrowsize="0.85")

    def same_rank(*nodes):
        with dot.subgraph() as r:
            r.attr(rank="same")
            for n in nodes:
                r.node(n)

    # ── Source rows with consistent padding ───────────────────────────────────
    src_max_name  = max(len(k) for k in sources)
    src_max_count = max(len(str(v)) for v in sources.values())
    source_lines  = [
        f"  \u2022 {name:<{src_max_name}}  {count:>{src_max_count}}"
        for name, count in sources.items()
    ]

    # ── Inclusion criteria rows ───────────────────────────────────────────────
    incl_rows = [(f"{k}  {v['desc']}", v["n_met"]) for k, v in inclusion.items()]
    incl_lines = pad_rows(incl_rows, sep="  ->  ")   # ASCII arrow: monospace-safe
    incl_label_lines = [
        f"Records passing title/abstract screening (n = {passed})",
        "Inclusion criteria satisfied (all four required):",
    ] + [f"  [+]  {line.strip()}" for line in incl_lines]

    # ── Exclusion criteria rows ───────────────────────────────────────────────
    excl_rows = [(f"{k}  {v['desc']}", v["n_triggered"]) for k, v in exclusion.items()]
    excl_lines = pad_rows(excl_rows, sep="  ->  ")
    excl_label_lines = [
        f"Records excluded at screening (n = {excl_total})",
        "Exclusion criteria applied:",
    ] + [f"  [-]  {line.strip()}" for line in excl_lines]

    # ══════════════════════════════════════════════════════════════════════════
    # NODES
    # ══════════════════════════════════════════════════════════════════════════

    # — Identification —
    with dot.subgraph(name="cluster_id") as g:
        g.attr(label="Identification", fontsize="14", fontname="Helvetica-Bold",
               style="dashed", color="#5D8AA8", labeljust="l",
               bgcolor="#EBF5FB", margin="14")
        g.node("db_search",
               L([f"Records identified from databases (n = {db_total})"] + source_lines),
               fillcolor="#D6EAF8", color="#2471A3")

    # — Deduplication —
    dot.node("dedup",
             L([f"Records after deduplication (n = {after_dedup})",
                f"  Duplicates removed: {duplicates}"]),
             fillcolor="#D6EAF8", color="#2471A3")

    # — Screening —
    with dot.subgraph(name="cluster_screen") as g:
        g.attr(label="Screening", fontsize="14", fontname="Helvetica-Bold",
               style="dashed", color="#1E8449", labeljust="l",
               bgcolor="#EAFAF1", margin="14")
        g.node("screened",
               L([f"Records screened — title/abstract (n = {screened})"]),
               fillcolor="#A9DFBF", color="#1E8449")
        g.node("passed_screen",
               L(incl_label_lines),
               fillcolor="#D5F5E3", color="#1A5276", penwidth="1.8",
               fontname="Courier", fontsize="12")

    # Screening exclusion side box
    dot.node("excl_screen",
             L(excl_label_lines),
             fillcolor="#FDEDEC", color="#C0392B",
             penwidth="1.8", style="filled,rounded,dashed",
             width="3.6",
             fontname="Courier", fontsize="12")

    # — Eligibility —
    with dot.subgraph(name="cluster_elig") as g:
        g.attr(label="Eligibility", fontsize="14", fontname="Helvetica-Bold",
               style="dashed", color="#5D8AA8", labeljust="l",
               bgcolor="#EBF5FB", margin="14")
        g.node("sought",
               L([f"Reports sought for retrieval (n = {sought})"]),
               fillcolor="#AED6F1", color="#2471A3")
        g.node("assessed",
               L([f"Reports assessed for eligibility (n = {assessed})"]),
               fillcolor="#AED6F1", color="#2471A3")

    # Eligibility side boxes
    dot.node("not_retrieved",
             L([f"Reports not retrieved (n = {not_retrieved})"]),
             fillcolor="#F2F3F4", color="#7F8C8D",
             style="filled,rounded,dashed", width="3.0")
    dot.node("excl_fulltext",
             L([f"Reports excluded at full-text (n = {excl_fulltext})",
                f"  (did not meet full-text eligibility criteria)"]),
             fillcolor="#F2F3F4", color="#7F8C8D",
             style="filled,rounded,dashed", width="3.0")

    # — Included —
    with dot.subgraph(name="cluster_incl") as g:
        g.attr(label="Included", fontsize="14", fontname="Helvetica-Bold",
               style="dashed", color="#C0392B", labeljust="l",
               bgcolor="#FEF9F9", margin="14")
        g.node("included",
               L([f"Studies included in systematic review (n = {included})"]),
               fillcolor="#FADBD8", color="#C0392B", penwidth="2.2")

    # ══════════════════════════════════════════════════════════════════════════
    # RANK ALIGNMENT
    # ══════════════════════════════════════════════════════════════════════════
    same_rank("screened",     "excl_screen")
    same_rank("sought",       "not_retrieved")
    same_rank("assessed",     "excl_fulltext")

    # ══════════════════════════════════════════════════════════════════════════
    # EDGES
    # ══════════════════════════════════════════════════════════════════════════
    dot.edge("db_search",     "dedup")
    dot.edge("dedup",         "screened")
    dot.edge("screened",      "passed_screen")
    dot.edge("passed_screen", "sought")
    dot.edge("sought",        "assessed")
    dot.edge("assessed",      "included")

    dot.edge("screened", "excl_screen",
             style="dashed", color="#C0392B",
             arrowhead="open", constraint="false", penwidth="1.4")
    dot.edge("sought", "not_retrieved",
             style="dashed", color="#7F8C8D",
             arrowhead="open", constraint="false")
    dot.edge("assessed", "excl_fulltext",
             style="dashed", color="#7F8C8D",
             arrowhead="open", constraint="false")

    # ── Render ────────────────────────────────────────────────────────────────
    dot.render(out_path, cleanup=True)
    print(f"✓  PRISMA diagram saved → {out_path}.pdf")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Generate PRISMA 2020 flow diagram from CSV stats.")
    parser.add_argument(
        "--fetch",
        default="/data/Deep_Angiography/AngioVision/slr/results/slr_fetching_stats.csv",
        help="Path to slr_fetching_stats.csv",
    )
    parser.add_argument(
        "--stage1",
        default="/data/Deep_Angiography/AngioVision/slr/results/stage1_results_criteria_summary.csv",
        help="Path to stage1_results_criteria_summary.csv",
    )
    parser.add_argument(
        "--out",
        default="/data/Deep_Angiography/AngioVision/slr/results/prisma_2020",
        help="Output filename without extension (default: results/prisma_2020)",
    )
    args = parser.parse_args()

    # Validate paths
    for label, path in [("--fetch", args.fetch), ("--stage1", args.stage1)]:
        if not Path(path).exists():
            print(f"ERROR: {label} file not found: {path}", file=sys.stderr)
            sys.exit(1)

    # Ensure output directory exists
    out_dir = Path(args.out).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading fetch stats   : {args.fetch}")
    fetch = load_fetch_stats(args.fetch)
    print(f"  Sources      : {fetch['sources']}")
    print(f"  DB total     : {fetch['db_total']}")
    print(f"  After dedup  : {fetch['after_dedup']}")
    print(f"  Duplicates   : {fetch['duplicates']}")

    print(f"Reading criteria      : {args.stage1}")
    criteria = load_criteria_summary(args.stage1)
    print(f"  Screened     : {criteria['screened']}")
    print(f"  Passed       : {criteria['passed']}")
    print(f"  Excl total   : {criteria['excl_total']}")

    build_diagram(fetch, criteria, args.out)


if __name__ == "__main__":
    main()