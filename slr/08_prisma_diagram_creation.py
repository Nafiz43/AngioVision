"""
PRISMA 2020 Flow Diagram — v5 (updated)
========================================
Reads counts directly from:
  1. slr_fetching_stats.csv                — identification / deduplication numbers
  2. stage1_results_criteria_summary.csv   — per-criterion inclusion/exclusion counts

Screening flow (sequential, all inline — no side-boxes):
    screened (title/abstract)
        ↓
    [green] passed_inclusion  — articles passing ALL inclusion criteria (n=353)
        ↓
    [red]   excl_screen       — articles excluded by exclusion criteria  (n=64)
        ↓
    sought for full-text      — 353 − 64 = 289

The "passed" count comes from the inclusion_summary row's n_passed_all_inclusion column.

Usage:
    python prisma_v5.py \\
        --fetch  /data/Deep_Angiography/AngioVision/slr/results/slr_fetching_stats.csv \\
        --stage1 /data/Deep_Angiography/AngioVision/slr/results/stage1_results_criteria_summary.csv \\
        --out    prisma_2020

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

    total_row   = df[df["source"].str.upper() == "TOTAL"].iloc[0]
    source_rows = df[df["source"].str.upper() != "TOTAL"].copy()

    return {
        "db_total":    int(total_row["before_dedup"]),
        "after_dedup": int(total_row["after_dedup"]),
        "duplicates":  int(total_row["duplicates_removed"]),
        "sources": {
            row["source"]: int(row["before_dedup"])
            for _, row in source_rows.iterrows()
        },
    }


def load_criteria_summary(path: str) -> dict:
    """
    Parse stage1_results_criteria_summary.csv.

    Expected columns:
        criterion, description, type, n_met, n_failed, n_triggered,
        n_passed_all_inclusion

    Row types handled:
        "inclusion"         — individual inclusion gates (I1, I2, …)
        "inclusion_summary" — aggregate row whose n_passed_all_inclusion gives
                              the count of articles satisfying ALL inclusion criteria
        "exclusion"         — individual exclusion triggers (E1, E2, …)

    Returns dict with keys:
        inclusion   : {crit: {"n_met", "n_failed", "desc"}}
        exclusion   : {crit: {"n_triggered", "desc"}}
        screened    : total records screened (n_met + n_failed of first inclusion row)
        passed      : articles passing ALL inclusion criteria (from inclusion_summary)
        excl_total  : sum of n_triggered across all exclusion rows
        after_excl  : passed − excl_total  (proceed to full-text)
    """
    df = pd.read_csv(path, skipinitialspace=True)
    df.columns      = df.columns.str.strip()
    df["criterion"] = df["criterion"].str.strip()
    df["type"]      = df["type"].str.strip()

    def short_desc(full_desc: str) -> str:
        """Strip leading 'Ix — ' or 'Ex — ' prefix."""
        full_desc = str(full_desc).strip()
        for sep in [" \u2014 ", " — "]:
            if sep in full_desc:
                full_desc = full_desc.split(sep, 1)[1]
                break
        return full_desc

    # Individual inclusion criteria
    inclusion = {}
    for _, row in df[df["type"] == "inclusion"].iterrows():
        crit = row["criterion"]
        inclusion[crit] = {
            "n_met":    int(row["n_met"])    if pd.notna(row.get("n_met"))    else 0,
            "n_failed": int(row["n_failed"]) if pd.notna(row.get("n_failed")) else 0,
            "desc":     short_desc(row["description"]),
        }

    # All-inclusion-passed count from the summary row
    summary_rows = df[df["type"] == "inclusion_summary"]
    if summary_rows.empty:
        raise ValueError(
            "No row with type='inclusion_summary' found. "
            "Please add the 'I1-I4 (ALL)' aggregate row with n_passed_all_inclusion."
        )
    passed = int(summary_rows.iloc[0]["n_passed_all_inclusion"])

    # Exclusion criteria
    exclusion = {}
    for _, row in df[df["type"] == "exclusion"].iterrows():
        crit = row["criterion"]
        exclusion[crit] = {
            "n_triggered": int(row["n_triggered"]) if pd.notna(row.get("n_triggered")) else 0,
            "desc":        short_desc(row["description"]),
        }

    first_incl = next(iter(inclusion.values()))
    screened   = first_incl["n_met"] + first_incl["n_failed"]
    excl_total = sum(v["n_triggered"] for v in exclusion.values())
    after_excl = passed - excl_total

    return {
        "inclusion":  inclusion,
        "exclusion":  exclusion,
        "screened":   screened,
        "passed":     passed,
        "excl_total": excl_total,
        "after_excl": after_excl,
    }


# ══════════════════════════════════════════════════════════════════════════════
# LABEL HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def L(lines: list) -> str:
    """Left-aligned multi-line DOT label."""
    return "\\l".join(str(x) for x in lines) + "\\l"


def pad_rows(rows: list, sep: str = "  ->  ") -> list:
    """Right-justify counts for monospace alignment."""
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
    sources     = fetch["sources"]

    screened    = criteria["screened"]
    passed      = criteria["passed"]      # articles passing ALL inclusion gates
    excl_total  = criteria["excl_total"]  # articles removed by exclusion criteria
    after_excl  = criteria["after_excl"]  # passed − excl_total → proceed to full-text
    inclusion   = criteria["inclusion"]
    exclusion   = criteria["exclusion"]

    # Eligibility / Included phase
    sought          = after_excl          # articles sent for full-text retrieval
    not_retrieved   = 15                   # update if known
    assessed        = sought - not_retrieved
    excl_fulltext   = 0                   # update if known
    included        = assessed - excl_fulltext

    # ── Graphviz setup ────────────────────────────────────────────────────────
    dot = graphviz.Digraph(name="PRISMA_2020", format="pdf")
    dot.attr(
        rankdir="TB",
        splines="ortho",
        nodesep="0.40",
        ranksep="0.42",
        fontname="Helvetica",
        fontsize="13",
        size="9,16",
        margin="0.2",
    )
    dot.attr("node",
             shape="box", style="filled,rounded",
             fontname="Helvetica", fontsize="13",
             color="#2C3E50", penwidth="1.6",
             margin="0.22,0.14", width="5.2")
    dot.attr("edge",
             color="#2C3E50", penwidth="1.3", arrowsize="0.85")

    def same_rank(*nodes):
        with dot.subgraph() as r:
            r.attr(rank="same")
            for n in nodes:
                r.node(n)

    # ── Source rows ───────────────────────────────────────────────────────────
    src_max_name  = max(len(k) for k in sources)
    src_max_count = max(len(str(v)) for v in sources.values())
    source_lines  = [
        f"  \u2022 {name:<{src_max_name}}  {count:>{src_max_count}}"
        for name, count in sources.items()
    ]

    # ── Inclusion label lines ─────────────────────────────────────────────────
    incl_rows  = [(f"{k}  {v['desc']}", v["n_met"]) for k, v in inclusion.items()]
    incl_lines = pad_rows(incl_rows)
    incl_label = (
        [
            f"Records passing ALL inclusion criteria (n = {passed})",
            "Inclusion criteria (all four required):",
        ]
        + [f"  [+]  {line.strip()}" for line in incl_lines]
    )

    # ── Exclusion label lines ─────────────────────────────────────────────────
    excl_rows  = [(f"{k}  {v['desc']}", v["n_triggered"]) for k, v in exclusion.items()]
    excl_lines = pad_rows(excl_rows)
    excl_label = (
        [
            f"Records excluded (n = {excl_total})",
            "Exclusion criteria applied:",
        ]
        + [f"  [-]  {line.strip()}" for line in excl_lines]
    )

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

        # 1. Records screened
        g.node("screened",
               L([f"Records screened — title/abstract (n = {screened})"]),
               fillcolor="#A9DFBF", color="#1E8449")

        # 2. Inclusion block (green) — sequential, inline
        g.node("passed_inclusion",
               L(incl_label),
               fillcolor="#D5F5E3", color="#1A5276", penwidth="1.8",
               fontname="Courier", fontsize="12")

        # 3. Exclusion block (red) — sequential, inline, still inside Screening cluster
        g.node("excl_screen",
               L(excl_label),
               fillcolor="#FDEDEC", color="#C0392B", penwidth="1.8",
               fontname="Courier", fontsize="12",
               style="filled,rounded")

    # — Eligibility —
    with dot.subgraph(name="cluster_elig") as g:
        g.attr(label="Eligibility", fontsize="14", fontname="Helvetica-Bold",
               style="dashed", color="#5D8AA8", labeljust="l",
               bgcolor="#EBF5FB", margin="14")
        g.node("sought",
               L([f"Reports sought for full-text retrieval (n = {sought})"]),
               fillcolor="#AED6F1", color="#2471A3")
        g.node("assessed",
               L([f"Reports assessed for eligibility (n = {assessed})"]),
               fillcolor="#AED6F1", color="#2471A3")

    # Eligibility side-boxes
    dot.node("not_retrieved",
             L([f"Reports not retrieved (n = {not_retrieved})"]),
             fillcolor="#F2F3F4", color="#7F8C8D",
             style="filled,rounded,dashed", width="3.0")
    dot.node("excl_fulltext",
             L([f"Reports excluded at full-text (n = {excl_fulltext})",
                "  (did not meet full-text eligibility criteria)"]),
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
    # RANK ALIGNMENT (eligibility side-boxes only)
    # ══════════════════════════════════════════════════════════════════════════
    same_rank("sought",   "not_retrieved")
    same_rank("assessed", "excl_fulltext")

    # ══════════════════════════════════════════════════════════════════════════
    # EDGES — main vertical chain
    # ══════════════════════════════════════════════════════════════════════════
    dot.edge("db_search",        "dedup")
    dot.edge("dedup",            "screened")
    dot.edge("screened",         "passed_inclusion")   # → inclusion block
    dot.edge("passed_inclusion", "excl_screen")        # → exclusion block (sequential)
    dot.edge("excl_screen",      "sought")             # → full-text (289)
    dot.edge("sought",           "assessed")
    dot.edge("assessed",         "included")

    # Eligibility lateral side-boxes (dashed)
    dot.edge("sought",   "not_retrieved",
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
    parser = argparse.ArgumentParser(
        description="Generate PRISMA 2020 flow diagram from CSV stats."
    )
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
        help="Output filename without extension",
    )
    args = parser.parse_args()

    for label, path in [("--fetch", args.fetch), ("--stage1", args.stage1)]:
        if not Path(path).exists():
            print(f"ERROR: {label} file not found: {path}", file=sys.stderr)
            sys.exit(1)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    print(f"Reading fetch stats   : {args.fetch}")
    fetch = load_fetch_stats(args.fetch)
    print(f"  Sources      : {fetch['sources']}")
    print(f"  DB total     : {fetch['db_total']}")
    print(f"  After dedup  : {fetch['after_dedup']}")
    print(f"  Duplicates   : {fetch['duplicates']}")

    print(f"Reading criteria      : {args.stage1}")
    criteria = load_criteria_summary(args.stage1)
    print(f"  Screened        : {criteria['screened']}")
    print(f"  Passed (all I)  : {criteria['passed']}")
    print(f"  Excl total      : {criteria['excl_total']}")
    print(f"  After exclusion : {criteria['after_excl']}  (→ full-text)")

    build_diagram(fetch, criteria, args.out)


if __name__ == "__main__":
    main()