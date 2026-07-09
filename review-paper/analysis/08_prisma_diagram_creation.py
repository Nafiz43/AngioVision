"""
PRISMA 2020 Flow Diagram — v6  (standard layout, right-side exclusion boxes)
=============================================================================
FIXES vs v5
-----------
  1. splines="ortho" → splines="polyline"
       The ortho mode uses graphviz's maze router which hits an internal
       assertion (maze.c:313) when constraint="false" edges coexist with
       rank="same" subgraphs on graphviz ≤ 2.43.  polyline gives the same
       angular elbow-style edges without the crash.

  2. All excluded items are RIGHT-SIDE boxes (standard PRISMA 2020 format).
       v5 placed passed_inclusion and excl_screen as inline sequential nodes
       in the main vertical spine — non-standard.  In v6 every exclusion
       box hangs to the right of its corresponding main-flow node.

  3. New node: records_removed (Identification cluster).
       Shows duplicate removal + records with no title/abstract, auto-computed
       as  after_dedup − screened  so no hard-coded values are needed.

  4. Invisible ordering edges inside every same_rank pair guarantee that the
       main-flow node is always positioned LEFT of its right-side box.

PRISMA flow (standard):
  ┌─ Identification ─────────────────────────────────┐
  │  db_search  →  records_removed                   │
  └──────────────────────┬───────────────────────────┘
                         ↓
  ┌─ Screening ──────────┴──────────────────────────────────────────────────┐
  │  screened (title/abstract)   ──────────────► [excl_inclusion: failed I] │
  │      ↓                                                                   │
  │  passed_incl (all I met)     ──────────────► [excl_screen:   failed E]  │
  └──────────────────────┬──────────────────────────────────────────────────┘
                         ↓
  ┌─ Eligibility ────────┴──────────────────────────────────────────────────┐
  │  sought for full-text        ──────────────► [not_retrieved]             │
  │      ↓                                                                   │
  │  assessed for eligibility    ──────────────► [excl_fulltext]             │
  └──────────────────────┬──────────────────────────────────────────────────┘
                         ↓
  ┌─ Included ───────────┴───────────────────────────┐
  │  included in systematic review                   │
  └──────────────────────────────────────────────────┘

Usage:
    python prisma_v6.py \\
        --fetch    slr_fetching_stats.csv \\
        --stage1   stage1_results_criteria_summary.csv \\
        --out      prisma_2020 \\
        [--not_retrieved 15] \\
        [--excl_fulltext  0]

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

    Returns dict with keys:
        db_total    : int   — total records before dedup (sum over all sources)
        after_dedup : int   — records after duplicate removal
        duplicates  : int   — duplicate entries removed
        sources     : dict  — {source_name: before_dedup_count}  (TOTAL row excluded)
    """
    df = pd.read_csv(path, skipinitialspace=True)
    df.columns      = df.columns.str.strip()
    df["source"]    = df["source"].str.strip()

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
        "inclusion_summary" — aggregate row; n_passed_all_inclusion gives
                              the count satisfying ALL inclusion criteria
        "exclusion"         — individual exclusion triggers (E1, E2, …)

    Returns dict with keys:
        inclusion   : {crit: {n_met, n_failed, desc}}
        exclusion   : {crit: {n_triggered, desc}}
        screened    : int — total records title/abstract screened
        passed      : int — records passing ALL inclusion criteria
        excl_total  : int — sum of n_triggered across all exclusion criteria
    """
    df = pd.read_csv(path, skipinitialspace=True)
    df.columns      = df.columns.str.strip()
    df["criterion"] = df["criterion"].str.strip()
    df["type"]      = df["type"].str.strip()

    def short_desc(full_desc: str) -> str:
        """Strip leading 'Ix — ' or 'Ex — ' prefix if present."""
        full_desc = str(full_desc).strip()
        for sep in [" \u2014 ", " — "]:
            if sep in full_desc:
                full_desc = full_desc.split(sep, 1)[1]
                break
        return full_desc

    # individual inclusion criteria
    inclusion = {}
    for _, row in df[df["type"] == "inclusion"].iterrows():
        crit = row["criterion"]
        inclusion[crit] = {
            "n_met":    int(row["n_met"])    if pd.notna(row.get("n_met"))    else 0,
            "n_failed": int(row["n_failed"]) if pd.notna(row.get("n_failed")) else 0,
            "desc":     short_desc(row["description"]),
        }

    # all-inclusion-passed count from the summary row
    summary_rows = df[df["type"] == "inclusion_summary"]
    if summary_rows.empty:
        raise ValueError(
            "No row with type='inclusion_summary' found in the criteria CSV.\n"
            "Add an aggregate row with n_passed_all_inclusion filled in."
        )
    passed = int(summary_rows.iloc[0]["n_passed_all_inclusion"])

    # exclusion criteria
    exclusion = {}
    for _, row in df[df["type"] == "exclusion"].iterrows():
        crit = row["criterion"]
        exclusion[crit] = {
            "n_triggered": int(row["n_triggered"]) if pd.notna(row.get("n_triggered")) else 0,
            "desc":        short_desc(row["description"]),
        }

    # screened = n_met + n_failed of the FIRST inclusion criterion row
    first_incl = next(iter(inclusion.values()))
    screened   = first_incl["n_met"] + first_incl["n_failed"]
    excl_total = sum(v["n_triggered"] for v in exclusion.values())

    return {
        "inclusion":  inclusion,
        "exclusion":  exclusion,
        "screened":   screened,
        "passed":     passed,
        "excl_total": excl_total,
    }


# ══════════════════════════════════════════════════════════════════════════════
# LABEL / FORMATTING HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def L(lines: list) -> str:
    """Build a left-aligned DOT label from a list of text lines."""
    return "\\l".join(str(x) for x in lines) + "\\l"


def align_rows(rows: list, sep: str = "  \u2192  ") -> list:
    """Right-justify counts for monospace alignment inside label boxes."""
    if not rows:
        return []
    max_lbl = max(len(str(r[0])) for r in rows)
    max_cnt = max(len(str(r[1])) for r in rows)
    return [
        f"  {str(label):<{max_lbl}}{sep}{str(count):>{max_cnt}}"
        for label, count in rows
    ]


# ══════════════════════════════════════════════════════════════════════════════
# DIAGRAM BUILDER
# ══════════════════════════════════════════════════════════════════════════════

def build_diagram(
    fetch: dict,
    criteria: dict,
    out_path: str,
    not_retrieved_n: int = 15,
    excl_fulltext_n: int = 0,
) -> None:
    """
    Render the PRISMA 2020 flow diagram to a PDF file.

    Parameters
    ----------
    fetch           : output of load_fetch_stats()
    criteria        : output of load_criteria_summary()
    out_path        : base output path WITHOUT extension (e.g. 'prisma_2020')
    not_retrieved_n : reports sought but not retrieved           (default 15)
    excl_fulltext_n : reports excluded at full-text stage        (default  0)
    """

    # ── Derived counts ────────────────────────────────────────────────────────
    db_total    = fetch["db_total"]         # e.g. 5 829
    after_dedup = fetch["after_dedup"]      # e.g. 5 064
    duplicates  = fetch["duplicates"]       # e.g.   765
    sources     = fetch["sources"]          # {"PubMed": 4071, "SemanticScholar": 1758}

    screened      = criteria["screened"]    # e.g. 5 035
    passed        = criteria["passed"]      # e.g.   353  (ALL inclusion criteria met)
    excl_screen_n = criteria["excl_total"]  # e.g.   144  (exclusion criteria triggered)
    inclusion     = criteria["inclusion"]
    exclusion     = criteria["exclusion"]

    # Records removed before screening (no abstract/title after dedup)
    no_abstract_n    = after_dedup - screened           # 5 064 − 5 035 = 29
    removed_before_n = duplicates  + no_abstract_n      #   765 +    29 = 794

    # Records failing ≥1 inclusion criterion
    excl_inclusion_n = screened - passed                # 5 035 − 353 = 4 682

    # Full-text flow
    sought_n   = passed   - excl_screen_n               #   353 − 144 =   209
    assessed_n = sought_n - not_retrieved_n             #   209 −  15 =   194
    included_n = assessed_n - excl_fulltext_n           #   194 −   0 =   194

    # ── Graphviz init ─────────────────────────────────────────────────────────
    dot = graphviz.Digraph(name="PRISMA_2020", format="pdf")
    dot.attr(
        rankdir  = "TB",
        # ⚠ "ortho" triggers a maze.c assertion crash on graphviz ≤ 2.43 when
        #   constraint="false" edges coexist with rank="same" subgraphs.
        # "polyline" gives identical angular elbow connections without the bug.
        splines  = "polyline",
        nodesep  = "0.6",
        ranksep  = "0.55",
        fontname = "Helvetica",
        fontsize = "12",
        size     = "12,22",
        margin   = "0.3",
        pad      = "0.4",
    )
    dot.attr("node",
             shape    = "box",
             style    = "filled,rounded",
             fontname = "Helvetica",
             fontsize = "12",
             color    = "#2C3E50",
             penwidth = "1.5",
             margin   = "0.22,0.14")
    dot.attr("edge",
             color     = "#2C3E50",
             penwidth  = "1.2",
             arrowsize = "0.85")

    # ── Rank helper ───────────────────────────────────────────────────────────
    def same_rank_lr(left_id: str, right_id: str) -> None:
        """
        Place two nodes at the same vertical rank AND force left_id to appear
        to the LEFT of right_id via an invisible ordering edge.
        This guarantees the main-flow node is always left of its side box.
        """
        with dot.subgraph() as r:
            r.attr(rank="same")
            r.node(left_id)
            r.node(right_id)
            # invisible edge: enforces horizontal ordering without affecting
            # the visible arrow routing
            r.edge(left_id, right_id, style="invis")

    # ── Label content ─────────────────────────────────────────────────────────

    # Source breakdown lines for db_search node
    sw = max(len(k) for k in sources)
    cw = max(len(str(v)) for v in sources.values())
    src_lines = [
        f"  \u2022 {name:<{sw}}  {count:>{cw}}"
        for name, count in sources.items()
    ]

    # Inclusion criteria: pass counts for right-side excl_inclusion box
    incl_pairs = [(f"{k}  {v['desc']}", v["n_met"]) for k, v in inclusion.items()]
    incl_lines = align_rows(incl_pairs, sep="  \u2192  ")

    # Exclusion criteria: trigger counts for right-side excl_screen box
    excl_pairs = [(f"{k}  {v['desc']}", v["n_triggered"]) for k, v in exclusion.items()]
    excl_lines = align_rows(excl_pairs, sep="  \u2192  ")

    # ── Width constants ───────────────────────────────────────────────────────
    MW = "5.8"   # main-flow node width
    SW = "4.6"   # right-side (exclusion/not-retrieved) box width

    # Shared visual style for all right-side boxes
    SIDE = dict(
        fillcolor = "#F8F9FA",
        color     = "#5D6D7E",
        penwidth  = "1.2",
        fontname  = "Courier",
        fontsize  = "11",
        width     = SW,
        style     = "filled,rounded",
    )

    # ══════════════════════════════════════════════════════════════════════════
    # IDENTIFICATION SECTION
    # ══════════════════════════════════════════════════════════════════════════
    with dot.subgraph(name="cluster_id") as g:
        g.attr(
            label     = "Identification",
            fontsize  = "14",
            fontname  = "Helvetica-Bold",
            style     = "dashed",
            color     = "#2471A3",
            labeljust = "l",
            bgcolor   = "#EBF5FB",
            margin    = "14",
        )
        g.node(
            "db_search",
            L([f"Records identified from databases (n = {db_total})"] + src_lines),
            fillcolor="#D6EAF8", color="#2471A3", width=MW,
        )
        g.node(
            "records_removed",
            L([
                f"Records removed before screening (n = {removed_before_n})",
                f"  Duplicate records removed:                   {duplicates}",
                f"  Records with no title / abstract:            {no_abstract_n}",
            ]),
            fillcolor="#D6EAF8", color="#2471A3", width=MW,
        )

    # ══════════════════════════════════════════════════════════════════════════
    # SCREENING SECTION — main-flow nodes
    # ══════════════════════════════════════════════════════════════════════════
    with dot.subgraph(name="cluster_screen") as g:
        g.attr(
            label     = "Screening",
            fontsize  = "14",
            fontname  = "Helvetica-Bold",
            style     = "dashed",
            color     = "#1A5276",
            labeljust = "l",
            bgcolor   = "#EAFAF1",
            margin    = "14",
        )
        g.node(
            "screened",
            L([f"Records screened \u2014 title / abstract  (n = {screened})"]),
            fillcolor="#A9DFBF", color="#1A5276", width=MW,
        )
        g.node(
            "passed_incl",
            L([f"Records passing ALL inclusion criteria  (n = {passed})"]),
            fillcolor="#A9DFBF", color="#1A5276", width=MW,
        )

    # ── Right-side box 1: failed ≥1 inclusion criterion ───────────────────────
    dot.node(
        "excl_inclusion",
        L(
            [f"Records excluded  (n = {excl_inclusion_n})",
             "(failed \u22651 inclusion criterion)",
             "Individual criterion pass counts:"]
            + [f"  [+] {ln.strip()}" for ln in incl_lines]
        ),
        **SIDE,
    )

    # ── Right-side box 2: triggered ≥1 exclusion criterion ────────────────────
    dot.node(
        "excl_screen",
        L(
            [f"Records excluded  (n = {excl_screen_n})",
             "Exclusion criteria triggered:"]
            + [f"  [\u2212] {ln.strip()}" for ln in excl_lines]
        ),
        **SIDE,
    )

    # ══════════════════════════════════════════════════════════════════════════
    # ELIGIBILITY SECTION — main-flow nodes
    # ══════════════════════════════════════════════════════════════════════════
    with dot.subgraph(name="cluster_elig") as g:
        g.attr(
            label     = "Eligibility",
            fontsize  = "14",
            fontname  = "Helvetica-Bold",
            style     = "dashed",
            color     = "#1F618D",
            labeljust = "l",
            bgcolor   = "#EBF5FB",
            margin    = "14",
        )
        g.node(
            "sought",
            L([f"Reports sought for full-text retrieval  (n = {sought_n})"]),
            fillcolor="#AED6F1", color="#1F618D", width=MW,
        )
        g.node(
            "assessed",
            L([f"Reports assessed for eligibility  (n = {assessed_n})"]),
            fillcolor="#AED6F1", color="#1F618D", width=MW,
        )

    # ── Right-side box 3: not retrieved ───────────────────────────────────────
    dot.node(
        "not_retrieved",
        L([f"Reports not retrieved  (n = {not_retrieved_n})"]),
        **SIDE,
    )

    # ── Right-side box 4: excluded at full-text ───────────────────────────────
    dot.node(
        "excl_fulltext",
        L([
            f"Reports excluded at full-text  (n = {excl_fulltext_n})",
            "  (did not meet full-text eligibility criteria)",
        ]),
        **SIDE,
    )

    # ══════════════════════════════════════════════════════════════════════════
    # INCLUDED SECTION
    # ══════════════════════════════════════════════════════════════════════════
    with dot.subgraph(name="cluster_incl") as g:
        g.attr(
            label     = "Included",
            fontsize  = "14",
            fontname  = "Helvetica-Bold",
            style     = "dashed",
            color     = "#922B21",
            labeljust = "l",
            bgcolor   = "#FDEDEC",
            margin    = "14",
        )
        g.node(
            "included",
            L([f"Studies included in systematic review  (n = {included_n})"]),
            fillcolor="#FADBD8", color="#922B21", penwidth="2.0", width=MW,
        )

    # ══════════════════════════════════════════════════════════════════════════
    # RANK ALIGNMENT + HORIZONTAL ORDERING
    # Each main-flow node shares a rank with its right-side box.
    # The invisible edge inside same_rank_lr() forces main LEFT of box.
    # ══════════════════════════════════════════════════════════════════════════
    same_rank_lr("screened",    "excl_inclusion")
    same_rank_lr("passed_incl", "excl_screen")
    same_rank_lr("sought",      "not_retrieved")
    same_rank_lr("assessed",    "excl_fulltext")

    # ══════════════════════════════════════════════════════════════════════════
    # MAIN VERTICAL EDGES
    # ══════════════════════════════════════════════════════════════════════════
    dot.edge("db_search",       "records_removed")
    dot.edge("records_removed", "screened")
    dot.edge("screened",        "passed_incl")
    dot.edge("passed_incl",     "sought")
    dot.edge("sought",          "assessed")
    dot.edge("assessed",        "included")

    # ══════════════════════════════════════════════════════════════════════════
    # LATERAL SIDE-BOX EDGES
    # constraint="false" prevents these horizontal arrows from contributing to
    # the vertical rank computation, keeping the main flow clean.
    # ══════════════════════════════════════════════════════════════════════════
    SIDE_EDGE = dict(
        color      = "#5D6D7E",
        penwidth   = "1.0",
        arrowhead  = "open",
        constraint = "false",
    )
    dot.edge("screened",    "excl_inclusion", **SIDE_EDGE)
    dot.edge("passed_incl", "excl_screen",    **SIDE_EDGE)
    dot.edge("sought",      "not_retrieved",  **SIDE_EDGE)
    dot.edge("assessed",    "excl_fulltext",  **SIDE_EDGE)

    # ── Render ────────────────────────────────────────────────────────────────
    dot.render(out_path, cleanup=True)
    print(f"\u2713  PRISMA 2020 diagram saved \u2192 {out_path}.pdf")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(
        description=(
            "Generate a PRISMA 2020 flow diagram "
            "(standard format: all exclusion boxes on the right side)."
        )
    )
    ap.add_argument(
        "--fetch",
        default="/data/Deep_Angiography/AngioVision/slr/results/slr_fetching_stats.csv",
        help="Path to slr_fetching_stats.csv",
    )
    ap.add_argument(
        "--stage1",
        default=(
            "/data/Deep_Angiography/AngioVision/slr/results/"
            "stage1_results_criteria_summary.csv"
        ),
        help="Path to stage1_results_criteria_summary.csv",
    )
    ap.add_argument(
        "--out",
        default="/data/Deep_Angiography/AngioVision/slr/results/prisma_2020",
        help="Output base path without extension (PDF appended automatically)",
    )
    ap.add_argument(
        "--not_retrieved",
        type=int, default=15,
        help="Full-text reports that could not be retrieved (default: 15)",
    )
    ap.add_argument(
        "--excl_fulltext",
        type=int, default=0,
        help="Reports excluded at full-text eligibility assessment (default: 0)",
    )
    args = ap.parse_args()

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

    print(f"\nReading criteria      : {args.stage1}")
    criteria = load_criteria_summary(args.stage1)

    # Echo derived counts for easy sanity-check
    no_abstract_n    = fetch["after_dedup"] - criteria["screened"]
    excl_inclusion_n = criteria["screened"] - criteria["passed"]
    sought_n         = criteria["passed"]   - criteria["excl_total"]
    assessed_n       = sought_n - args.not_retrieved
    included_n       = assessed_n - args.excl_fulltext

    rows = [
        ("DB total (all sources)",           fetch["db_total"]),
        ("  Duplicates removed",              fetch["duplicates"]),
        ("  No title/abstract removed",       no_abstract_n),
        ("Records screened (title/abstract)", criteria["screened"]),
        ("  Excluded (failed inclusion)",     excl_inclusion_n),
        ("  Passed ALL inclusion criteria",   criteria["passed"]),
        ("  Excluded (exclusion criteria)",   criteria["excl_total"]),
        ("Reports sought for full-text",      sought_n),
        ("  Not retrieved",                   args.not_retrieved),
        ("Reports assessed for eligibility",  assessed_n),
        ("  Excluded at full-text",           args.excl_fulltext),
        ("Studies included",                  included_n),
    ]
    w = max(len(r[0]) for r in rows)
    print("\n  \u2500\u2500 PRISMA flow (derived) " + "\u2500" * 30)
    for lbl, val in rows:
        print(f"  {lbl:<{w}}  {val:>6}")
    print("  " + "\u2500" * (w + 10))

    build_diagram(
        fetch, criteria, args.out,
        not_retrieved_n = args.not_retrieved,
        excl_fulltext_n = args.excl_fulltext,
    )


if __name__ == "__main__":
    main()