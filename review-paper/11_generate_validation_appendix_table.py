"""
Generate the Appendix C Screening Validation Table
====================================================

Regenerates AngioVision_SLR/tables/screening_validation_full.tex from
validation/full_validation_master.json (the 673-record Cochran/FPC-sized
validation set, see 09_expand_validation_sample.py), joined against
whatever decisions are currently available in validation/annotator_A.csv,
validation/annotator_B.csv, and validation/final_adjudication.json.

Records with no decision yet simply render blank in the Rev. A / Rev. B /
Final columns -- this is expected while the two human annotators are still
working through the set, and the table can be regenerated at any point
(including after annotation is complete) by re-running this script.

Usage:
    python3 11_generate_validation_appendix_table.py
"""

import csv
import json
from pathlib import Path

FULL_MASTER_PATH = Path("validation/full_validation_master.json")
ANNOTATOR_A_PATH = Path("validation/annotator_A.csv")
ANNOTATOR_B_PATH = Path("validation/annotator_B.csv")
FINAL_ADJUDICATION_PATH = Path("validation/final_adjudication.json")
OUTPUT_TABLE_PATH = Path("AngioVision_SLR/tables/screening_validation_full.tex")

DECISION_ABBREV = {"INCLUDE": "I", "EXCLUDE": "E", "UNCERTAIN": "U", "": ""}


def escape_latex(text):
    replacements = {
        "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#",
        "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    for char, escaped in replacements.items():
        text = text.replace(char, escaped)
    return text


def read_annotator_csv(path):
    if not path.exists():
        return {}
    with open(path, newline="", encoding="utf-8") as f:
        return {int(row["sample_id"]): row["decision"].strip().upper() for row in csv.DictReader(f)}


def main():
    records = json.loads(FULL_MASTER_PATH.read_text())
    records.sort(key=lambda r: r["sample_id"])

    dec_a = read_annotator_csv(ANNOTATOR_A_PATH)
    dec_b = read_annotator_csv(ANNOTATOR_B_PATH)
    adjudication = {}
    if FINAL_ADJUDICATION_PATH.exists():
        adjudication = {
            d["sample_id"]: d["final_decision"]
            for d in json.loads(FINAL_ADJUDICATION_PATH.read_text())["finalDecisions"]
        }

    rows = []
    for i, rec in enumerate(records, start=1):
        sid = rec["sample_id"]
        a = dec_a.get(sid, "")
        b = dec_b.get(sid, "")
        if a and b and a == b:
            final = a
        elif sid in adjudication:
            final = adjudication[sid]
        else:
            final = ""
        needed_adjudication = bool(a and b and a != b)
        rows.append({
            "n": i,
            "title": escape_latex(rec["title"]),
            "rev_a": DECISION_ABBREV.get(a, ""),
            "rev_b": DECISION_ABBREV.get(b, ""),
            "final": DECISION_ABBREV.get(final, "") + (r"$^{\dagger}$" if needed_adjudication and final else ""),
            "pipeline": DECISION_ABBREV.get(rec["llm_decision"], ""),
        })

    n_total = len(rows)
    n_resolved = sum(1 for r in rows if r["final"])

    lines = []
    lines.append(r"\renewcommand{\arraystretch}{1.05}")
    lines.append(r"\footnotesize")
    lines.append(r"\onecolumn")
    lines.append(r"\begin{longtable}{@{}r p{9cm} c c c c@{}}")
    lines.append(
        r"\caption{Full " + str(n_total) + r"-record title/abstract screening validation sample "
        r"(Cochran/FPC-sized: full census of the 276 pipeline-INCLUDE records plus a 397-record "
        r"probability sample of the pipeline-EXCLUDE/UNCERTAIN stratum; see Section~\ref{sec:validation}). "
        r"\textbf{Rev.\ A} / \textbf{Rev.\ B}: independent human-annotator decisions (I = Include, "
        r"E = Exclude, U = Uncertain; blank where annotation is still pending). \textbf{Final}: "
        r"adjudicated label used as ground truth (marked $^{\dagger}$ where adjudication was required "
        r"to resolve a Rev.\ A/Rev.\ B disagreement). \textbf{Pipeline}: the original stage-1 screening "
        r"pipeline decision from \texttt{results/stage1\_results.csv}."
        r" As of this draft, " + str(n_resolved) + r" of " + str(n_total) + r" records have a resolved Final label.}"
    )
    lines.append(r"\label{tab:screening_validation_full} \\")
    lines.append(r"\toprule")
    lines.append(r"\textbf{\#} & \textbf{Title} & \textbf{Rev.\,A} & \textbf{Rev.\,B} & \textbf{Final} & \textbf{Pipeline} \\")
    lines.append(r"\midrule")
    lines.append(r"\endfirsthead")
    lines.append(r"\multicolumn{6}{c}{\tablename~\thetable{} -- \textit{continued from previous page}} \\")
    lines.append(r"\toprule")
    lines.append(r"\textbf{\#} & \textbf{Title} & \textbf{Rev.\,A} & \textbf{Rev.\,B} & \textbf{Final} & \textbf{Pipeline} \\")
    lines.append(r"\midrule")
    lines.append(r"\endhead")
    lines.append(r"\midrule")
    lines.append(r"\multicolumn{6}{r}{\textit{Continued on next page}} \\")
    lines.append(r"\endfoot")
    lines.append(r"\bottomrule")
    lines.append(r"\endlastfoot")

    for r in rows:
        lines.append(f"{r['n']} & {r['title']} & {r['rev_a']} & {r['rev_b']} & {r['final']} & {r['pipeline']} \\\\")

    lines.append(r"\end{longtable}")

    OUTPUT_TABLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_TABLE_PATH.write_text("\n".join(lines) + "\n")
    print(f"Wrote {n_total} rows ({n_resolved} resolved) to {OUTPUT_TABLE_PATH}")


if __name__ == "__main__":
    main()
