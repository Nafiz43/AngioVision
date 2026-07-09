#!/usr/bin/env python3
"""
Generate a tiny synthetic reports CSV to smoke-test the pipeline
end-to-end. Written to sample_data/reports_raw.csv (gitignored).

Each report deliberately exercises the cleaning rules: header metadata
block, abbreviations (LAD, HTN, w/, h/o, ...), a HISTORY section,
bullets, and a trailing attestation/signature block.
"""

from __future__ import annotations

import csv
from pathlib import Path

PIPELINE_DIR = Path(__file__).resolve().parent.parent
OUT_CSV = PIPELINE_DIR / "sample_data" / "reports_raw.csv"

REPORT_TEMPLATE = """\
PATIENT: DOE, JOHN {i}.          MRN: 0123456{i}       DOB: 03/14/195{i}
ACCESSION #: A2024031{i}-998{i}    DATE OF PROCEDURE: 03/1{i}/2024
REFERRING PHYSICIAN: Dr. Sarah Mitchell, MD
LOCATION: UC Davis Medical Center, Sacramento, CA
PHONE: (916) 555-014{i}

HISTORY: 6{i} y/o M w/ h/o HTN, DM2, and CAD. Prior HCC involving
Segment {i} treated with TACE.

PROCEDURE: DSA performed for vascular evaluation of the LAD and RCA.

FINDINGS:
    LM:  Normal.
    LAD: 70% stenosis in the proximal segment, TIMI 3 flow.
    RCA: Diffuse 30% disease. No AV shunting.
    * EBL: 50 mL
    * LVEF: 45-50%

IMPRESSION:
    1. Significant proximal LAD disease -- recommend PCI w/ DES.
    2. F/u in clinic in 4 wks.

ATTESTATION:
I have personally reviewed the images and agree with the findings.
Electronically signed by: Robert J. Hansen, MD
Dictated by: R. Hansen, MD
---END OF REPORT---
"""


def main() -> int:
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Orig Acc #", "Anon Acc #", "radrpt"])
        for i in range(1, 4):
            w.writerow([f"ORIG{i:03d}", f"ANON{i:03d}",
                        REPORT_TEMPLATE.format(i=i)])

    print(f"Dummy reports CSV ready: {OUT_CSV} (3 reports)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
