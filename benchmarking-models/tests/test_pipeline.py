#!/usr/bin/env python3
"""Self-contained smoke test: step 02 statistics on synthetic data + the
shared normalization rules. No Ollama, no lab data — safe anywhere the
Python deps (pandas, sklearn, statsmodels) are installed.

    python3 tests/test_pipeline.py
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from config import load_config, dequote
from bmk import common, s04_statistics


def main():
    tmp = Path(tempfile.mkdtemp())
    bdir, ftdir, rundir = tmp / "baselines", tmp / "ft", tmp / "run"
    for d in (bdir, ftdir, rundir):
        d.mkdir()

    # Synthetic: 40 paired rows; baseline ~55% accurate, fine-tuned ~95%.
    rng = np.random.default_rng(0)
    n = 40
    gt = rng.integers(0, 2, n)
    rows = [{
        "AccessionNumber": f"A{i % 5}",
        "SOPInstanceUID": f"1.2.{i}",
        "Question": "Is there stenosis?",
        "GroundTruth": "YES" if gt[i] else "NO",
        "Predicted": ("YES" if gt[i] else "NO") if rng.random() < 0.55
                     else ("NO" if gt[i] else "YES"),
    } for i in range(n)]
    pd.DataFrame(rows).to_csv(bdir / "fakevlm_predictions.csv", index=False)

    ft_rows = [{
        "AccessionNumber": r["AccessionNumber"],
        "SOPInstanceUID": r["SOPInstanceUID"],
        "Question": r["Question"],
        "Answer": r["GroundTruth"] if rng.random() < 0.95
                  else ("NO" if r["GroundTruth"] == "YES" else "YES"),
    } for r in rows]
    pd.DataFrame(ft_rows).to_csv(ftdir / "clip_predictions.csv", index=False)

    cfg = load_config()
    cfg.baselines_dir = str(bdir)
    cfg.ft_predictions_dir = str(ftdir)

    summary = s04_statistics.run(cfg, rundir)
    assert summary["n_comparisons"] == 4, summary  # 1 real + 3 controls

    out = pd.read_csv(rundir / "statistical_comparison.csv")
    assert {"mcnemar_p_value", "bootstrap_ci_lower",
            "fine_tuned_f1_score"} <= set(out.columns)
    row = out[out.baseline_model == "fakevlm_predictions"].iloc[0]
    assert row["fine_tuned_accuracy"] > row["baseline_accuracy"]
    assert (out.baseline_model == "All_No_Model").any()

    # Normalization rules — shared by both steps; drift here corrupts stats.
    assert common.normalize_llm_answer("Yes.") == "YES"
    assert common.normalize_llm_answer("Unclear — cannot determine") == "NO"
    assert common.normalize_llm_answer(None) == "NO"
    assert common.normalize_gt_answer("positive") == "YES"
    assert common.normalize_binary("YES") == 1
    assert common.normalize_binary("no") == 0
    assert common.sanitize_model_tag("qwen3-vl:32b") == "qwen3-vl_32b"

    # The quoted-path bug that once created a directory named `'`.
    assert dequote("'/data/foo'") == "/data/foo"
    assert dequote('"/data/foo"') == "/data/foo"

    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
