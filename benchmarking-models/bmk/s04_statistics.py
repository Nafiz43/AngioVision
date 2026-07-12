"""Step 04 — paired statistical comparison: every baseline vs every fine-tuned model.

Port of utils/07_mcneimer_test.py with the hardcoded BASELINE_FILES replaced by
auto-discovery: baselines are the *_predictions.csv files step 01 wrote into
cfg.baselines_dir, fine-tuned predictions are *predictions*.csv files in
cfg.ft_predictions_dir. Statistics are unchanged: McNemar (exact when
discordant < 20), bootstrap CI on delta accuracy, per-side confusion metrics,
plus three synthetic controls (seeded-random / all-yes / all-no).

Output: statistical_comparison.csv in the per-run directory, sorted so
significant fine-tuned wins come first.
"""

from __future__ import annotations

import glob
import os

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from statsmodels.stats.contingency_tables import mcnemar

from .common import MERGE_COLS, normalize_binary


def _normalize_keys(df: pd.DataFrame) -> pd.DataFrame:
    for col in MERGE_COLS:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
    return df


def load_baseline_predictions(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = MERGE_COLS + ["GroundTruth", "Predicted"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Baseline file {path} missing columns: {missing}")

    df = df.rename(columns={"GroundTruth": "y_true", "Predicted": "baseline_pred"})
    df = _normalize_keys(df)
    df["y_true"] = df["y_true"].apply(normalize_binary)
    df["baseline_pred"] = df["baseline_pred"].apply(normalize_binary)
    return df[MERGE_COLS + ["y_true", "baseline_pred"]].drop_duplicates()


def load_finetuned_predictions(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = MERGE_COLS + ["Answer"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Fine-tuned file {path} missing columns: {missing}")

    df = df.rename(columns={"Answer": "ft_pred"})
    df = _normalize_keys(df)
    df["ft_pred"] = df["ft_pred"].apply(normalize_binary)
    return df[MERGE_COLS + ["ft_pred"]].drop_duplicates()


def build_control_baselines(reference_df: pd.DataFrame, random_seed: int) -> dict:
    """Synthetic controls on the reference baseline's key space."""
    ref = reference_df[MERGE_COLS + ["y_true"]].drop_duplicates().copy()
    rng = np.random.default_rng(random_seed)

    random_df = ref.copy()
    random_df["baseline_pred"] = rng.integers(0, 2, size=len(random_df))
    all_yes = ref.copy()
    all_yes["baseline_pred"] = 1
    all_no = ref.copy()
    all_no["baseline_pred"] = 0

    return {
        f"Random_Seeded_{random_seed}": random_df,
        "All_Yes_Model": all_yes,
        "All_No_Model": all_no,
    }


def compute_binary_metrics(y_true, y_pred, prefix: str) -> dict:
    y_true = pd.Series(y_true).astype(int)
    y_pred = pd.Series(y_pred).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        f"{prefix}_tp": int(tp), f"{prefix}_tn": int(tn),
        f"{prefix}_fp": int(fp), f"{prefix}_fn": int(fn),
        f"{prefix}_precision": float(precision_score(y_true, y_pred, zero_division=0)),
        f"{prefix}_recall": float(recall_score(y_true, y_pred, zero_division=0)),
        f"{prefix}_f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def bootstrap_accuracy_test(merged: pd.DataFrame, n_bootstrap: int,
                            alpha: float, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    y_true = merged["y_true"].to_numpy()
    baseline_pred = merged["baseline_pred"].to_numpy()
    ft_pred = merged["ft_pred"].to_numpy()
    n = len(merged)

    baseline_acc = float(np.mean(baseline_pred == y_true))
    ft_acc = float(np.mean(ft_pred == y_true))

    diffs = np.empty(n_bootstrap, dtype=float)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        diffs[i] = (np.mean(ft_pred[idx] == y_true[idx])
                    - np.mean(baseline_pred[idx] == y_true[idx]))

    lower = float(np.percentile(diffs, 100 * (alpha / 2)))
    upper = float(np.percentile(diffs, 100 * (1 - alpha / 2)))

    if lower > 0:
        significant, interpretation = "Yes", (
            f"The fine-tuned model shows a statistically meaningful accuracy "
            f"improvement over the baseline under bootstrap resampling at "
            f"alpha={alpha:.2f}."
        )
    elif upper < 0:
        significant, interpretation = "Yes", (
            f"The baseline model shows a statistically meaningful accuracy "
            f"improvement over the fine-tuned model under bootstrap resampling "
            f"at alpha={alpha:.2f}."
        )
    else:
        significant, interpretation = "No", (
            f"No statistically meaningful accuracy difference was observed "
            f"under bootstrap resampling at alpha={alpha:.2f}."
        )

    return {
        "baseline_accuracy": baseline_acc,
        "fine_tuned_accuracy": ft_acc,
        "delta_accuracy_ft_minus_baseline": float(np.mean(diffs)),
        "bootstrap_ci_lower": lower,
        "bootstrap_ci_upper": upper,
        "bootstrap_significant_at_alpha": significant,
        "bootstrap_interpretation": interpretation,
    }


def run_mcnemar_comparison(merged: pd.DataFrame, model_a_name: str,
                           model_b_name: str, alpha: float) -> dict:
    baseline_correct = merged["baseline_pred"] == merged["y_true"]
    ft_correct = merged["ft_pred"] == merged["y_true"]

    both_correct = int((baseline_correct & ft_correct).sum())
    both_wrong = int((~baseline_correct & ~ft_correct).sum())
    b = int((~baseline_correct & ft_correct).sum())   # baseline wrong, FT correct
    c = int((baseline_correct & ~ft_correct).sum())   # baseline correct, FT wrong
    discordant = b + c

    if discordant == 0:
        p_value, test_type = 1.0, "not_applicable"
    else:
        use_exact = discordant < 20
        result = mcnemar([[both_correct, c], [b, both_wrong]],
                         exact=use_exact, correction=not use_exact)
        p_value = float(result.pvalue)
        test_type = "exact" if use_exact else "chi-square with continuity correction"

    if p_value < alpha:
        significant = "Yes"
        if b > c:
            winner = model_b_name
        elif c > b:
            winner = model_a_name
        else:
            winner = "Tie"
        interpretation = (
            f"Statistically significant difference by McNemar's test at "
            f"alpha={alpha:.2f}; winner: {winner}."
        )
    else:
        significant = "No"
        lean = model_b_name if b > c else model_a_name if c > b else "Neither model"
        winner = "No significant difference"
        interpretation = (
            f"No statistically significant difference by McNemar's test at "
            f"alpha={alpha:.2f}. Discordant counts lean toward: {lean}."
        )

    return {
        "baseline_model": model_a_name,
        "fine_tuned_model": model_b_name,
        "matched_samples": int(len(merged)),
        "both_correct": both_correct,
        "both_wrong": both_wrong,
        "baseline_wrong_ft_correct_b": b,
        "baseline_correct_ft_wrong_c": c,
        "discordant_total_b_plus_c": discordant,
        "mcnemar_test_type": test_type,
        "alpha": alpha,
        "mcnemar_p_value": p_value,
        "significant_at_alpha": significant,
        "mcnemar_winner": winner,
        "mcnemar_interpretation": interpretation,
    }


def _failed_row(baseline_name: str, ft_name: str, alpha: float, msg: str) -> dict:
    return {
        "baseline_model": baseline_name,
        "fine_tuned_model": ft_name,
        "matched_samples": np.nan,
        "mcnemar_test_type": "error",
        "alpha": alpha,
        "mcnemar_p_value": np.nan,
        "significant_at_alpha": "No",
        "mcnemar_winner": "Comparison failed",
        "mcnemar_interpretation": msg,
    }


def compare_one_pair(baseline_df, baseline_name, ft_df, ft_name,
                     alpha, n_bootstrap, bootstrap_seed) -> dict:
    merged = pd.merge(baseline_df, ft_df, on=MERGE_COLS, how="inner")
    if merged.empty:
        return _failed_row(
            baseline_name, ft_name, alpha,
            "No matched rows after merging on "
            "AccessionNumber, SOPInstanceUID, and Question.",
        )

    out = {}
    out.update(run_mcnemar_comparison(merged, baseline_name, ft_name, alpha))
    out.update(bootstrap_accuracy_test(merged, n_bootstrap, alpha, bootstrap_seed))
    out.update(compute_binary_metrics(merged["y_true"], merged["baseline_pred"], "baseline"))
    out.update(compute_binary_metrics(merged["y_true"], merged["ft_pred"], "fine_tuned"))
    return out


def add_sorting_columns(results_df: pd.DataFrame) -> pd.DataFrame:
    """Significant FT wins first, then significant losses, then the rest."""
    def priority(row):
        significant = str(row.get("significant_at_alpha", "")).strip().lower() == "yes"
        ft_won = (str(row.get("mcnemar_winner", "")).strip()
                  == str(row.get("fine_tuned_model", "")).strip())
        if significant and ft_won:
            return 0
        if significant:
            return 1
        return 2

    results_df["sort_priority"] = results_df.apply(priority, axis=1)
    results_df["ft_model_statistically_significant_winner"] = results_df.apply(
        lambda r: "Yes" if priority(r) == 0 else "No", axis=1
    )
    return results_df


def run(cfg, run_dir) -> dict:
    baseline_files = sorted(
        glob.glob(os.path.join(cfg.baselines_dir, "*_predictions.csv"))
    )
    if not baseline_files:
        raise FileNotFoundError(
            f"No *_predictions.csv baselines in {cfg.baselines_dir} — "
            "run step 01 first."
        )

    ft_files = sorted(
        glob.glob(os.path.join(cfg.ft_predictions_dir, "*predictions*.csv"))
    )
    if not ft_files:
        raise FileNotFoundError(
            f"No *predictions*.csv files in {cfg.ft_predictions_dir}"
        )

    def name_of(path):
        return os.path.splitext(os.path.basename(path))[0]

    ft_data = {}
    for f in ft_files:
        try:
            ft_data[name_of(f)] = load_finetuned_predictions(f)
        except Exception as e:
            ft_data[name_of(f)] = e

    # Real baselines + synthetic controls on the first baseline's key space.
    all_baselines = {name_of(f): f for f in baseline_files}
    try:
        reference = load_baseline_predictions(baseline_files[0])
        all_baselines.update(
            build_control_baselines(reference, cfg.random_baseline_seed)
        )
    except Exception as e:
        print(f"[WARN] could not build control baselines: {e}")

    results = []
    for baseline_name, source in all_baselines.items():
        try:
            baseline_df = (source.copy() if isinstance(source, pd.DataFrame)
                           else load_baseline_predictions(source))
        except Exception as e:
            results += [
                _failed_row(baseline_name, ft, cfg.alpha,
                            f"Could not load baseline: {e}")
                for ft in ft_data
            ]
            continue

        for ft_name, ft_item in ft_data.items():
            if isinstance(ft_item, Exception):
                results.append(_failed_row(
                    baseline_name, ft_name, cfg.alpha,
                    f"Could not load fine-tuned file: {ft_item}"))
                continue
            try:
                results.append(compare_one_pair(
                    baseline_df, baseline_name, ft_item, ft_name,
                    cfg.alpha, cfg.n_bootstrap, cfg.bootstrap_seed))
            except Exception as e:
                results.append(_failed_row(
                    baseline_name, ft_name, cfg.alpha,
                    f"Error during comparison: {e}"))

    results_df = add_sorting_columns(pd.DataFrame(results))
    results_df = results_df.sort_values(
        by=["sort_priority", "mcnemar_p_value", "delta_accuracy_ft_minus_baseline"],
        ascending=[True, True, False],
        na_position="last",
    ).drop(columns=["sort_priority"]).reset_index(drop=True)

    output_csv = os.path.join(str(run_dir), "statistical_comparison.csv")
    results_df.to_csv(output_csv, index=False)

    print(f"comparisons: {len(results_df)} "
          f"({len(all_baselines)} baselines x {len(ft_data)} fine-tuned)")
    print(f"results: {output_csv}")
    print(results_df.head(10).to_string(index=False))

    return {
        "output_csv": output_csv,
        "n_baselines": len(all_baselines),
        "n_finetuned": len(ft_data),
        "n_comparisons": len(results_df),
        "ft_significant_wins": int(
            (results_df["ft_model_statistically_significant_winner"] == "Yes").sum()
        ),
    }
