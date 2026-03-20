#!/usr/bin/env python3

import os
import glob
import numpy as np
import pandas as pd
from statsmodels.stats.contingency_tables import mcnemar

# =========================================================
# CONFIG
# =========================================================
BASELINE_FILES = {
    "Gemma3_27B": "/data/Deep_Angiography/AngioVision/frame-processing/model_runs/gemma3_27b_predictions.csv",
    "GPT_OSS_20B": "/data/Deep_Angiography/AngioVision/frame-processing/model_runs/gpt-oss_20b_predictions.csv",
    "Llama3.2_Vision_11B": "/data/Deep_Angiography/AngioVision/frame-processing/model_runs/llama3.2-vision_11b_predictions.csv",
    "LLaVA_34B": "/data/Deep_Angiography/AngioVision/frame-processing/model_runs/llava_34b_predictions.csv",
    "Qwen3_VL_8B": "/data/Deep_Angiography/AngioVision/frame-processing/model_runs/qwen3-vl_8b_predictions.csv",
    "Qwen3_VL_32B": "/data/Deep_Angiography/AngioVision/frame-processing/model_runs/qwen3-vl_32b_predictions.csv",
}

FT_DIR = "/data/Deep_Angiography/AngioVision/fine-tuning/output"
OUTPUT_CSV = "/data/Deep_Angiography/AngioVision/fine-tuning/statistical-test-result/statistical_comparison_all_baselines_vs_all_finetuned_with_controls.csv"

MERGE_COLS = ["AccessionNumber", "SOPInstanceUID", "Question"]

# User-requested looser threshold
ALPHA = 0.15

# Bootstrap settings
N_BOOTSTRAP = 2000
BOOTSTRAP_SEED = 42

# Random baseline settings
RANDOM_BASELINE_SEED = 12345


# =========================================================
# HELPERS
# =========================================================
def normalize_text(x):
    if pd.isna(x):
        return ""
    return str(x).strip().lower()


def normalize_binary(x):
    """
    Convert common binary label formats into 0/1.
    Adjust mappings if your labels use a custom scheme.
    """
    x_norm = normalize_text(x)

    positive_values = {
        "1", "true", "yes", "y", "positive", "pos",
        "present", "abnormal", "disease", "stenosis"
    }
    negative_values = {
        "0", "false", "no", "n", "negative", "neg",
        "absent", "normal", "no disease", "no stenosis"
    }

    if x_norm in positive_values:
        return 1
    if x_norm in negative_values:
        return 0

    try:
        return 1 if float(x_norm) == 1 else 0
    except Exception:
        return 0


def extract_model_name_from_filename(path):
    return os.path.splitext(os.path.basename(path))[0]


def safe_key_normalization(df):
    """
    Normalize merge key columns to reduce accidental mismatches.
    """
    for col in MERGE_COLS:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
    return df


def load_baseline_predictions(path):
    """
    Expected baseline columns:
    Timestamp, Model Name, AccessionNumber, SOPInstanceUID, Question,
    GroundTruth, Predicted, Raw_LLM_Output, sequence_dir, mosaic_path
    """
    df = pd.read_csv(path)

    required_cols = ["AccessionNumber", "SOPInstanceUID", "Question", "GroundTruth", "Predicted"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Baseline file missing required columns: {missing}")

    df = df.rename(columns={
        "GroundTruth": "y_true",
        "Predicted": "baseline_pred"
    })

    df = safe_key_normalization(df)

    df["y_true"] = df["y_true"].apply(normalize_binary)
    df["baseline_pred"] = df["baseline_pred"].apply(normalize_binary)

    keep_cols = MERGE_COLS + ["y_true", "baseline_pred"]
    df = df[keep_cols].drop_duplicates()

    return df


def load_finetuned_predictions(path):
    """
    Expected fine-tuned columns:
    AccessionNumber, SOPInstanceUID, Question, Answer
    """
    df = pd.read_csv(path)

    required_cols = ["AccessionNumber", "SOPInstanceUID", "Question", "Answer"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Fine-tuned file missing required columns: {missing}")

    df = df.rename(columns={"Answer": "ft_pred"})

    df = safe_key_normalization(df)

    df["ft_pred"] = df["ft_pred"].apply(normalize_binary)

    keep_cols = MERGE_COLS + ["ft_pred"]
    df = df[keep_cols].drop_duplicates()

    return df


def build_control_baselines_from_reference(reference_df, random_seed=12345):
    """
    Build three synthetic baseline models on the same matched keys:
      1) Random_Seeded
      2) All_Yes
      3) All_No

    reference_df must contain:
      AccessionNumber, SOPInstanceUID, Question, y_true
    """
    ref = reference_df[MERGE_COLS + ["y_true"]].drop_duplicates().copy()

    rng = np.random.default_rng(random_seed)

    random_df = ref.copy()
    random_df["baseline_pred"] = rng.integers(0, 2, size=len(random_df))

    all_yes_df = ref.copy()
    all_yes_df["baseline_pred"] = 1

    all_no_df = ref.copy()
    all_no_df["baseline_pred"] = 0

    controls = {
        f"Random_Seeded_{random_seed}": random_df,
        "All_Yes_Model": all_yes_df,
        "All_No_Model": all_no_df,
    }
    return controls


def bootstrap_accuracy_test(df_merged, n_bootstrap=2000, alpha=0.15, seed=42):
    """
    Bootstrap CI for delta accuracy:
        fine-tuned accuracy - baseline accuracy
    Significant if CI does not include 0.
    """
    rng = np.random.default_rng(seed)

    y_true = df_merged["y_true"].to_numpy()
    baseline_pred = df_merged["baseline_pred"].to_numpy()
    ft_pred = df_merged["ft_pred"].to_numpy()

    n = len(df_merged)
    if n == 0:
        return {
            "baseline_accuracy": np.nan,
            "fine_tuned_accuracy": np.nan,
            "delta_accuracy_ft_minus_baseline": np.nan,
            "bootstrap_ci_lower": np.nan,
            "bootstrap_ci_upper": np.nan,
            "bootstrap_significant_at_alpha": "No",
            "bootstrap_interpretation": "Bootstrap comparison could not be computed because no matched samples were available."
        }

    baseline_acc = float(np.mean(baseline_pred == y_true))
    ft_acc = float(np.mean(ft_pred == y_true))

    diffs = np.empty(n_bootstrap, dtype=float)

    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        y_sample = y_true[idx]
        baseline_sample = baseline_pred[idx]
        ft_sample = ft_pred[idx]

        acc_baseline_sample = np.mean(baseline_sample == y_sample)
        acc_ft_sample = np.mean(ft_sample == y_sample)

        diffs[i] = acc_ft_sample - acc_baseline_sample

    lower = float(np.percentile(diffs, 100 * (alpha / 2)))
    upper = float(np.percentile(diffs, 100 * (1 - alpha / 2)))
    delta = float(np.mean(diffs))

    if lower > 0:
        significant = "Yes"
        interpretation = (
            f"The fine-tuned model shows a statistically meaningful accuracy improvement "
            f"over the baseline under bootstrap resampling at alpha={alpha:.2f}."
        )
    elif upper < 0:
        significant = "Yes"
        interpretation = (
            f"The baseline model shows a statistically meaningful accuracy improvement "
            f"over the fine-tuned model under bootstrap resampling at alpha={alpha:.2f}."
        )
    else:
        significant = "No"
        interpretation = (
            f"No statistically meaningful accuracy difference was observed under bootstrap "
            f"resampling at alpha={alpha:.2f}."
        )

    return {
        "baseline_accuracy": baseline_acc,
        "fine_tuned_accuracy": ft_acc,
        "delta_accuracy_ft_minus_baseline": delta,
        "bootstrap_ci_lower": lower,
        "bootstrap_ci_upper": upper,
        "bootstrap_significant_at_alpha": significant,
        "bootstrap_interpretation": interpretation
    }


def run_mcnemar_comparison(df_merged, model_a_name, model_b_name, alpha=0.15):
    """
    model_a = baseline
    model_b = fine-tuned
    """
    baseline_correct = df_merged["baseline_pred"] == df_merged["y_true"]
    ft_correct = df_merged["ft_pred"] == df_merged["y_true"]

    both_correct = int((baseline_correct & ft_correct).sum())
    both_wrong = int((~baseline_correct & ~ft_correct).sum())

    # Discordant counts
    b = int((~baseline_correct & ft_correct).sum())   # baseline wrong, FT correct
    c = int((baseline_correct & ~ft_correct).sum())   # baseline correct, FT wrong

    table = [[both_correct, c],
             [b, both_wrong]]

    discordant_total = b + c

    if discordant_total == 0:
        p_value = 1.0
        test_type = "not_applicable"
    else:
        use_exact = discordant_total < 20
        result = mcnemar(table, exact=use_exact, correction=not use_exact)
        p_value = float(result.pvalue)
        test_type = "exact" if use_exact else "chi-square with continuity correction"

    if p_value < alpha:
        significant = "Yes"
        if b > c:
            winner = model_b_name
            interpretation = (
                f"{model_b_name} outperforms {model_a_name} on paired samples, "
                f"and the difference is statistically significant by McNemar's test at alpha={alpha:.2f}."
            )
        elif c > b:
            winner = model_a_name
            interpretation = (
                f"{model_a_name} outperforms {model_b_name} on paired samples, "
                f"and the difference is statistically significant by McNemar's test at alpha={alpha:.2f}."
            )
        else:
            winner = "Tie"
            interpretation = (
                f"A statistically significant difference was detected by McNemar's test at alpha={alpha:.2f}, "
                "but discordant wins are balanced."
            )
    else:
        significant = "No"
        if b > c:
            lean = model_b_name
        elif c > b:
            lean = model_a_name
        else:
            lean = "Neither model"

        winner = "No significant difference"
        interpretation = (
            f"No statistically significant difference was observed by McNemar's test at alpha={alpha:.2f}. "
            f"Discordant counts lean toward: {lean}."
        )

    return {
        "baseline_model": model_a_name,
        "fine_tuned_model": model_b_name,
        "matched_samples": int(len(df_merged)),
        "both_correct": both_correct,
        "both_wrong": both_wrong,
        "baseline_wrong_ft_correct_b": b,
        "baseline_correct_ft_wrong_c": c,
        "discordant_total_b_plus_c": discordant_total,
        "mcnemar_test_type": test_type,
        "alpha": alpha,
        "mcnemar_p_value": p_value,
        "significant_at_alpha": significant,
        "mcnemar_winner": winner,
        "mcnemar_interpretation": interpretation,
    }


def compare_one_pair(baseline_df, baseline_name, ft_df, ft_name, alpha=0.15, n_bootstrap=2000):
    merged = pd.merge(
        baseline_df,
        ft_df,
        on=MERGE_COLS,
        how="inner"
    )

    if merged.empty:
        return {
            "baseline_model": baseline_name,
            "fine_tuned_model": ft_name,
            "matched_samples": 0,
            "both_correct": 0,
            "both_wrong": 0,
            "baseline_wrong_ft_correct_b": 0,
            "baseline_correct_ft_wrong_c": 0,
            "discordant_total_b_plus_c": 0,
            "mcnemar_test_type": "not_run",
            "alpha": alpha,
            "mcnemar_p_value": 1.0,
            "significant_at_alpha": "No",
            "mcnemar_winner": "Comparison not possible",
            "mcnemar_interpretation": "No matched rows found after merging on AccessionNumber, SOPInstanceUID, and Question.",
            "baseline_accuracy": np.nan,
            "fine_tuned_accuracy": np.nan,
            "delta_accuracy_ft_minus_baseline": np.nan,
            "bootstrap_ci_lower": np.nan,
            "bootstrap_ci_upper": np.nan,
            "bootstrap_significant_at_alpha": "No",
            "bootstrap_interpretation": "Bootstrap comparison could not be computed because no matched samples were available.",
        }

    mcnemar_result = run_mcnemar_comparison(
        merged,
        model_a_name=baseline_name,
        model_b_name=ft_name,
        alpha=alpha
    )

    bootstrap_result = bootstrap_accuracy_test(
        merged,
        n_bootstrap=n_bootstrap,
        alpha=alpha,
        seed=BOOTSTRAP_SEED
    )

    out = {}
    out.update(mcnemar_result)
    out.update(bootstrap_result)
    return out


# =========================================================
# MAIN
# =========================================================
def main():
    ft_pattern = os.path.join(FT_DIR, "*predictions*.csv")
    ft_files = sorted(glob.glob(ft_pattern))

    if not ft_files:
        raise FileNotFoundError(f"No files containing 'predictions' found in: {FT_DIR}")

    results = []

    # Preload fine-tuned files once
    ft_data = {}
    for ft_file in ft_files:
        ft_model_name = extract_model_name_from_filename(ft_file)
        try:
            ft_df = load_finetuned_predictions(ft_file)
            ft_data[ft_model_name] = ft_df
        except Exception as e:
            ft_data[ft_model_name] = e

    # Build a reference df for control baselines using Gemma's matched key space
    control_baselines = {}
    try:
        gemma_reference = load_baseline_predictions(BASELINE_FILES["Gemma3_27B"])
        control_baselines = build_control_baselines_from_reference(
            gemma_reference,
            random_seed=RANDOM_BASELINE_SEED
        )
    except Exception as e:
        print(f"Warning: could not create control baselines from Gemma reference: {e}")

    # Combine real baselines + synthetic controls
    all_baselines = {}

    for baseline_name, baseline_path in BASELINE_FILES.items():
        all_baselines[baseline_name] = baseline_path

    for control_name, control_df in control_baselines.items():
        all_baselines[control_name] = control_df

    # Compare each baseline against each fine-tuned file
    for baseline_name, baseline_source in all_baselines.items():
        # baseline_source may be a path or an already-built dataframe
        try:
            if isinstance(baseline_source, pd.DataFrame):
                baseline_df = baseline_source.copy()
            else:
                baseline_df = load_baseline_predictions(baseline_source)
        except Exception as e:
            for ft_model_name in ft_data.keys():
                results.append({
                    "baseline_model": baseline_name,
                    "fine_tuned_model": ft_model_name,
                    "matched_samples": np.nan,
                    "both_correct": np.nan,
                    "both_wrong": np.nan,
                    "baseline_wrong_ft_correct_b": np.nan,
                    "baseline_correct_ft_wrong_c": np.nan,
                    "discordant_total_b_plus_c": np.nan,
                    "mcnemar_test_type": "error",
                    "alpha": ALPHA,
                    "mcnemar_p_value": np.nan,
                    "significant_at_alpha": "No",
                    "mcnemar_winner": "Comparison failed",
                    "mcnemar_interpretation": f"Could not load baseline source: {str(e)}",
                    "baseline_accuracy": np.nan,
                    "fine_tuned_accuracy": np.nan,
                    "delta_accuracy_ft_minus_baseline": np.nan,
                    "bootstrap_ci_lower": np.nan,
                    "bootstrap_ci_upper": np.nan,
                    "bootstrap_significant_at_alpha": "No",
                    "bootstrap_interpretation": "Bootstrap comparison was not run because the baseline source could not be loaded.",
                })
            continue

        for ft_model_name, ft_item in ft_data.items():
            if isinstance(ft_item, Exception):
                results.append({
                    "baseline_model": baseline_name,
                    "fine_tuned_model": ft_model_name,
                    "matched_samples": np.nan,
                    "both_correct": np.nan,
                    "both_wrong": np.nan,
                    "baseline_wrong_ft_correct_b": np.nan,
                    "baseline_correct_ft_wrong_c": np.nan,
                    "discordant_total_b_plus_c": np.nan,
                    "mcnemar_test_type": "error",
                    "alpha": ALPHA,
                    "mcnemar_p_value": np.nan,
                    "significant_at_alpha": "No",
                    "mcnemar_winner": "Comparison failed",
                    "mcnemar_interpretation": f"Could not load fine-tuned file: {str(ft_item)}",
                    "baseline_accuracy": np.nan,
                    "fine_tuned_accuracy": np.nan,
                    "delta_accuracy_ft_minus_baseline": np.nan,
                    "bootstrap_ci_lower": np.nan,
                    "bootstrap_ci_upper": np.nan,
                    "bootstrap_significant_at_alpha": "No",
                    "bootstrap_interpretation": "Bootstrap comparison was not run because the fine-tuned file could not be loaded.",
                })
                continue

            try:
                row = compare_one_pair(
                    baseline_df=baseline_df,
                    baseline_name=baseline_name,
                    ft_df=ft_item,
                    ft_name=ft_model_name,
                    alpha=ALPHA,
                    n_bootstrap=N_BOOTSTRAP
                )
                results.append(row)

            except Exception as e:
                results.append({
                    "baseline_model": baseline_name,
                    "fine_tuned_model": ft_model_name,
                    "matched_samples": np.nan,
                    "both_correct": np.nan,
                    "both_wrong": np.nan,
                    "baseline_wrong_ft_correct_b": np.nan,
                    "baseline_correct_ft_wrong_c": np.nan,
                    "discordant_total_b_plus_c": np.nan,
                    "mcnemar_test_type": "error",
                    "alpha": ALPHA,
                    "mcnemar_p_value": np.nan,
                    "significant_at_alpha": "No",
                    "mcnemar_winner": "Comparison failed",
                    "mcnemar_interpretation": f"Error during comparison: {str(e)}",
                    "baseline_accuracy": np.nan,
                    "fine_tuned_accuracy": np.nan,
                    "delta_accuracy_ft_minus_baseline": np.nan,
                    "bootstrap_ci_lower": np.nan,
                    "bootstrap_ci_upper": np.nan,
                    "bootstrap_significant_at_alpha": "No",
                    "bootstrap_interpretation": "Bootstrap comparison was not completed because the pairwise comparison failed.",
                })

    results_df = pd.DataFrame(results)

    # Sort rows: significance Yes on top, No below, then by McNemar p-value
    significance_order = {"Yes": 0, "No": 1}
    results_df["significance_sort_key"] = results_df["significant_at_alpha"].map(significance_order).fillna(2)

    results_df = results_df.sort_values(
        by=["significance_sort_key", "mcnemar_p_value"],
        ascending=[True, True],
        na_position="last"
    ).drop(columns=["significance_sort_key"]).reset_index(drop=True)

    # Save
    results_df.to_csv(OUTPUT_CSV, index=False)
    abs_path = os.path.abspath(OUTPUT_CSV)

    print("\n" + "=" * 80)
    print("McNemar + Bootstrap Analysis Complete")
    print("=" * 80)
    print(f"Results saved to:\n{abs_path}")
    print("=" * 80 + "\n")

    print("Preview of results:\n")
    print(results_df.head(15).to_string(index=False))


if __name__ == "__main__":
    main()