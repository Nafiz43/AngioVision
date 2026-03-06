#!/usr/bin/env python3

import pandas as pd
import argparse

DEFAULT_PRED_PATH = "/data/Deep_Angiography/AngioVision/fine-tuning/output/clip_binary_qa_predictions.csv"
DEFAULT_GT_PATH   = "/data/Deep_Angiography/Validation_Data/Validation_Data_2026_03_04/VLM_Test_Data_2026_03_04_v01.csv"
DEFAULT_OUT_PATH  = "missing_gt_rows.csv"


def normalize_str_series(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.lower()


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_path", default=DEFAULT_PRED_PATH)
    parser.add_argument("--gt_path", default=DEFAULT_GT_PATH)
    parser.add_argument("--out_path", default=DEFAULT_OUT_PATH)

    args = parser.parse_args()

    print("\nLoading files...")
    print("Pred:", args.pred_path)
    print("GT:  ", args.gt_path)

    pred = pd.read_csv(args.pred_path)
    gt   = pd.read_csv(args.gt_path)

    # normalize columns
    pred["AccessionNumber"] = normalize_str_series(pred["AccessionNumber"])
    pred["SOPInstanceUID"]  = normalize_str_series(pred["SOPInstanceUID"])
    pred["Question"]        = normalize_str_series(pred["Question"])

    gt["Accession"]        = normalize_str_series(gt["Accession"])
    gt["SOPInstanceUID"]   = normalize_str_series(gt["SOPInstanceUID"])
    gt["Question"]         = normalize_str_series(gt["Question"])

    pred_std = pred.rename(columns={
        "AccessionNumber": "accession",
        "SOPInstanceUID": "sopinstanceuid",
        "Question": "question"
    })[["accession","sopinstanceuid","question"]]

    gt_std = gt.rename(columns={
        "Accession": "accession",
        "SOPInstanceUID": "sopinstanceuid",
        "Question": "question"
    })

    # left join to find GT rows without predictions
    merged = gt_std.merge(
        pred_std,
        how="left",
        on=["accession","sopinstanceuid","question"],
        indicator=True
    )

    missing = merged[merged["_merge"] == "left_only"].copy()

    print("\nGT rows without prediction:", len(missing))

    # save them
    missing.to_csv(args.out_path, index=False)

    print("Saved to:", args.out_path)


if __name__ == "__main__":
    main()