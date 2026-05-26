# -*- coding: utf-8 -*-
"""
SMOTE_ENN_Dataset1.py

Apply SMOTE-ENN resampling to Dataset1 features.

Dataset1 setting used in the thesis:
    positive = 775
    negative = 17807
    total    = 18582

Main idea:
    1. Load Dataset1 labels.
    2. Load a feature matrix, such as dataset1_saprot_center_mean.npy.
    3. Apply SMOTE to expand positive samples.
    4. Apply ENN to clean majority-class noisy/boundary samples.
    5. Save the resampled feature matrix and labels.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import EditedNearestNeighbours
from sklearn.utils import shuffle

EXPECTED_TOTAL = 18582
EXPECTED_POSITIVE = 775
EXPECTED_NEGATIVE = 17807
DEFAULT_TARGET_POSITIVE = 9291


def count_labels(y: np.ndarray) -> dict:
    c = Counter(np.asarray(y, dtype=np.int64).tolist())
    return {
        "negative": int(c.get(0, 0)),
        "positive": int(c.get(1, 0)),
        "total": int(len(y)),
    }


def load_labels(dataset_file: str | Path) -> np.ndarray:
    data = np.load(dataset_file, allow_pickle=True).item()
    labels = np.asarray(data["labels"], dtype=np.int64)

    if len(labels) != EXPECTED_TOTAL:
        print(f"Warning: Dataset1 total is {len(labels)}, expected {EXPECTED_TOTAL}.")
    if int(np.sum(labels == 1)) != EXPECTED_POSITIVE:
        print(f"Warning: positive count is {int(np.sum(labels == 1))}, expected {EXPECTED_POSITIVE}.")
    if int(np.sum(labels == 0)) != EXPECTED_NEGATIVE:
        print(f"Warning: negative count is {int(np.sum(labels == 0))}, expected {EXPECTED_NEGATIVE}.")

    return labels


def apply_smote_enn(
    X: np.ndarray,
    y: np.ndarray,
    target_positive: int = DEFAULT_TARGET_POSITIVE,
    random_state: int = 42,
):
    print("Original:", count_labels(y))

    smote = SMOTE(
        sampling_strategy={1: target_positive},
        k_neighbors=5,
        random_state=random_state,
    )
    X_smote, y_smote = smote.fit_resample(X, y)
    print("After SMOTE:", count_labels(y_smote))

    enn = EditedNearestNeighbours(
        sampling_strategy="majority",
        n_neighbors=3,
        kind_sel="all",
    )
    X_res, y_res = enn.fit_resample(X_smote, y_smote)
    print("After ENN:", count_labels(y_res))

    X_res, y_res = shuffle(X_res, y_res, random_state=random_state)
    return X_res.astype(np.float32), y_res.astype(np.int64)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="Data/dataset1.npy")
    parser.add_argument("--feature", default="features/dataset1_saprot_center_mean.npy")
    parser.add_argument("--out_dir", default="resampled_features")
    parser.add_argument("--feature_name", default=None)
    parser.add_argument("--target_positive", type=int, default=DEFAULT_TARGET_POSITIVE)
    parser.add_argument("--random_state", type=int, default=42)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    feature_file = Path(args.feature)
    feature_name = args.feature_name or feature_file.stem.replace("dataset1_", "")

    y = load_labels(args.dataset)
    X = np.load(feature_file).astype(np.float32)

    if X.shape[0] != len(y):
        raise ValueError(f"Feature rows {X.shape[0]} do not match labels {len(y)}.")

    X_res, y_res = apply_smote_enn(
        X,
        y,
        target_positive=args.target_positive,
        random_state=args.random_state,
    )

    x_out = out_dir / f"dataset1_{feature_name}_smoteenn_X.npy"
    y_out = out_dir / f"dataset1_{feature_name}_smoteenn_y.npy"
    summary_out = out_dir / f"dataset1_{feature_name}_smoteenn_summary.json"

    np.save(x_out, X_res)
    np.save(y_out, y_res)

    summary = {
        "dataset": str(args.dataset),
        "feature": str(feature_file),
        "feature_name": feature_name,
        "before": count_labels(y),
        "after": count_labels(y_res),
        "x_out": str(x_out),
        "y_out": str(y_out),
    }
    with open(summary_out, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    pd.DataFrame([summary["before"], summary["after"]], index=["before", "after"]).to_csv(
        out_dir / f"dataset1_{feature_name}_smoteenn_summary.csv", encoding="utf-8-sig"
    )

    print(f"Saved: {x_out}")
    print(f"Saved: {y_out}")


if __name__ == "__main__":
    main()
