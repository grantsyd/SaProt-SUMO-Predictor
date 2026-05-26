# -*- coding: utf-8 -*-
"""
NearMiss_Dataset2.py

Apply NearMiss undersampling to Dataset2 and create a fixed 7:3 split.

Dataset2 setting used in the thesis:
    positive = 4493
    negative = 24456
    total    = 28949

After NearMiss:
    positive = 4493
    negative = 5178
    total    = 9671

Then:
    stratified train/test split = 7:3
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from collections import Counter

import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import NearMiss

EXPECTED_TOTAL = 28949
EXPECTED_POSITIVE = 4493
EXPECTED_NEGATIVE = 24456
TARGET_NEGATIVE = 5178


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
    print("Dataset2 original:", count_labels(labels))
    return labels


def make_nearmiss_split(X, y, target_negative, test_size, random_state):
    sampling_strategy = {0: target_negative, 1: int(np.sum(y == 1))}
    nearmiss = NearMiss(version=1, n_neighbors=3, sampling_strategy=sampling_strategy)
    X_res, y_res = nearmiss.fit_resample(X, y)

    sampled_indices = nearmiss.sample_indices_.astype(np.int64)
    print("After NearMiss:", count_labels(y_res))

    train_local, test_local = train_test_split(
        np.arange(len(sampled_indices)),
        test_size=test_size,
        random_state=random_state,
        shuffle=True,
        stratify=y_res,
    )

    train_idx = sampled_indices[train_local].astype(np.int64)
    test_idx = sampled_indices[test_local].astype(np.int64)

    return sampled_indices, train_idx, test_idx


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="Data/dataset2.npy")
    parser.add_argument("--feature", default="features/dataset2_saprot_center_mean_AE384.npy")
    parser.add_argument("--out_dir", default="splits/ae_family_nearmiss5178")
    parser.add_argument("--prefix", default="dataset2_nearmiss_saprot_center_mean_AE384_neg5178_seed42")
    parser.add_argument("--target_negative", type=int, default=TARGET_NEGATIVE)
    parser.add_argument("--test_size", type=float, default=0.30)
    parser.add_argument("--random_state", type=int, default=42)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    y = load_labels(args.dataset)
    X = np.load(args.feature).astype(np.float32)

    if X.shape[0] != len(y):
        raise ValueError(f"Feature rows {X.shape[0]} do not match labels {len(y)}.")

    sampled_idx, train_idx, test_idx = make_nearmiss_split(
        X,
        y,
        target_negative=args.target_negative,
        test_size=args.test_size,
        random_state=args.random_state,
    )

    np.save(out_dir / f"{args.prefix}_indices.npy", sampled_idx)
    np.save(out_dir / f"{args.prefix}_train_idx.npy", train_idx)
    np.save(out_dir / f"{args.prefix}_test_idx.npy", test_idx)

    summary = {
        "dataset_file": args.dataset,
        "feature_file": args.feature,
        "sampled": count_labels(y[sampled_idx]),
        "train": count_labels(y[train_idx]),
        "test": count_labels(y[test_idx]),
        "test_size": args.test_size,
        "random_state": args.random_state,
    }

    with open(out_dir / f"{args.prefix}_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    with open(out_dir / f"{args.prefix}.pkl", "wb") as f:
        pickle.dump({
            "sampled_indices": sampled_idx,
            "train_idx": train_idx,
            "test_idx": test_idx,
            "summary": summary,
        }, f)

    print("Train:", summary["train"])
    print("Test :", summary["test"])
    print(f"Saved split files to: {out_dir}")


if __name__ == "__main__":
    main()
