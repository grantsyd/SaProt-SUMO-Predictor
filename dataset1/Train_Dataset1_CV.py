# -*- coding: utf-8 -*-
"""
Train_Dataset1_CV.py

Train and evaluate Dataset1 models using 10-fold stratified cross-validation.

Input examples:
    resampled_features/dataset1_saprot_center_mean_smoteenn_X.npy
    resampled_features/dataset1_saprot_center_mean_smoteenn_y.npy

Metrics:
    Acc, Sn, Sp, MCC, AUC
"""

from __future__ import annotations

import argparse
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, matthews_corrcoef, roc_auc_score, confusion_matrix
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


def build_model(model_name: str, random_state: int = 42):
    model_name = model_name.lower()
    if model_name == "xgboost":
        return XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.85,
            colsample_bytree=0.85,
            objective="binary:logistic",
            eval_metric="auc",
            tree_method="hist",
            random_state=random_state,
            n_jobs=-1,
        )
    if model_name == "lightgbm":
        return LGBMClassifier(
            n_estimators=400,
            learning_rate=0.03,
            num_leaves=31,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=random_state,
            n_jobs=-1,
        )
    if model_name == "extratrees":
        return ExtraTreesClassifier(
            n_estimators=500,
            max_features="sqrt",
            random_state=random_state,
            n_jobs=-1,
        )
    if model_name == "randomforest":
        return RandomForestClassifier(
            n_estimators=500,
            max_features="sqrt",
            random_state=random_state,
            n_jobs=-1,
        )
    if model_name == "softvoting":
        et = ExtraTreesClassifier(n_estimators=500, max_features="sqrt", random_state=random_state, n_jobs=-1)
        xgb = XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=5, eval_metric="auc",
                            tree_method="hist", random_state=random_state, n_jobs=-1)
        lgbm = LGBMClassifier(n_estimators=400, learning_rate=0.03, random_state=random_state, n_jobs=-1)
        return VotingClassifier(
            estimators=[("et", et), ("xgb", xgb), ("lgbm", lgbm)],
            voting="soft",
            weights=[1, 1, 1],
            n_jobs=-1,
        )
    raise ValueError(f"Unknown model name: {model_name}")


def calc_metrics(y_true, prob, threshold: float = 0.5) -> dict:
    pred = (prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    sn = tp / (tp + fn) if (tp + fn) else 0.0
    sp = tn / (tn + fp) if (tn + fp) else 0.0
    return {
        "Acc": accuracy_score(y_true, pred),
        "Sn": sn,
        "Sp": sp,
        "MCC": matthews_corrcoef(y_true, pred),
        "AUC": roc_auc_score(y_true, prob),
        "TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn),
    }


def run_cv(X, y, model_name: str, n_splits: int, random_state: int, threshold: float):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    base_model = build_model(model_name, random_state=random_state)
    rows = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
        model = clone(base_model)
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        model.fit(X_train, y_train)
        prob = model.predict_proba(X_val)[:, 1]
        metrics = calc_metrics(y_val, prob, threshold=threshold)
        metrics.update({
            "fold": fold,
            "train_total": int(len(train_idx)),
            "val_total": int(len(val_idx)),
            "train_pos": int(np.sum(y_train == 1)),
            "train_neg": int(np.sum(y_train == 0)),
            "val_pos": int(np.sum(y_val == 1)),
            "val_neg": int(np.sum(y_val == 0)),
        })
        rows.append(metrics)
        print(f"Fold {fold}: MCC={metrics['MCC']:.4f}, AUC={metrics['AUC']:.4f}")

    fold_df = pd.DataFrame(rows)
    summary = fold_df[["Acc", "Sn", "Sp", "MCC", "AUC"]].agg(["mean", "std"]).reset_index()
    return fold_df, summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--X", default="resampled_features/dataset1_saprot_center_mean_smoteenn_X.npy")
    parser.add_argument("--y", default="resampled_features/dataset1_saprot_center_mean_smoteenn_y.npy")
    parser.add_argument("--model", default="xgboost", choices=["xgboost", "lightgbm", "extratrees", "randomforest", "softvoting"])
    parser.add_argument("--out_dir", default="results")
    parser.add_argument("--n_splits", type=int, default=10)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--random_state", type=int, default=42)
    args = parser.parse_args()

    X = np.load(args.X).astype(np.float32)
    y = np.load(args.y).astype(np.int64)
    print("Label distribution:", Counter(y.tolist()))

    fold_df, summary_df = run_cv(X, y, args.model, args.n_splits, args.random_state, args.threshold)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"dataset1_{Path(args.X).stem}_{args.model}_{args.n_splits}fold"
    fold_df.to_csv(out_dir / f"{prefix}_folds.csv", index=False, encoding="utf-8-sig")
    summary_df.to_csv(out_dir / f"{prefix}_summary.csv", index=False, encoding="utf-8-sig")
    print(summary_df)


if __name__ == "__main__":
    main()
