"""Rerun S3 CatBoost with SSL-evaluation hyperparameters.

Compares the original S3 CatBoost defaults (depth=6, iter=1500, lr=0.03)
against the SSL evaluation config (depth=4, iter=300, lr=0.05, min_data_in_leaf=5).

Runs two variants per imbalance mode:
  A) SSL hparams WITH early stopping (eval_set passed to .fit)
  B) SSL hparams WITHOUT early stopping (trains all 300 iterations)

Uses the same S3 CV splits (StratifiedKFold seed=42) and data loading.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold
from imblearn.over_sampling import SMOTE

import sys
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from src.data_external import load_uci_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s")
logger = logging.getLogger(__name__)

# SSL evaluation CatBoost config (from ssl/ssl/evaluate.py lines 140-142)
SSL_HPARAMS = {
    "iterations": 300,
    "learning_rate": 0.05,
    "depth": 4,
    "min_data_in_leaf": 5,
    "loss_function": "Logloss",
    "eval_metric": "AUC",
    "verbose": False,
    "allow_writing_files": False,
}

DATASETS = ["cleveland", "hungarian", "switzerland", "va_longbeach"]
IMBALANCE_MODES = ["native", "smote"]


def get_cv_splitter(dataset_name, seed=42):
    if dataset_name in ("switzerland", "va_longbeach"):
        return RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=seed)
    return StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)


def compute_metrics(y_true, y_prob):
    y_pred = (y_prob >= 0.5).astype(int)
    has_both = len(np.unique(y_true)) > 1
    return {
        "auc": float(roc_auc_score(y_true, y_prob)) if has_both else float("nan"),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def run_catboost_cv(ds_name, X, y, params, imbalance="native", early_stop=True):
    cv = get_cv_splitter(ds_name)
    X_arr = X.values if hasattr(X, "values") else np.asarray(X)
    y_arr = np.asarray(y)

    fold_metrics = []
    oof_preds = np.zeros(len(y_arr), dtype=np.float64)
    oof_counts = np.zeros(len(y_arr), dtype=np.int32)

    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X_arr, y_arr)):
        x_tr = X_arr[train_idx].copy()
        x_va = X_arr[val_idx].copy()
        y_tr = y_arr[train_idx].copy()
        y_va = y_arr[val_idx].copy()

        fold_params = dict(params)

        # SMOTE oversampling on training fold
        if imbalance == "smote":
            smote = SMOTE(random_state=42 + fold_idx)
            x_tr, y_tr = smote.fit_resample(x_tr, y_tr)

        # Native class balancing
        if imbalance == "native":
            fold_params["auto_class_weights"] = "Balanced"

        model = CatBoostClassifier(random_seed=42, **fold_params)

        if early_stop:
            model.fit(x_tr, y_tr, eval_set=(x_va, y_va), verbose=False)
        else:
            model.fit(x_tr, y_tr, verbose=False)

        val_scores = model.predict_proba(x_va)[:, 1]
        fm = compute_metrics(y_va, val_scores)
        fold_metrics.append(fm)
        oof_preds[val_idx] += val_scores
        oof_counts[val_idx] += 1

    valid_mask = oof_counts > 0
    oof_preds[valid_mask] /= oof_counts[valid_mask]
    overall = compute_metrics(y_arr[valid_mask], oof_preds[valid_mask])

    fold_aucs = [fm["auc"] for fm in fold_metrics]
    return {
        "oof_auc": overall["auc"],
        "mean_fold_auc": float(np.mean(fold_aucs)),
        "std_fold_auc": float(np.std(fold_aucs)),
        "metrics": overall,
        "fold_aucs": fold_aucs,
        "n_folds": len(fold_metrics),
    }


def main():
    results = {}
    out_dir = project_root / "results" / "external" / "evals"
    out_dir.mkdir(parents=True, exist_ok=True)

    for ds_name in DATASETS:
        X, y, meta = load_uci_dataset(ds_name)
        logger.info("Dataset: %s (n=%d, prevalence=%.1f%%)",
                     ds_name, meta["n_samples"], meta["prevalence"])
        results[ds_name] = {}

        for imbalance in IMBALANCE_MODES:
            # A) SSL hparams WITH early stopping
            t0 = time.perf_counter()
            res_a = run_catboost_cv(ds_name, X, y, SSL_HPARAMS,
                                     imbalance=imbalance, early_stop=True)
            res_a["runtime_s"] = round(time.perf_counter() - t0, 2)
            res_a["config"] = "ssl_hparams_early_stop"
            results[ds_name][f"ssl_hparams_earlystop_{imbalance}"] = res_a
            logger.info("  ssl_hparams+earlystop/%s: OOF_AUC=%.4f  mean_fold=%.4f",
                         imbalance, res_a["oof_auc"], res_a["mean_fold_auc"])

            # B) SSL hparams WITHOUT early stopping
            t0 = time.perf_counter()
            res_b = run_catboost_cv(ds_name, X, y, SSL_HPARAMS,
                                     imbalance=imbalance, early_stop=False)
            res_b["runtime_s"] = round(time.perf_counter() - t0, 2)
            res_b["config"] = "ssl_hparams_no_early_stop"
            results[ds_name][f"ssl_hparams_noearlystop_{imbalance}"] = res_b
            logger.info("  ssl_hparams+noearlystop/%s: OOF_AUC=%.4f  mean_fold=%.4f",
                         imbalance, res_b["oof_auc"], res_b["mean_fold_auc"])

    # Save all results
    out_file = out_dir / "catboost_ssl_hparams_rerun.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Results saved to %s", out_file)

    # Print comparison summary
    print("\n" + "=" * 95)
    print("COMPARISON: Original S3 defaults vs SSL hparams (with/without early stop)")
    print("=" * 95)
    print(f"{'Dataset':<15} {'Imbalance':<10} {'Orig S3 OOF':<14} {'SSL+ES OOF':<14} {'SSL noES OOF':<14} {'SSL eval raw':<14}")
    print("-" * 95)

    # SSL eval raw baselines (from ssl_uci_transfer_results.json)
    ssl_raw_baselines = {
        "cleveland": 0.9198, "hungarian": 0.9033,
        "switzerland": 0.7623, "va_longbeach": 0.6938,
    }

    for ds_name in DATASETS:
        for imb in IMBALANCE_MODES:
            orig_file = out_dir / f"s3_catboost_{ds_name}_default_{imb}.json"
            if orig_file.exists():
                with open(orig_file) as f:
                    orig = json.load(f)
                orig_auc = f"{orig.get('auc_mean', orig.get('oof_auc', 'N/A')):.4f}"
            else:
                orig_auc = "N/A"

            ssl_es = results[ds_name][f"ssl_hparams_earlystop_{imb}"]["oof_auc"]
            ssl_no = results[ds_name][f"ssl_hparams_noearlystop_{imb}"]["oof_auc"]
            ssl_raw = ssl_raw_baselines.get(ds_name, "N/A")

            print(f"{ds_name:<15} {imb:<10} {orig_auc:<14} {ssl_es:<14.4f} {ssl_no:<14.4f} {ssl_raw:<14}")

    print("\nNote: 'SSL eval raw' = Raw(13d)->CatBoost from ssl_uci_transfer_results.json")
    print("      (no class balancing, no early stopping, same folds)")


if __name__ == "__main__":
    main()
