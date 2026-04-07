"""Scenario 1 & 2: Zero-shot transfer from competition to UCI datasets.

Train models on the full 630K competition dataset and evaluate predictions
on UCI clinical datasets without any retraining.

Scenario 1: Raw features only (13 clinical variables).
Scenario 2: Raw + competition-derived target statistics.
  Competition stats are NOT leakage because S6E2 data is external to the
  UCI clinical datasets. Note: S6E2 was CTGAN-generated from UCI Cleveland,
  so Cleveland results with competition stats should be interpreted with
  this caveat in the paper.

For each model family, the script:
  1. Loads the full competition training set and builds features
  2. Trains the model with competition-verified parameters (no CV, no val split)
  3. Builds the same features on each UCI dataset
  4. Predicts on aligned columns and computes metrics

Also runs a rank-blend ensemble of CatBoost, XGBoost, LightGBM.

Usage:
    .venv/bin/python -m experiments.external.zero_shot_transfer
    .venv/bin/python -m experiments.external.zero_shot_transfer --scenario 1
    .venv/bin/python -m experiments.external.zero_shot_transfer --models catboost xgboost
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from scipy.stats import rankdata

# PyTorch must be imported before CatBoost on Python 3.14
import torch  # noqa: F401

# Competition data and feature engineering
from src.config import FILE_ORDER_FEATURES
from src.data import load_train_test
from src.features import FEATURE_BUILDERS

# UCI dataset loading
from src.data_external import load_uci_dataset

# Fold trainers for model training
from src.models.boosting import FOLD_TRAINERS as BOOSTING_TRAINERS
from src.models.linear import FOLD_TRAINERS as LINEAR_TRAINERS
from src.models.trees import FOLD_TRAINERS as TREE_TRAINERS
from src.models.neighbors import FOLD_TRAINERS as NEIGHBOR_TRAINERS
from src.models.neural import FOLD_TRAINERS as NEURAL_TRAINERS
from src.models.tabular_dl import FOLD_TRAINERS as TABULAR_DL_TRAINERS

# Competition-verified hyperparameters
from experiments.external.tuning import DEFAULT_PARAMS, get_competition_params

# Per-fold scaling for models that need it
from sklearn.preprocessing import StandardScaler
from experiments.external.cv_utils import scale_fold

logger = logging.getLogger(__name__)

ALL_FOLD_TRAINERS = {}
ALL_FOLD_TRAINERS.update(BOOSTING_TRAINERS)
ALL_FOLD_TRAINERS.update(LINEAR_TRAINERS)
ALL_FOLD_TRAINERS.update(TREE_TRAINERS)
ALL_FOLD_TRAINERS.update(NEIGHBOR_TRAINERS)
ALL_FOLD_TRAINERS.update(NEURAL_TRAINERS)
ALL_FOLD_TRAINERS.update(TABULAR_DL_TRAINERS)
ALL_FOLD_TRAINERS["lasso"] = LINEAR_TRAINERS["logistic_regression"]
ALL_FOLD_TRAINERS["elastic_net"] = LINEAR_TRAINERS["logistic_regression"]

EXTERNAL_DATASETS = ["cleveland", "hungarian", "switzerland", "va_longbeach"]

# Models that need feature scaling before training
NEEDS_SCALING = {"svc", "svm", "knn", "logistic_regression", "ridge", "lasso", "elastic_net"}

# Families to exclude from zero-shot transfer.
# TabPFN and RealMLP are designed for small-N CV, not 630K training.
EXCLUDE_FAMILIES = {"tabpfn", "realmlp"}

# Families used in S1 (raw features).
S1_FAMILIES = [
    "catboost", "xgboost", "lightgbm",
    "random_forest", "extra_trees",
    "logistic_regression", "ridge", "lasso", "elastic_net",
    "svc", "knn", "pytorch_mlp",
]

# Families used in S2 (raw + competition stats): tree models + logistic
S2_FAMILIES = [
    "catboost", "xgboost", "lightgbm",
    "random_forest", "extra_trees",
    "logistic_regression",
]

# Families for the rank-blend ensemble
ENSEMBLE_FAMILIES = ["catboost", "xgboost", "lightgbm"]


# -----------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------

def _compute_metrics(y_true, y_prob):
    """Compute the full metric suite including optimal threshold."""
    y_pred = (y_prob >= 0.5).astype(int)
    has_both = len(np.unique(y_true)) > 1

    threshold = 0.5
    if has_both:
        fpr, tpr, thresholds = roc_curve(y_true, y_prob)
        threshold = float(thresholds[np.argmax(tpr - fpr)])

    return {
        "auc": float(roc_auc_score(y_true, y_prob)) if has_both else float("nan"),
        "pr_auc": float(average_precision_score(y_true, y_prob)) if has_both else float("nan"),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "threshold": threshold,
    }


def _bootstrap_ci(y_true, y_prob, n_boot=2000, alpha=0.05, seed=42):
    """Bootstrap 95% confidence interval for AUC."""
    rng = np.random.RandomState(seed)
    n = len(y_true)
    aucs = []
    for _ in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        if len(np.unique(y_true[idx])) < 2:
            continue
        aucs.append(roc_auc_score(y_true[idx], y_prob[idx]))
    aucs = sorted(aucs)
    lo = aucs[int(alpha / 2 * len(aucs))]
    hi = aucs[int((1 - alpha / 2) * len(aucs))]
    return lo, hi


# -----------------------------------------------------------------------
# Training and prediction
# -----------------------------------------------------------------------

def _train_and_predict(family, x_train, y_train, x_external, params):
    """Train a model on competition data and predict on external data.

    Trains on the full competition training set (no validation split,
    no early stopping for boosters). Predicts on UCI data using aligned
    columns.
    """
    trainer = ALL_FOLD_TRAINERS[family]
    config = {"params": dict(params), "family": family}

    # TabPFN subsample params
    if family == "tabpfn":
        config["params"].setdefault("sub_size", 3000)
        config["params"].setdefault("n_sub", 8)
        config["params"].setdefault("n_estimators", 4)
        config["params"].setdefault("device", "cpu")

    # Align columns: external data must match training column order
    train_cols = list(x_train.columns) if hasattr(x_train, "columns") else None
    if train_cols is not None and hasattr(x_external, "columns"):
        common_cols = [c for c in train_cols if c in x_external.columns]
        x_train_aligned = x_train[common_cols]
        x_ext_aligned = x_external[common_cols]
    else:
        x_train_aligned = x_train
        x_ext_aligned = x_external
        common_cols = None

    # Scale if needed (fit on training, transform both)
    if family in NEEDS_SCALING:
        scaler = StandardScaler()
        x_train_np = scaler.fit_transform(
            x_train_aligned.values if hasattr(x_train_aligned, "values") else x_train_aligned)
        x_ext_np = scaler.transform(
            x_ext_aligned.values if hasattr(x_ext_aligned, "values") else x_ext_aligned)
    else:
        x_train_np = x_train_aligned
        x_ext_np = x_ext_aligned

    # Train on full competition data (no validation, no early stopping)
    t0 = time.perf_counter()
    _, test_scores, _ = trainer(
        x_train_np, y_train, x_te=x_ext_np,
        config=config, seed=42)
    train_time = time.perf_counter() - t0

    n_features = len(common_cols) if common_cols else x_train_aligned.shape[1]
    return test_scores, train_time, n_features


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="S1/S2: Zero-shot transfer from competition to UCI.")
    parser.add_argument("--datasets", nargs="+", default=EXTERNAL_DATASETS,
                        choices=EXTERNAL_DATASETS)
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--scenario", type=int, choices=[1, 2], default=None,
                        help="1=raw only, 2=raw+stats. Default: both.")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s")

    scenarios = [1, 2] if args.scenario is None else [args.scenario]

    project_root = Path(__file__).resolve().parent.parent.parent
    output_dir = project_root / "results" / "external"
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        s1_count = len(S1_FAMILIES) * len(args.datasets) + len(args.datasets)  # +ensembles
        s2_count = len(S2_FAMILIES) * len(args.datasets) + len(args.datasets)
        total = 0
        if 1 in scenarios:
            total += s1_count
        if 2 in scenarios:
            total += s2_count
        print(f"Scenarios: {scenarios}, total evaluations: ~{total}")
        return

    # Load competition data
    logger.info("Loading competition training data...")
    train_df, test_df = load_train_test()
    y_train = train_df["target"].to_numpy()
    logger.info("Competition: %d samples", len(y_train))

    # Load UCI datasets
    logger.info("Loading UCI datasets...")
    uci_data = {}
    for ds_name in args.datasets:
        X, y, meta = load_uci_dataset(ds_name)
        uci_data[ds_name] = (X, y, meta)
        logger.info("  %s: n=%d, prevalence=%.1f%%",
                     ds_name, meta["n_samples"], meta["prevalence"])

    all_results = []

    # --- Scenario 1: Raw features ---
    if 1 in scenarios:
        logger.info("\n=== SCENARIO 1: Zero-Shot Transfer (Raw Features) ===")
        families = args.models or S1_FAMILIES
        families = [f for f in families if f not in EXCLUDE_FAMILIES]

        # Build raw features in CSV file column order, which ensures
        # consistent tree split tie-breaking across runs.
        x_train_raw = train_df[FILE_ORDER_FEATURES].copy()

        for family in families:
            params = get_competition_params(family)
            if params is None:
                # Fall back to defaults for families without competition params
                # (e.g. pytorch_mlp which uses a different architecture than
                # the competition's sklearn MLPClassifier).
                params = DEFAULT_PARAMS.get(family)
            if params is None:
                logger.info("  [skip] %s: no params available", family)
                continue

            logger.info("  Training %s on competition data (%d features)...",
                        family, x_train_raw.shape[1])

            for ds_name, (X_ext, y_ext, meta) in uci_data.items():
                preds, train_time, n_feat = _train_and_predict(
                    family, x_train_raw, y_train, X_ext, params)

                metrics = _compute_metrics(y_ext, preds)
                ci_lo, ci_hi = _bootstrap_ci(y_ext, preds)

                result = {
                    "scenario": "zero_shot_raw",
                    "model": family,
                    "dataset": ds_name,
                    "feature_set": "raw",
                    "n_features": n_feat,
                    "n_samples": meta["n_samples"],
                    "prevalence": meta["prevalence"],
                    "metrics": metrics,
                    "auc_ci_lower": ci_lo,
                    "auc_ci_upper": ci_hi,
                    "train_time_s": round(train_time, 2),
                }
                all_results.append(result)
                logger.info("    %s -> %s: AUC=%.4f [%.4f, %.4f]",
                            family, ds_name, metrics["auc"], ci_lo, ci_hi)

        # Rank-blend ensemble of tree-based models
        logger.info("\n  Ensemble: rank blend of %s...", ENSEMBLE_FAMILIES)
        ensemble_preds = {}  # ds_name -> {family: preds}

        for family in ENSEMBLE_FAMILIES:
            params = get_competition_params(family)
            if params is None:
                continue
            for ds_name, (X_ext, y_ext, meta) in uci_data.items():
                preds, _, _ = _train_and_predict(
                    family, x_train_raw, y_train, X_ext, params)
                ensemble_preds.setdefault(ds_name, {})[family] = preds

        for ds_name, family_preds in ensemble_preds.items():
            y_ext = uci_data[ds_name][1]
            n = len(y_ext)
            ranked = [rankdata(p) / n for p in family_preds.values()]
            blend = np.mean(ranked, axis=0)
            metrics = _compute_metrics(y_ext, blend)
            ci_lo, ci_hi = _bootstrap_ci(y_ext, blend)

            all_results.append({
                "scenario": "zero_shot_raw",
                "model": "ensemble_rank_blend_raw",
                "dataset": ds_name,
                "feature_set": "raw",
                "n_features": x_train_raw.shape[1],
                "n_samples": uci_data[ds_name][2]["n_samples"],
                "prevalence": uci_data[ds_name][2]["prevalence"],
                "metrics": metrics,
                "auc_ci_lower": ci_lo,
                "auc_ci_upper": ci_hi,
                "components": list(family_preds.keys()),
            })
            logger.info("    ensemble -> %s: AUC=%.4f [%.4f, %.4f]",
                        ds_name, metrics["auc"], ci_lo, ci_hi)

    # --- Scenario 2: Raw + Competition Stats ---
    if 2 in scenarios:
        logger.info("\n=== SCENARIO 2: Zero-Shot Transfer (Raw + Competition Stats) ===")
        families = args.models or S2_FAMILIES
        families = [f for f in families if f not in EXCLUDE_FAMILIES]

        # Build raw + competition-derived target statistics (78 columns).
        # Uses the 630K competition dataset as the reference, not the UCI
        # original (which would be leakage since UCI datasets are the originals).
        if "competition_stats" in FEATURE_BUILDERS:
            x_train_stats, _ = FEATURE_BUILDERS["competition_stats"](train_df, test_df)
            logger.info("  S2 features: %d columns", x_train_stats.shape[1])

            # Build competition stats on each UCI dataset too, so the model
            # sees the same features at prediction time.
            uci_stats = {}
            for ds_name, (X_ext, y_ext, meta) in uci_data.items():
                x_ext_stats, _ = FEATURE_BUILDERS["competition_stats"](X_ext, X_ext)
                uci_stats[ds_name] = x_ext_stats
                logger.info("  %s: %d stat features built", ds_name, x_ext_stats.shape[1])

            for family in families:
                params = get_competition_params(family)
                if params is None:
                    continue

                logger.info("  Training %s on competition data (%d features)...",
                            family, x_train_stats.shape[1])

                for ds_name, (X_ext, y_ext, meta) in uci_data.items():
                    preds, train_time, n_feat = _train_and_predict(
                        family, x_train_stats, y_train, uci_stats[ds_name], params)

                    metrics = _compute_metrics(y_ext, preds)
                    ci_lo, ci_hi = _bootstrap_ci(y_ext, preds)

                    all_results.append({
                        "scenario": "zero_shot_stats",
                        "model": family,
                        "dataset": ds_name,
                        "feature_set": "raw+competition_stats",
                        "n_features": n_feat,
                        "n_samples": meta["n_samples"],
                        "prevalence": meta["prevalence"],
                        "metrics": metrics,
                        "auc_ci_lower": ci_lo,
                        "auc_ci_upper": ci_hi,
                        "train_time_s": round(train_time, 2),
                    })
                    logger.info("    %s -> %s: AUC=%.4f [%.4f, %.4f]",
                                family, ds_name, metrics["auc"], ci_lo, ci_hi)

            # Ensemble for S2
            logger.info("\n  Ensemble: rank blend with stats...")
            for ds_name, (X_ext, y_ext, meta) in uci_data.items():
                family_preds = {}
                for family in ENSEMBLE_FAMILIES:
                    params = get_competition_params(family)
                    if params is None:
                        continue
                    preds, _, _ = _train_and_predict(
                        family, x_train_stats, y_train, uci_stats[ds_name], params)
                    family_preds[family] = preds

                if len(family_preds) >= 2:
                    n = len(y_ext)
                    ranked = [rankdata(p) / n for p in family_preds.values()]
                    blend = np.mean(ranked, axis=0)
                    metrics = _compute_metrics(y_ext, blend)
                    ci_lo, ci_hi = _bootstrap_ci(y_ext, blend)

                    all_results.append({
                        "scenario": "zero_shot_stats",
                        "model": "ensemble_rank_blend_stats",
                        "dataset": ds_name,
                        "feature_set": "raw+competition_stats",
                        "n_features": x_train_stats.shape[1],
                        "n_samples": meta["n_samples"],
                        "prevalence": meta["prevalence"],
                        "metrics": metrics,
                        "auc_ci_lower": ci_lo,
                        "auc_ci_upper": ci_hi,
                        "components": list(family_preds.keys()),
                    })
                    logger.info("    ensemble -> %s: AUC=%.4f [%.4f, %.4f]",
                                ds_name, metrics["auc"], ci_lo, ci_hi)
        else:
            logger.warning("  S2 skipped: 'competition_stats' feature builder not available")

    # Save results
    json_path = output_dir / "zero_shot_transfer.json"
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    logger.info("Saved %d results to %s", len(all_results), json_path)

    csv_rows = []
    for r in all_results:
        row = {k: v for k, v in r.items() if k not in ("metrics", "components")}
        row.update(r["metrics"])
        if "components" in r:
            row["components"] = ",".join(r["components"])
        csv_rows.append(row)
    csv_path = output_dir / "zero_shot_transfer.csv"
    pd.DataFrame(csv_rows).to_csv(csv_path, index=False)
    logger.info("Saved CSV to %s", csv_path)


if __name__ == "__main__":
    main()
