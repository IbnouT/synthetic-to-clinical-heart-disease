"""Scenario 3 & 4: Within-dataset cross-validation on UCI clinical datasets.

Train and evaluate models directly on each external dataset using
stratified cross-validation. Tests how each model family performs on
small real clinical datasets without any transfer from competition data.

Scenario 3 (S3) uses raw features only (13 clinical variables).
Scenario 4 (S4) uses raw + competition-derived target statistics (78 columns).
  Competition stats come from the 630K S6E2 dataset and are NOT leakage
  since that data is external to the UCI clinical sources. They provide
  each sample with population-level context from a much larger cohort.

The evaluation grid covers up to 336 combinations per scenario:
  14 model families x 4 datasets x 3 tuning modes x 2 imbalance handlers

Tuning modes:
  - default:     practitioner-default hyperparameters
  - competition: hyperparameters from the competition-winning models
  - dataset:     per-fold Optuna tuning with nested 3-fold inner CV

Imbalance handling:
  - native: model-level class weighting (where supported)
  - smote:  SMOTE oversampling applied inside each outer training fold

CV strategy adapts to dataset size:
  - Cleveland (n=303) and Hungarian (n=294): 10-fold stratified
  - Switzerland (n=123) and VA Long Beach (n=200): 5-fold x 3 repeats

Each evaluation saves a JSON result file and an NPZ file with OOF
predictions for downstream analysis and crash recovery.

Usage:
    .venv/bin/python -m experiments.external.within_dataset_cv
    .venv/bin/python -m experiments.external.within_dataset_cv --scenario 4
    .venv/bin/python -m experiments.external.within_dataset_cv --models catboost xgboost
    .venv/bin/python -m experiments.external.within_dataset_cv --tuning default --imbalance native
    .venv/bin/python -m experiments.external.within_dataset_cv --resume
"""
from __future__ import annotations

import argparse
import gc
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

# Nested CV for Optuna inner loop
from sklearn.model_selection import StratifiedKFold

import optuna

import torch  # noqa: F401 — must be imported before CatBoost on Python 3.14

from src.data_external import load_uci_dataset
from src.config import FILE_ORDER_FEATURES
from src.data import load_train_test
from src.features import FEATURE_BUILDERS

# Fold trainers from existing model files
from src.models.boosting import FOLD_TRAINERS as BOOSTING_TRAINERS
from src.models.linear import FOLD_TRAINERS as LINEAR_TRAINERS
from src.models.trees import FOLD_TRAINERS as TREE_TRAINERS
from src.models.neighbors import FOLD_TRAINERS as NEIGHBOR_TRAINERS
from src.models.neural import FOLD_TRAINERS as NEURAL_TRAINERS
from src.models.tabular_dl import FOLD_TRAINERS as TABULAR_DL_TRAINERS

# UCI experiment utilities
from experiments.external.cv_utils import (
    get_cv_splitter,
    apply_smote,
    scale_fold,
    apply_native_balance,
)
from experiments.external.tuning import (
    DEFAULT_PARAMS,
    OPTUNA_SPACES,
    OPTUNA_TRIALS,
    OPTUNA_TRIALS_DEFAULT,
    get_competition_params,
)

logger = logging.getLogger(__name__)

# Combine all fold trainers into a single lookup.
# Lasso and elastic_net are logistic regression with different penalty params,
# so they share the same fold trainer function.
ALL_FOLD_TRAINERS = {}
ALL_FOLD_TRAINERS.update(BOOSTING_TRAINERS)
ALL_FOLD_TRAINERS.update(LINEAR_TRAINERS)
ALL_FOLD_TRAINERS.update(TREE_TRAINERS)
ALL_FOLD_TRAINERS.update(NEIGHBOR_TRAINERS)
ALL_FOLD_TRAINERS.update(NEURAL_TRAINERS)
ALL_FOLD_TRAINERS.update(TABULAR_DL_TRAINERS)
ALL_FOLD_TRAINERS["lasso"] = LINEAR_TRAINERS["logistic_regression"]
ALL_FOLD_TRAINERS["elastic_net"] = LINEAR_TRAINERS["logistic_regression"]

# The 14 families evaluated in UCI experiments.
UCI_FAMILIES = [
    "catboost", "xgboost", "lightgbm",
    "random_forest", "extra_trees",
    "logistic_regression", "ridge", "lasso", "elastic_net",
    "svc", "knn",
    "pytorch_mlp", "tabpfn", "realmlp",
]

EXTERNAL_DATASETS = ["cleveland", "hungarian", "switzerland", "va_longbeach"]

# Models that need per-fold feature scaling before training.
NEEDS_SCALING = {"svc", "knn", "logistic_regression", "ridge", "lasso", "elastic_net"}




# -----------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------

def _compute_metrics(y_true, y_prob):
    """Compute the full metric suite from ground truth and predicted probabilities.

    Includes the optimal classification threshold (Youden's J statistic),
    which maximizes sensitivity + specificity. Threshold-dependent metrics
    (accuracy, precision, recall, F1) use the standard 0.5 cutoff.
    """
    y_pred = (y_prob >= 0.5).astype(int)
    has_both = len(np.unique(y_true)) > 1

    # Optimal threshold via Youden's J = TPR - FPR
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


# -----------------------------------------------------------------------
# Fold trainer config builder
# -----------------------------------------------------------------------

def _build_fold_config(family, params, seed=42):
    """Build the config dict expected by fold trainer functions.

    Fold trainers receive a config dict with at minimum a "params" key.
    Some families need additional keys (family name for tree dispatch,
    cat_features for CatBoost).
    """
    config = {"params": dict(params), "family": family}

    # CatBoost: cat_features is not used for UCI experiments because
    # the data arrives as float numpy arrays after imputation, and
    # CatBoost Pool requires integer-typed categorical columns.
    # The competition used cat_features with pre-typed DataFrames.

    # TabPFN: ensure required subsample params are present
    if family == "tabpfn":
        config["params"].setdefault("sub_size", 3000)
        config["params"].setdefault("n_sub", 8)
        config["params"].setdefault("n_estimators", 4)
        config["params"].setdefault("device", "cpu")

    return config


# -----------------------------------------------------------------------
# Optuna nested CV tuning
# -----------------------------------------------------------------------

def _clean_params(params, family):
    """Fix parameter conflicts that arise from merging Optuna suggestions
    with base params. Optuna's best_params stores raw suggested values,
    not the compatibility-adjusted ones from the space functions.

    CatBoost: remove subsample/bagging_temperature that conflicts with
    the chosen bootstrap_type.
    Logistic regression: ensure solver is compatible with the penalty.
    """
    params = dict(params)

    if family == "catboost" and "bootstrap_type" in params:
        bt = params["bootstrap_type"]
        if bt == "Bayesian":
            params.pop("subsample", None)
        else:
            params.pop("bagging_temperature", None)

    if family in ("logistic_regression", "lasso", "elastic_net"):
        penalty = params.get("penalty", "l2")
        solver = params.get("solver", "lbfgs")
        if penalty == "l1" and solver not in ("liblinear", "saga"):
            params["solver"] = "saga"
        elif penalty == "elasticnet":
            params["solver"] = "saga"
        # Remove l1_ratio when not using elasticnet
        if penalty != "elasticnet":
            params.pop("l1_ratio", None)

    return params


def _tune_optuna(family, x_train, y_train, dataset_name, imbalance,
                 use_smote, base_params, n_trials, outer_fold_idx):
    """Find the best hyperparameters using Optuna with inner 3-fold CV.

    Runs inside each outer CV fold on only the outer training data.
    The inner CV mirrors the outer imbalance condition (native or SMOTE).
    """
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    space_fn = OPTUNA_SPACES[family]
    trainer = ALL_FOLD_TRAINERS[family]

    inner_cv = StratifiedKFold(
        n_splits=3, shuffle=True, random_state=42 + outer_fold_idx)

    def objective(trial):
        trial_params = {**base_params, **space_fn(trial)}

        trial_params = _clean_params(trial_params, family)

        inner_aucs = []

        for inner_idx, (itr_idx, ival_idx) in enumerate(inner_cv.split(x_train, y_train)):
            x_itr = x_train[itr_idx] if isinstance(x_train, np.ndarray) else x_train.iloc[itr_idx]
            x_ival = x_train[ival_idx] if isinstance(x_train, np.ndarray) else x_train.iloc[ival_idx]
            y_itr = y_train[itr_idx]
            y_ival = y_train[ival_idx]

            # Apply SMOTE if the outer condition uses it
            if use_smote:
                inner_seed = outer_fold_idx * 100 + inner_idx
                x_itr, y_itr = apply_smote(x_itr, y_itr, dataset_name, inner_seed)

            # Apply native class balancing
            if imbalance == "native":
                fold_params = apply_native_balance(trial_params, family, y_itr)
            else:
                fold_params = dict(trial_params)

            # Scale features for models that need it
            if family in NEEDS_SCALING:
                x_itr, x_ival = scale_fold(x_itr, x_ival)

            config = _build_fold_config(family, fold_params, seed=42)
            val_scores, _, _ = trainer(x_itr, y_itr, x_ival, y_ival,
                                    config=config, seed=42, fold_idx=inner_idx)

            if len(np.unique(y_ival)) > 1:
                inner_aucs.append(roc_auc_score(y_ival, val_scores))

            gc.collect()

        return float(np.mean(inner_aucs)) if inner_aucs else 0.0

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42 + outer_fold_idx),
    )
    # catch ValueError (numerical overflow), TypeError (pytabkit hidden_sizes
    # "funnel" concatenation bug), and RuntimeError (TabPFN/RealMLP internal)
    # so these trials are pruned instead of crashing the entire sweep.
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False,
                   catch=(ValueError, TypeError, RuntimeError))

    best = {**base_params, **study.best_params}

    # Apply the same param conditioning as in the objective. Optuna's
    # best_params stores raw suggested values, not the adjusted ones
    # from the space function (e.g. solver compatibility for LR).
    best = _clean_params(best, family)

    logger.info("    Optuna (%d trials): best inner AUC=%.4f",
                n_trials, study.best_value)
    return best


# -----------------------------------------------------------------------
# Main CV evaluation loop
# -----------------------------------------------------------------------

def _eval_key(model, dataset, tuning, imbalance, scenario=3):
    """Unique filename stem for one evaluation result.

    S3 uses raw features, S4 uses raw + competition stats. The scenario
    prefix keeps result files from different scenarios cleanly separated.
    """
    prefix = f"s{scenario}"
    return f"{prefix}_{model}_{dataset}_{tuning}_{imbalance}"


def run_evaluation(family, dataset_name, X, y, tuning, imbalance,
                   base_params, n_trials):
    """Run full CV for one family x dataset x tuning x imbalance combo.

    Returns a result dict with OOF AUC, per-fold metrics, and timing,
    plus the OOF prediction array.
    """
    trainer = ALL_FOLD_TRAINERS[family]
    cv = get_cv_splitter(dataset_name)
    use_smote = (imbalance == "smote")
    t0 = time.perf_counter()

    # Convert to numpy for consistent indexing
    X_arr = X.values if hasattr(X, "values") else np.asarray(X)
    y_arr = np.asarray(y)

    fold_metrics = []
    oof_preds = np.zeros(len(y_arr), dtype=np.float64)
    oof_counts = np.zeros(len(y_arr), dtype=np.int32)
    all_fold_params = []

    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X_arr, y_arr)):
        x_train_fold = X_arr[train_idx]
        x_val_fold = X_arr[val_idx]
        y_train_fold = y_arr[train_idx]
        y_val_fold = y_arr[val_idx]

        # Determine params for this fold
        if tuning == "dataset":
            params = _tune_optuna(
                family, x_train_fold, y_train_fold,
                dataset_name, imbalance, use_smote,
                base_params, n_trials, fold_idx)
        else:
            params = base_params

        # Apply SMOTE oversampling if requested (training data only)
        if use_smote:
            x_train_fold, y_train_fold = apply_smote(
                x_train_fold, y_train_fold, dataset_name, 42 + fold_idx)

        # Apply native class balancing
        if imbalance == "native":
            fold_params = apply_native_balance(params, family, y_train_fold)
        else:
            fold_params = dict(params)

        all_fold_params.append(dict(fold_params))

        # Scale features for models that need it
        if family in NEEDS_SCALING:
            x_train_fold, x_val_fold = scale_fold(x_train_fold, x_val_fold)

        # Build config and train. Catch numerical failures (e.g. Ridge/LR
        # solver overflow on small datasets with many competition stat
        # features) so one bad fold doesn't kill the entire evaluation.
        config = _build_fold_config(family, fold_params, seed=42)
        try:
            val_scores, _, _ = trainer(
                x_train_fold, y_train_fold, x_val_fold, y_val_fold,
                config=config, seed=42, fold_idx=fold_idx)
        except (ValueError, RuntimeError) as e:
            logger.warning("    Fold %d failed (%s): %s", fold_idx, family, e)
            continue

        fm = _compute_metrics(y_val_fold, val_scores)
        fold_metrics.append(fm)

        oof_preds[val_idx] += val_scores
        oof_counts[val_idx] += 1

        del val_scores
        gc.collect()

    elapsed = time.perf_counter() - t0

    # Average OOF predictions (repeated CV produces multiple predictions per sample)
    valid_mask = oof_counts > 0
    oof_preds[valid_mask] /= oof_counts[valid_mask]
    overall = _compute_metrics(y_arr[valid_mask], oof_preds[valid_mask])

    fold_aucs = [fm["auc"] for fm in fold_metrics if not np.isnan(fm["auc"])]

    result = {
        "model": family,
        "dataset": dataset_name,
        "tuning": tuning,
        "imbalance": imbalance,
        "n_folds": len(fold_metrics),
        "oof_auc": overall["auc"],
        "oof_pr_auc": overall["pr_auc"],
        "mean_fold_auc": float(np.mean(fold_aucs)) if fold_aucs else float("nan"),
        "std_fold_auc": float(np.std(fold_aucs)) if fold_aucs else float("nan"),
        "metrics": overall,
        "runtime_s": round(elapsed, 2),
        "best_params": dict(all_fold_params[0]) if all_fold_params else {},
        "per_fold_params": all_fold_params,
        "fold_metrics": fold_metrics,
    }
    return result, oof_preds


# -----------------------------------------------------------------------
# I/O helpers
# -----------------------------------------------------------------------

def _save_eval(result, oof_preds, y_true, eval_dir, key):
    """Save one evaluation result atomically (NPZ first, then JSON sentinel)."""
    eval_dir.mkdir(parents=True, exist_ok=True)

    # Save OOF predictions
    npz_target = eval_dir / f"{key}.npz"
    npz_tmp = eval_dir / f"{key}_tmp.npz"
    np.savez(npz_tmp, oof=oof_preds, y=y_true)
    npz_tmp.rename(npz_target)

    # Save result JSON (acts as completion sentinel)
    json_target = eval_dir / f"{key}.json"
    json_tmp = eval_dir / f"{key}.tmp"
    with open(json_tmp, "w") as f:
        json.dump(result, f, indent=2, default=str)
    json_tmp.rename(json_target)


def _find_completed(eval_dir, scenario=3):
    """Find evaluation keys that have both JSON and NPZ files.

    Searches for the given scenario prefix (s3 or s4) to avoid
    confusing results from different feature sets.
    """
    if not eval_dir.exists():
        return set()
    prefix = f"s{scenario}"
    json_keys = {p.stem for p in eval_dir.glob(f"{prefix}_*.json")}
    npz_keys = {p.stem for p in eval_dir.glob(f"{prefix}_*.npz")}
    return json_keys & npz_keys


# -----------------------------------------------------------------------
# CLI and main
# -----------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="S3/S4: Within-dataset CV on UCI clinical datasets.")
    parser.add_argument("--scenario", type=int, choices=[3, 4], default=3,
                        help="3=raw features (S3), 4=raw+competition stats (S4).")
    parser.add_argument("--datasets", nargs="+", default=EXTERNAL_DATASETS,
                        choices=EXTERNAL_DATASETS)
    parser.add_argument("--models", nargs="+", default=None,
                        help="Model families to evaluate (default: all 14).")
    parser.add_argument("--tuning", choices=["default", "competition", "dataset", "all"],
                        default="all")
    parser.add_argument("--imbalance", choices=["native", "smote", "all"],
                        default="all")
    parser.add_argument("--resume", action="store_true",
                        help="Skip evaluations that already have result files.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--n-trials", type=int, default=None,
                        help="Override Optuna trial count.")
    return parser.parse_args()


def _build_s4_features(uci_datasets):
    """Enrich each UCI dataset with competition-derived target statistics.

    Loads the 630K competition dataset once and computes per-value target
    statistics (mean, median, std, skew, count) for each of the 13 clinical
    features. These statistics provide population-level context from a much
    larger cohort without any label leakage.

    Returns a dict mapping dataset name to enriched (X, y, meta) tuples
    with 78 columns (13 raw + 65 competition stats).
    """
    logger.info("Building S4 features (raw + competition stats)...")

    builder = FEATURE_BUILDERS["competition_stats"]
    enriched = {}

    for ds_name, (X, y, meta) in uci_datasets.items():
        # The builder takes (train_df, test_df) and returns enriched frames.
        # For within-dataset CV, we pass the same frame as both train and test
        # to get the full stat columns on all rows. The actual CV splitting
        # happens later in the evaluation loop.
        x_enriched, _ = builder(X, X)
        enriched[ds_name] = (x_enriched, y, meta)
        logger.info("  %s: %d -> %d features", ds_name, X.shape[1], x_enriched.shape[1])

    return enriched


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s")

    scenario = args.scenario
    families = args.models or UCI_FAMILIES
    tuning_modes = ["default", "competition", "dataset"] if args.tuning == "all" else [args.tuning]
    imbalance_modes = ["native", "smote"] if args.imbalance == "all" else [args.imbalance]

    # Results directory
    project_root = Path(__file__).resolve().parent.parent.parent
    eval_dir = project_root / "results" / "external" / "evals"

    # Find already-completed evaluations for this scenario
    completed = _find_completed(eval_dir, scenario=scenario) if args.resume else set()

    # Build evaluation grid
    grid = []
    for ds_name in args.datasets:
        for family in families:
            for tuning in tuning_modes:
                for imbalance in imbalance_modes:
                    grid.append((ds_name, family, tuning, imbalance))

    if args.dry_run:
        skip = sum(1 for ds, f, t, i in grid
                   if _eval_key(f, ds, t, i, scenario=scenario) in completed)
        print(f"Scenario {scenario}, Total: {len(grid)}, already done: {skip}, "
              f"remaining: {len(grid) - skip}")
        return

    feature_label = "raw" if scenario == 3 else "raw+competition_stats"
    logger.info("Starting %d evaluations (S%d within-dataset CV, features=%s)",
                len(grid), scenario, feature_label)

    # Load all UCI datasets up front
    raw_datasets = {}
    for ds_name in args.datasets:
        X, y, meta = load_uci_dataset(ds_name)
        raw_datasets[ds_name] = (X, y, meta)
        logger.info("Loaded %s: n=%d, prevalence=%.1f%%",
                     ds_name, meta["n_samples"], meta["prevalence"])

    # For S4, enrich datasets with competition-derived statistics
    if scenario == 4:
        datasets = _build_s4_features(raw_datasets)
    else:
        datasets = raw_datasets

    for eval_idx, (ds_name, family, tuning, imbalance) in enumerate(grid, 1):
        key = _eval_key(family, ds_name, tuning, imbalance, scenario=scenario)

        if key in completed:
            logger.info("[skip] %s (already done)", key)
            continue

        # Get base params for this tuning mode
        if tuning == "default":
            base_params = DEFAULT_PARAMS.get(family, {})
        elif tuning == "competition":
            base_params = get_competition_params(family)
            if base_params is None:
                logger.info("[skip] %s (no competition params for %s)", key, family)
                continue
        elif tuning == "dataset":
            # Start from competition params if available, else defaults
            base_params = get_competition_params(family) or DEFAULT_PARAMS.get(family, {})

        n_trials = args.n_trials or OPTUNA_TRIALS.get(ds_name, OPTUNA_TRIALS_DEFAULT)

        X, y, meta = datasets[ds_name]

        logger.info("[%d/%d] %s/%s/%s/%s",
                    eval_idx, len(grid), family, ds_name, tuning, imbalance)

        result, oof_preds = run_evaluation(
            family, ds_name, X, y,
            tuning, imbalance, base_params, n_trials)

        result["scenario"] = scenario
        result["feature_set"] = feature_label
        result["n_samples"] = meta["n_samples"]
        result["prevalence"] = meta["prevalence"]

        _save_eval(result, oof_preds, y, eval_dir, key)

        logger.info("    AUC=%.4f (±%.4f), %.1fs",
                    result["oof_auc"], result["std_fold_auc"], result["runtime_s"])

    logger.info("Done.")


if __name__ == "__main__":
    main()
