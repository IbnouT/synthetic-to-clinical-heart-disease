"""
Scenario 3: Within-dataset cross-validation on UCI clinical datasets.

Train and evaluate each model family directly on each external UCI dataset
using dataset-adaptive stratified cross-validation. This measures how well
models perform on small real clinical data without any transfer learning
from the competition dataset.

The evaluation grid covers up to 336 combinations:
  14 model families x 4 datasets x 3 tuning modes x 2 imbalance handlers

Tuning modes:
  - default:     use the model family's standard hyperparameters
  - competition: use the best parameters found during the Kaggle competition
  - dataset:     tune per outer fold with Optuna nested 3-fold CV

Imbalance handling:
  - native: model-level class weighting (where supported)
  - smote:  synthetic minority oversampling applied inside each fold

CV strategy adapts to dataset size:
  - Cleveland (n=303) and Hungarian (n=294): 10-fold stratified
  - Switzerland (n=123) and VA Long Beach (n=200): 5-fold x 3 repeats

Each evaluation saves a JSON result file and an NPZ file with OOF
predictions, enabling downstream ensemble analysis and reproducibility
checks.

Usage:
    .venv/bin/python experiments/03_within_dataset_cv.py
    .venv/bin/python experiments/03_within_dataset_cv.py --datasets cleveland --models catboost
    .venv/bin/python experiments/03_within_dataset_cv.py --tuning default --imbalance native
    .venv/bin/python experiments/03_within_dataset_cv.py --resume --dry-run
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
)

from sklearn.model_selection import StratifiedKFold
import optuna
from imblearn.over_sampling import SMOTE

import torch  # noqa: F401 — must be imported before CatBoost on Python 3.14

from src.data_external import DATASET_INFO, load_uci_dataset, get_cv_strategy
from src.models.registry import get_model, ALL_FAMILIES

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------

EXTERNAL_DATASETS = list(DATASET_INFO.keys())

# Optuna trial budget scales with dataset size: more data supports
# a larger search without overfitting to validation noise
OPTUNA_TRIALS_PER_DATASET = {
    "cleveland": 100,
    "hungarian": 100,
    "va_longbeach": 50,
    "switzerland": 30,
}

# Native class-balancing parameters for each model family.
# Applied when imbalance="native" to let models handle skewed classes
# through their built-in mechanisms rather than resampling.
NATIVE_BALANCE_PARAMS = {
    "catboost": {"auto_class_weights": "Balanced"},
    "xgboost": {},  # scale_pos_weight computed dynamically from fold class counts
    "lightgbm": {"is_unbalance": True},
    "random_forest": {"class_weight": "balanced"},
    "extra_trees": {"class_weight": "balanced"},
    "logistic_regression": {"class_weight": "balanced"},
    "ridge": {"class_weight": "balanced"},
    "lasso": {},
    "elastic_net": {},
    "svm": {"class_weight": "balanced"},
    "knn": {},
    "mlp": {},
    "tabpfn": {},
    "realmlp": {},
}

# SMOTE neighbor count for Switzerland (only 8 minority samples) must be
# very small to avoid the "k_neighbors > n_minority" error
SMOTE_K_DEFAULT = 5
SMOTE_K_SWITZERLAND = 2


# -----------------------------------------------------------------------
# JSON serialization helper
# -----------------------------------------------------------------------

class _NumpyEncoder(json.JSONEncoder):
    """Serialize numpy types that the default encoder rejects."""

    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


# -----------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the evaluation grid."""
    parser = argparse.ArgumentParser(
        description="S3 Within-dataset CV on UCI clinical datasets.")
    parser.add_argument(
        "--datasets", nargs="+", default=EXTERNAL_DATASETS,
        choices=EXTERNAL_DATASETS,
        help="Which UCI datasets to evaluate (default: all four).")
    parser.add_argument(
        "--models", nargs="+", default=None,
        help="Model families to evaluate (default: all 14 from registry).")
    parser.add_argument(
        "--tuning", choices=["default", "competition", "dataset", "all"],
        default="all",
        help="Tuning condition: default params, competition-verified, or Optuna.")
    parser.add_argument(
        "--imbalance", choices=["native", "smote", "all"],
        default="all",
        help="Imbalance handling: native class weights or SMOTE oversampling.")
    parser.add_argument(
        "--n-trials", type=int, default=None,
        help="Override Optuna trial count (default: per-dataset schedule).")
    parser.add_argument(
        "--resume", action="store_true",
        help="Skip evaluations where both .json and .npz already exist.")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print evaluation count without running anything.")
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory (default: results/external/).")
    return parser.parse_args()


# -----------------------------------------------------------------------
# Competition-verified parameter loading
# -----------------------------------------------------------------------

def _load_competition_params(model_name: str, project_root: Path) -> dict[str, any] | None:
    """Load competition-verified hyperparameters for a model family.

    These are the best parameters found during the Kaggle competition,
    saved as JSON files in results/verified_params/. Returns None if
    no verified parameters exist for this family.
    """
    param_dir = project_root / "results" / "verified_params"
    if not param_dir.exists():
        return None

    candidates = sorted(param_dir.glob(f"{model_name}*.json"))
    if not candidates:
        return None

    with open(candidates[0]) as f:
        data = json.load(f)

    # Handle both {params: {...}} and flat {...} formats
    if isinstance(data, dict) and "params" in data:
        return data["params"]
    return data


# -----------------------------------------------------------------------
# Class imbalance handling
# -----------------------------------------------------------------------

def _apply_native_balance(
    params: dict[str, any],
    model_name: str,
    y_train: np.ndarray,
) -> dict[str, any]:
    """Add model-specific class-balancing parameters.

    For XGBoost, scale_pos_weight is computed from the actual class
    distribution in the training fold. Other models use their built-in
    balanced class weight option.
    """
    params = dict(params)
    native = NATIVE_BALANCE_PARAMS.get(model_name, {})
    params.update(native)

    if model_name == "xgboost":
        n_neg = int(np.sum(y_train == 0))
        n_pos = int(np.sum(y_train == 1))
        if n_pos > 0:
            params["scale_pos_weight"] = n_neg / n_pos

    return params


def _apply_smote(
    X_train: np.ndarray,
    y_train: np.ndarray,
    dataset_name: str,
    fold_seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply SMOTE oversampling to the training portion of a fold.

    The neighbor count adapts to dataset size: Switzerland's extreme
    imbalance (only ~8 minority samples) requires k=2, while larger
    datasets use k=5. If the minority class is still too small even
    for the configured k, we fall back to k = min_class - 1.
    """
    k_configured = (SMOTE_K_SWITZERLAND if dataset_name == "switzerland"
                     else SMOTE_K_DEFAULT)

    min_class_count = min(np.bincount(y_train.astype(int)))
    k = min(k_configured, min_class_count - 1)

    if k < 1:
        logger.warning(
            "Too few minority samples for SMOTE (n=%d), skipping oversampling",
            min_class_count)
        return X_train, y_train

    sm = SMOTE(random_state=fold_seed, k_neighbors=k)
    return sm.fit_resample(X_train, y_train)


# -----------------------------------------------------------------------
# Per-fold evaluation
# -----------------------------------------------------------------------

def _compute_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    """Compute standard metrics with a guard for single-class folds.

    Unlike src/metrics.compute_metrics(), this handles degenerate folds
    where all samples belong to one class (possible with small UCI
    datasets like Switzerland n=123). Returns NaN for AUC in that case.
    """
    y_pred = (y_prob >= 0.5).astype(int)
    has_both_classes = len(np.unique(y_true)) > 1

    return {
        "auc": float(roc_auc_score(y_true, y_prob)) if has_both_classes else float("nan"),
        "pr_auc": float(average_precision_score(y_true, y_prob)) if has_both_classes else float("nan"),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def _cleanup_memory():
    """Free GPU/MPS memory between folds to prevent accumulation."""
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


# -----------------------------------------------------------------------
# Optuna nested CV tuning
# -----------------------------------------------------------------------

def _tune_optuna(
    model_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    dataset_name: str,
    imbalance: str,
    use_smote: bool,
    base_params: dict[str, any],
    n_trials: int,
    outer_fold_idx: int,
) -> dict[str, any]:
    """Find the best hyperparameters using Optuna with inner 3-fold CV.

    This runs inside each outer CV fold on only the outer training data.
    The inner CV mirrors the outer imbalance condition (native or SMOTE)
    to ensure the tuned parameters are appropriate for the actual
    training pipeline.
    """
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    mod = get_model(model_name)

    # Inner CV uses 3 folds with a seed offset to avoid correlation
    # with the outer fold split
    inner_cv = StratifiedKFold(
        n_splits=3, shuffle=True, random_state=42 + outer_fold_idx)

    # Convert to DataFrame for iloc indexing in inner loop
    X_df = pd.DataFrame(X_train)

    def objective(trial: optuna.Trial) -> float:
        trial_params = {**base_params, **mod.get_optuna_space(trial)}
        inner_aucs = []

        for inner_idx, (itr_idx, ival_idx) in enumerate(inner_cv.split(X_train, y_train)):
            X_itr = X_train[itr_idx]
            y_itr = y_train[itr_idx]
            X_ival = X_train[ival_idx]
            y_ival = y_train[ival_idx]

            # Apply the same imbalance handling as the outer loop
            if use_smote:
                inner_seed = outer_fold_idx * 100 + inner_idx
                X_itr, y_itr = _apply_smote(X_itr, y_itr, dataset_name, inner_seed)

            fold_params = _apply_native_balance(trial_params, model_name, y_itr) \
                if imbalance == "native" else dict(trial_params)

            model = mod.build_model(fold_params)
            mod.train(model, X_itr, y_itr, X_ival, y_ival)

            y_prob = mod.predict(model, X_ival)
            if len(np.unique(y_ival)) > 1:
                inner_aucs.append(roc_auc_score(y_ival, y_prob))

            del model
            _cleanup_memory()

        return float(np.mean(inner_aucs)) if inner_aucs else 0.0

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42 + outer_fold_idx),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best = {**base_params, **study.best_params}
    logger.info(
        "    Optuna (%d trials): best inner AUC=%.4f",
        n_trials, study.best_value)
    return best


# -----------------------------------------------------------------------
# Main CV evaluation loop
# -----------------------------------------------------------------------

def _eval_key(model: str, dataset: str, tuning: str, imbalance: str) -> str:
    """Generate a unique filename stem for one evaluation result."""
    return f"s3_{model}_{dataset}_{tuning}_{imbalance}"


def run_evaluation(
    model_name: str,
    dataset_name: str,
    X: pd.DataFrame,
    y: np.ndarray,
    tuning: str,
    imbalance: str,
    base_params: dict[str, any],
    n_trials: int,
) -> tuple[dict[str, any], np.ndarray]:
    """Run cross-validated evaluation for one grid cell.

    Iterates through the outer CV folds, optionally tuning with Optuna
    inside each fold (for the "dataset" condition). Collects per-fold
    metrics and OOF predictions for the full evaluation summary.

    Returns (result_dict, oof_predictions).
    """
    mod = get_model(model_name)
    cv = get_cv_strategy(dataset_name)
    use_smote = (imbalance == "smote")
    X_arr = X.values
    t0 = time.perf_counter()

    fold_metrics: list[dict[str, float]] = []
    fold_params_list: list[dict[str, any]] = []

    # OOF prediction accumulator: handles repeated CV by averaging
    # predictions across repeats for the same sample
    oof_preds = np.zeros(len(y), dtype=np.float64)
    oof_counts = np.zeros(len(y), dtype=np.int32)

    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X_arr, y)):
        X_train_fold = X_arr[train_idx]
        y_train_fold = y[train_idx]
        X_val_fold = X_arr[val_idx]
        y_val_fold = y[val_idx]

        # Determine hyperparameters for this fold
        if tuning == "dataset":
            fold_params = _tune_optuna(
                model_name, X_train_fold, y_train_fold,
                dataset_name, imbalance, use_smote, base_params,
                n_trials, fold_idx)
        else:
            fold_params = dict(base_params)

        # Apply SMOTE oversampling if requested (on training data only)
        if use_smote:
            X_train_fold, y_train_fold = _apply_smote(
                X_train_fold, y_train_fold, dataset_name, 42 + fold_idx)

        # Apply native class balancing if requested
        if imbalance == "native":
            fold_params = _apply_native_balance(
                fold_params, model_name, y_train_fold)

        fold_params_list.append(dict(fold_params))

        # Build, train, and predict
        model = mod.build_model(fold_params)
        mod.train(model, X_train_fold, y_train_fold, X_val_fold, y_val_fold)

        y_prob = mod.predict(model, X_val_fold)
        fm = _compute_metrics(y_val_fold, y_prob)
        fold_metrics.append(fm)

        # Accumulate OOF predictions
        oof_preds[val_idx] += y_prob
        oof_counts[val_idx] += 1

        del model
        _cleanup_memory()

    elapsed = time.perf_counter() - t0

    # Average OOF predictions for samples seen in multiple repeats
    valid = oof_counts > 0
    oof_preds[valid] /= oof_counts[valid]

    # Overall metrics from the full OOF prediction vector
    overall = _compute_metrics(y[valid], oof_preds[valid])

    # Per-fold AUC statistics
    fold_aucs = [fm["auc"] for fm in fold_metrics if not np.isnan(fm["auc"])]
    mean_auc = float(np.mean(fold_aucs)) if fold_aucs else float("nan")
    std_auc = float(np.std(fold_aucs)) if fold_aucs else float("nan")

    # Record the params from the best-performing fold for reference
    auc_arr = np.array([fm["auc"] for fm in fold_metrics])
    best_fold = int(np.nanargmax(auc_arr)) if not np.all(np.isnan(auc_arr)) else 0

    result = {
        "model": model_name,
        "dataset": dataset_name,
        "feature_set": "raw",
        "tuning": tuning,
        "imbalance": imbalance,
        "scenario": 3,
        "n_folds": len(fold_metrics),
        "oof_auc": overall["auc"],
        "oof_pr_auc": overall["pr_auc"],
        "mean_fold_auc": mean_auc,
        "std_fold_auc": std_auc,
        "metrics": overall,
        "runtime_s": round(elapsed, 2),
        "best_params": fold_params_list[best_fold] if fold_params_list else {},
        "per_fold_params": fold_params_list,
        "fold_metrics": fold_metrics,
    }

    return result, oof_preds


# -----------------------------------------------------------------------
# Result persistence
# -----------------------------------------------------------------------

def _save_eval(
    result: dict[str, any],
    oof_preds: np.ndarray,
    y_true: np.ndarray,
    eval_dir: Path,
    key: str,
) -> None:
    """Save one evaluation result atomically.

    Writes the NPZ file first, then the JSON. The JSON acts as a
    completion sentinel: if only the NPZ exists, the eval is treated
    as incomplete and will be rerun on --resume.
    """
    eval_dir.mkdir(parents=True, exist_ok=True)

    # Save OOF arrays first (NPZ before JSON for atomicity)
    npz_tmp = eval_dir / f"{key}_tmp.npz"
    npz_target = eval_dir / f"{key}.npz"
    np.savez(npz_tmp, oof=oof_preds, y=y_true)
    npz_tmp.rename(npz_target)

    # Save the result JSON
    json_tmp = eval_dir / f"{key}.tmp"
    json_target = eval_dir / f"{key}.json"
    with open(json_tmp, "w") as f:
        json.dump(result, f, indent=2, cls=_NumpyEncoder)
    json_tmp.rename(json_target)


def _find_completed(eval_dir: Path) -> set:
    """Find evaluation keys where both .json and .npz exist."""
    if not eval_dir.exists():
        return set()
    json_keys = {p.stem for p in eval_dir.glob("s3_*.json")}
    npz_keys = {p.stem for p in eval_dir.glob("s3_*.npz")}
    return json_keys & npz_keys


def _load_all_results(eval_dir: Path) -> list[dict[str, any]]:
    """Load all completed S3 evaluation results from the evals directory."""
    results = []
    if eval_dir.exists():
        for p in sorted(eval_dir.glob("s3_*.json")):
            with open(p) as f:
                results.append(json.load(f))
    return results


# -----------------------------------------------------------------------
# Summary output
# -----------------------------------------------------------------------

def _print_summary(results: list[dict[str, any]]) -> None:
    """Print a sorted summary table of all completed evaluations."""
    if not results:
        return

    print(f"\n{'='*100}")
    print("S3 WITHIN-DATASET CV RESULTS")
    print(f"{'='*100}")
    print(f"{'Model':<22} {'Dataset':<14} {'Tuning':<13} {'Imbalance':<10} "
          f"{'OOF-AUC':>8} {'Mean±Std':>14} {'Time':>7}")
    print("-" * 100)

    for r in sorted(results, key=lambda x: (x["dataset"], x["tuning"], -x["oof_auc"])):
        mean_std = f"{r['mean_fold_auc']:.4f}±{r['std_fold_auc']:.4f}"
        print(f"{r['model']:<22} {r['dataset']:<14} {r['tuning']:<13} "
              f"{r['imbalance']:<10} {r['oof_auc']:>8.4f} {mean_std:>14} "
              f"{r['runtime_s']:>6.1f}s")

    print(f"{'='*100}\n")


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def main() -> None:
    """Run the S3 within-dataset evaluation grid."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s")

    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    output_dir = (Path(args.output_dir) if args.output_dir
                  else project_root / "results" / "external")
    eval_dir = output_dir / "evals"

    # Resolve which model families to evaluate
    model_names = args.models or ALL_FAMILIES

    # Expand "all" into the individual conditions
    tuning_conditions = (
        ["default", "competition", "dataset"] if args.tuning == "all"
        else [args.tuning])
    imbalance_conditions = (
        ["native", "smote"] if args.imbalance == "all"
        else [args.imbalance])

    # Check what's already completed for --resume
    completed = _find_completed(eval_dir) if args.resume else set()

    # Build the full evaluation grid and count remaining work
    grid: list[tuple[str, str, str, str]] = []
    for ds in args.datasets:
        for model in model_names:
            for tuning in tuning_conditions:
                for imb in imbalance_conditions:
                    key = _eval_key(model, ds, tuning, imb)
                    if key not in completed:
                        grid.append((model, ds, tuning, imb))

    if args.dry_run:
        total = (len(model_names) * len(args.datasets)
                 * len(tuning_conditions) * len(imbalance_conditions))
        print(f"Total evaluations: {total}")
        print(f"Already completed: {total - len(grid)}")
        print(f"Remaining:         {len(grid)}")
        return

    if not grid:
        logger.info("All evaluations already completed. Nothing to do.")
        all_results = _load_all_results(eval_dir)
        _print_summary(all_results)
        return

    logger.info("Starting %d evaluations (S3 within-dataset CV)", len(grid))

    # Pre-load all requested datasets to avoid repeated disk I/O
    datasets: dict[str, tuple[pd.DataFrame, np.ndarray, dict]] = {}
    for ds_name in args.datasets:
        X, y, meta = load_uci_dataset(ds_name)
        datasets[ds_name] = (X, y, meta)
        logger.info(
            "Loaded %s: n=%d, prevalence=%.1f%%",
            ds_name, meta["n_samples"], meta["prevalence"])

    # Run each evaluation in the grid
    for eval_idx, (model_name, ds_name, tuning, imbalance) in enumerate(grid, 1):
        key = _eval_key(model_name, ds_name, tuning, imbalance)
        X, y, meta = datasets[ds_name]
        mod = get_model(model_name)

        # Resolve base hyperparameters for this tuning condition
        if tuning == "default":
            base_params = mod.get_default_params()
        elif tuning == "competition":
            base_params = _load_competition_params(model_name, project_root)
            if base_params is None:
                logger.info(
                    "[%d/%d] %s: no competition params found, skipping",
                    eval_idx, len(grid), key)
                continue
        elif tuning == "dataset":
            # Start Optuna from competition params if available, else defaults
            base_params = (_load_competition_params(model_name, project_root)
                           or mod.get_default_params())

        n_trials = args.n_trials or OPTUNA_TRIALS_PER_DATASET.get(ds_name, 100)

        logger.info(
            "[%d/%d] %s/%s/%s/%s",
            eval_idx, len(grid), model_name, ds_name, tuning, imbalance)

        try:
            result, oof_preds = run_evaluation(
                model_name, ds_name, X, y,
                tuning, imbalance, base_params, n_trials)

            # Add dataset metadata to the result
            result["n_samples"] = meta["n_samples"]
            result["prevalence"] = meta["prevalence"]

            _save_eval(result, oof_preds, y, eval_dir, key)

            logger.info(
                "    AUC=%.4f (±%.4f), %.1fs",
                result["oof_auc"], result["std_fold_auc"], result["runtime_s"])

        except Exception as e:
            logger.error("    FAILED: %s", e, exc_info=True)

    # Print final summary of all completed evaluations
    all_results = _load_all_results(eval_dir)
    _print_summary(all_results)

    # Save merged results as JSON + CSV
    if all_results:
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / "within_cv_scenario3.json", "w") as f:
            json.dump(all_results, f, indent=2, cls=_NumpyEncoder)

        rows = []
        for r in all_results:
            row = {k: v for k, v in r.items()
                   if k not in ("metrics", "fold_metrics", "per_fold_params", "best_params")}
            row.update(r.get("metrics", {}))
            rows.append(row)
        pd.DataFrame(rows).to_csv(
            output_dir / "within_cv_scenario3.csv", index=False)

        logger.info(
            "Saved %d results to %s/within_cv_scenario3 (.json + .csv)",
            len(all_results), output_dir)

    logger.info("Done.")


if __name__ == "__main__":
    main()
