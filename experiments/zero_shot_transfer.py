"""Zero-shot transfer from competition models to UCI clinical datasets.

Train each model family on the full 630K competition dataset using raw
features, then predict on four UCI Heart Disease subsets without any
retraining. This measures how well competition-tuned models generalize
to real clinical data of different sizes and class distributions.

The 12 model families are evaluated (TabPFN and RealMLP excluded as
they are designed for small-N in-context learning, not 630K training).
Bootstrap confidence intervals quantify prediction uncertainty on each
external dataset.

Usage:
    python -m experiments.zero_shot_transfer
    python -m experiments.zero_shot_transfer --datasets cleveland hungarian
    python -m experiments.zero_shot_transfer --models catboost xgboost
    python -m experiments.zero_shot_transfer --dry-run
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
)

from scipy.stats import rankdata

from src.data_external import load_uci_dataset, DATASET_INFO

from src.data import load_train_test

from src.config import ALL_FEATURES
from src.models.registry import get_model, ALL_FAMILIES

logger = logging.getLogger(__name__)

EXTERNAL_DATASETS = list(DATASET_INFO.keys())

# TabPFN and RealMLP are excluded: they are designed for small-N
# in-context learning and subsample ensembling, not 630K zero-shot
EXCLUDED_FAMILIES = {"tabpfn", "realmlp"}


def parse_args():
    """Parse command-line arguments for zero-shot evaluation."""
    parser = argparse.ArgumentParser(
        description="Zero-shot transfer: competition-trained models on UCI data.")
    parser.add_argument(
        "--datasets", nargs="+", default=EXTERNAL_DATASETS,
        choices=EXTERNAL_DATASETS)
    parser.add_argument(
        "--models", nargs="+", default=None,
        help="Model families to evaluate (default: all except TabPFN/RealMLP).")
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory (default: results/external/).")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print evaluation plan without running.")
    return parser.parse_args()


def _load_verified_params(model_name, project_root):
    """Load competition-verified hyperparameters from results/verified_params/."""
    param_dir = project_root / "results" / "verified_params"
    if not param_dir.exists():
        return None
    candidates = sorted(param_dir.glob(f"{model_name}*.json"))
    if not candidates:
        return None
    with open(candidates[0]) as f:
        data = json.load(f)
    if isinstance(data, dict) and "params" in data:
        return data["params"]
    return data


def _compute_metrics(y_true, y_prob):
    """Compute AUC, PR-AUC, accuracy, precision, recall, and F1."""
    y_pred = (y_prob >= 0.5).astype(int)
    return {
        "auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
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


class _NumpyEncoder(json.JSONEncoder):
    """Serialize numpy types for JSON output."""
    def default(self, obj):
        """Handle numpy int, float, and array serialization."""
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def _align_features(X_external, train_columns):
    """Align external dataset columns to training column order.

    External UCI datasets may have slightly different column sets.
    Only keep columns present in both sets, preserving training order.
    Returns the aligned DataFrame and the list of common columns.
    """
    common = [c for c in train_columns if c in X_external.columns]
    return X_external[common], common


def main():
    """Train on competition data and evaluate zero-shot on UCI datasets."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s")

    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    output_dir = (Path(args.output_dir) if args.output_dir
                  else project_root / "results" / "external")

    # Resolve model list (exclude TabPFN/RealMLP by default)
    if args.models:
        model_names = args.models
    else:
        model_names = [m for m in ALL_FAMILIES if m not in EXCLUDED_FAMILIES]

    if args.dry_run:
        print(f"Models:   {model_names}")
        print(f"Datasets: {args.datasets}")
        print(f"Total:    {len(model_names)} models x {len(args.datasets)} datasets"
              f" = {len(model_names) * len(args.datasets)} evaluations")
        return

    # Load competition training data (630K samples, 13 raw features)
    logger.info("Loading competition training data...")
    train_df, test_df = load_train_test()
    y_train = train_df["target"].to_numpy()

    # Use raw 13 features for zero-shot (same features as UCI datasets)
    X_train = train_df[ALL_FEATURES].copy()
    train_columns = list(X_train.columns)
    logger.info("Competition data: %d samples, %d features", len(y_train), len(train_columns))

    # Load external UCI datasets
    logger.info("Loading external datasets...")
    ext_datasets = {}
    for ds_name in args.datasets:
        X, y, meta = load_uci_dataset(ds_name)
        ext_datasets[ds_name] = (X, y, meta)
        logger.info("  %s: n=%d, prevalence=%.1f%%",
                    ds_name, meta["n_samples"], meta["prevalence"])

    all_results = []
    trained_models = {}

    # Train each model on full competition data, then predict on each UCI dataset
    for model_name in model_names:
        logger.info("\nTraining %s on competition data...", model_name)
        mod = get_model(model_name)

        # Use verified competition params if available, else defaults
        params = _load_verified_params(model_name, project_root)
        params_source = "verified"
        if params is None:
            params = mod.get_default_params()
            params_source = "defaults"

        # Train on full competition data
        t0 = time.perf_counter()
        try:
            model = mod.build_model(params)
            mod.train(model, X_train, y_train)
            train_time = round(time.perf_counter() - t0, 2)
            logger.info("  Trained in %.1fs (%s params)", train_time, params_source)
        except Exception as e:
            logger.error("  %s training FAILED: %s", model_name, e)
            continue

        trained_models[model_name] = (model, mod)

        # Predict on each external dataset
        for ds_name, (X_ext, y_ext, meta) in ext_datasets.items():
            X_aligned, common_cols = _align_features(X_ext, train_columns)

            if len(common_cols) < len(train_columns):
                logger.warning("  %s: using %d/%d features",
                              ds_name, len(common_cols), len(train_columns))

            y_prob = mod.predict(model, X_aligned)
            metrics = _compute_metrics(y_ext, y_prob)
            ci_lo, ci_hi = _bootstrap_ci(y_ext, y_prob)

            result = {
                "scenario": "zero_shot",
                "model": model_name,
                "dataset": ds_name,
                "feature_set": "raw",
                "params_source": params_source,
                "n_features": len(common_cols),
                "n_samples": meta["n_samples"],
                "prevalence": meta["prevalence"],
                "metrics": metrics,
                "auc_ci_lower": ci_lo,
                "auc_ci_upper": ci_hi,
                "train_time_s": train_time,
            }
            all_results.append(result)

            logger.info("  %s -> %s: AUC=%.4f [%.4f, %.4f]",
                        model_name, ds_name, metrics["auc"], ci_lo, ci_hi)

    # Rank-blend ensemble of tree-based models
    tree_families = ["catboost", "xgboost", "lightgbm"]
    available_trees = {n: tm for n, tm in trained_models.items() if n in tree_families}

    if len(available_trees) >= 2:
        logger.info("\nRunning rank-blend ensemble (trees)...")
        for ds_name, (X_ext, y_ext, meta) in ext_datasets.items():
            X_aligned, _ = _align_features(X_ext, train_columns)

            preds = {}
            for name, (model, mod) in available_trees.items():
                preds[name] = mod.predict(model, X_aligned)

            # Equal-weight rank blend
            n = len(y_ext)
            blend = np.mean([rankdata(p) / n for p in preds.values()], axis=0)

            metrics = _compute_metrics(y_ext, blend)
            ci_lo, ci_hi = _bootstrap_ci(y_ext, blend)

            all_results.append({
                "scenario": "zero_shot",
                "model": "ensemble_rank_blend",
                "dataset": ds_name,
                "feature_set": "raw",
                "params_source": "ensemble",
                "n_features": len(train_columns),
                "n_samples": meta["n_samples"],
                "prevalence": meta["prevalence"],
                "metrics": metrics,
                "auc_ci_lower": ci_lo,
                "auc_ci_upper": ci_hi,
                "components": list(preds.keys()),
            })

            logger.info("  ensemble -> %s: AUC=%.4f [%.4f, %.4f]",
                        ds_name, metrics["auc"], ci_lo, ci_hi)

    # Print summary
    print(f"\n{'='*110}")
    print("ZERO-SHOT TRANSFER RESULTS (Competition -> UCI)")
    print(f"{'='*110}")
    print(f"{'Model':<25} {'Dataset':<14} {'AUC':>7} {'CI_Lo':>7} {'CI_Hi':>7} "
          f"{'Acc':>7} {'PR-AUC':>7} {'N':>5}")
    print("-" * 110)
    for r in sorted(all_results, key=lambda x: (x["dataset"], -x["metrics"]["auc"])):
        print(f"{r['model']:<25} {r['dataset']:<14} "
              f"{r['metrics']['auc']:>7.4f} "
              f"{r.get('auc_ci_lower', float('nan')):>7.4f} "
              f"{r.get('auc_ci_upper', float('nan')):>7.4f} "
              f"{r['metrics']['accuracy']:>7.4f} "
              f"{r['metrics']['pr_auc']:>7.4f} "
              f"{r['n_samples']:>5}")
    print(f"{'='*110}\n")

    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "zero_shot_transfer.json", "w") as f:
        json.dump(all_results, f, indent=2, cls=_NumpyEncoder)

    rows = []
    for r in all_results:
        row = {k: v for k, v in r.items()
               if k not in ("metrics", "components")}
        row.update(r["metrics"])
        if "components" in r:
            row["components"] = ",".join(r["components"])
        rows.append(row)
    pd.DataFrame(rows).to_csv(output_dir / "zero_shot_transfer.csv", index=False)

    logger.info("Saved %d results to %s", len(all_results), output_dir)
    logger.info("Done.")


if __name__ == "__main__":
    main()
