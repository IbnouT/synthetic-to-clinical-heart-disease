"""Scenario 5a & 5b: Ensemble transfer to UCI clinical datasets.

S5a (competition ensemble replica):
    Trains the same 7 base models used in the competition's hillclimb_v4
    ensemble on each UCI dataset, using competition-adapted feature
    pipelines. Tests whether the ensemble *design* transfers to real
    clinical data with 3 ensemble methods:
      - Equal-weight rank blend
      - Greedy hillclimb (forward selection maximizing AUC)
      - PFE meta-classifier (RF on rank-normalized OOFs)

S5b (diverse model ensemble):
    Selects the best configuration per model family per dataset from
    S3/S4 results, retrains to collect OOFs, then tests 6 ensemble
    methods. Depends on S3+S4 being complete.

Usage:
    .venv/bin/python -m experiments.external.ensemble_transfer --scenario 5a
    .venv/bin/python -m experiments.external.ensemble_transfer --scenario 5b
    .venv/bin/python -m experiments.external.ensemble_transfer --scenario 5a --datasets cleveland
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

from sklearn.metrics import average_precision_score, roc_auc_score

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from scipy.stats import rankdata

# PyTorch must be imported before CatBoost on Python 3.14
import torch  # noqa: F401

from src.data_external import load_uci_dataset
from src.data import load_train_test
from src.config import MODEL_CONFIGS, ALL_FEATURES
from src.features import FEATURE_BUILDERS
from src.features.helpers import frequency_encode_column, target_encode_oof

# Fold trainers
from src.models.boosting import FOLD_TRAINERS as BOOSTING_TRAINERS
from src.models.linear import FOLD_TRAINERS as LINEAR_TRAINERS

# Scaling for logistic regression
from experiments.external.cv_utils import get_cv_splitter, scale_fold

logger = logging.getLogger(__name__)

EXTERNAL_DATASETS = ["cleveland", "hungarian", "switzerland", "va_longbeach"]

# -----------------------------------------------------------------------
# S5a: Competition ensemble base model definitions
# -----------------------------------------------------------------------
# These 7 models mirror the competition's hillclimb_v4 components,
# adapted for UCI datasets. Feature pipelines use competition stats
# (from 630K) instead of UCI original stats to avoid leakage.

S5A_BASE_MODELS = [
    {
        "name": "cb_baseline",
        "family": "catboost",
        "config_key": "02_cb_raw",
        "feature_pipeline": "cb_baseline_clean",
        "param_overrides": {},
    },
    {
        "name": "cb_origstats_best",
        "family": "catboost",
        "config_key": "07_cb_origstats",
        "feature_pipeline": "competition_stats_full",
        "param_overrides": {},
    },
    {
        "name": "cb_origstats_norf",
        "family": "catboost",
        "config_key": "07_cb_origstats",
        "feature_pipeline": "competition_stats_full",
        "param_overrides": {"random_strength": 0},
    },
    {
        "name": "cb_shrink_001",
        "family": "catboost",
        "config_key": "02_cb_raw",
        "feature_pipeline": "cb_baseline_clean",
        "param_overrides": {"l2_leaf_reg": 0.01},
    },
    {
        "name": "lr_onehot",
        "family": "logistic_regression",
        "config_key": "06_lr_onehot",
        "feature_pipeline": "onehot",
        "param_overrides": {},
    },
    {
        "name": "te_cb_a10",
        "family": "catboost",
        "config_key": "te_cb_a10",
        "feature_pipeline": "te_alpha10",
        "param_overrides": {},
    },
    {
        "name": "top_pipe_f10",
        "family": "catboost",
        "config_key": "02_cb_raw",
        "feature_pipeline": "top_pipe",
        "param_overrides": {},
    },
]

# Native class balancing applied during training
NATIVE_BALANCE = {
    "catboost": {"auto_class_weights": "Balanced"},
    "logistic_regression": {"class_weight": "balanced"},
}

# Models that need per-fold feature scaling
NEEDS_SCALING = {"logistic_regression"}

# Fold trainer lookup
FOLD_TRAINERS = {}
FOLD_TRAINERS.update(BOOSTING_TRAINERS)
FOLD_TRAINERS.update(LINEAR_TRAINERS)


# -----------------------------------------------------------------------
# Feature pipeline builders for UCI data
# -----------------------------------------------------------------------

def _build_features_for_uci(pipeline_name, X_train, X_val, y_train,
                              train_idx, val_idx):
    """Build features for one CV fold on UCI data.

    Uses v12's existing feature builders from src/features/. For the
    competition_stats_full pipeline (used by origstats models in S5a),
    combines the competition_stats builder with frequency and target
    encoding computed within the fold to avoid leakage.
    """
    def _to_array(x):
        return x.values if hasattr(x, "values") else np.asarray(x)

    # Drop target column before passing to builders that don't expect it
    feature_cols = [c for c in X_train.columns if c != "target"]

    # Pipelines that map directly to a v12 feature builder
    direct_builders = {
        "raw": "raw",
        "cb_baseline": "cb_baseline",
        "competition_stats": "competition_stats",
        "onehot": "onehot",
        "te_alpha10": "te_alpha10",
        "top_pipe": "top_pipe",
    }

    if pipeline_name == "cb_baseline_clean":
        # Leakage-free version of cb_baseline: 13 raw (as str for CatBoost
        # native categoricals) + 13 frequency + 13 target-mean columns = 39.
        # Same structure as original cb_baseline, but _orig columns use
        # target means from the 630K competition data instead of the 270-row
        # UCI Cleveland dataset (which overlaps 89% with the eval set).
        x_tr = X_train[ALL_FEATURES].copy().astype(str)
        x_va = X_val[ALL_FEATURES].copy().astype(str)

        # Load competition data for target means (no leakage)
        comp_train, _ = load_train_test()
        comp_global_mean = float(comp_train["target"].mean())

        for col in ALL_FEATURES:
            freq_tr, freq_va = frequency_encode_column(
                X_train[col], X_val[col], source="combined")
            x_tr[f"{col}_freq"] = freq_tr
            x_va[f"{col}_freq"] = freq_va

            # Target means from 630K competition data instead of UCI
            comp_mean_map = comp_train.groupby(col)["target"].mean().to_dict()
            x_tr[f"{col}_orig"] = X_train[col].map(comp_mean_map).fillna(comp_global_mean)
            x_va[f"{col}_orig"] = X_val[col].map(comp_mean_map).fillna(comp_global_mean)

        return _to_array(x_tr), _to_array(x_va)

    if pipeline_name in direct_builders:
        builder = FEATURE_BUILDERS[direct_builders[pipeline_name]]
        x_tr, x_va = builder(X_train, X_val)
        return _to_array(x_tr), _to_array(x_va)

    if pipeline_name == "competition_stats_full":
        # Competition equivalent of origstats_full: competition stats (78)
        # + frequency encoding (13) + OOF target encoding (13) = ~104 cols.
        # Uses 630K competition data for statistics (not UCI original)
        # to avoid leakage when evaluating on UCI datasets.
        builder = FEATURE_BUILDERS["competition_stats"]
        x_tr, _ = builder(X_train[feature_cols], X_train[feature_cols])
        x_va, _ = builder(X_val[feature_cols], X_val[feature_cols])

        # Add frequency encoding from training fold
        for col in ALL_FEATURES:
            freq_tr, freq_va = frequency_encode_column(
                X_train[col], X_val[col], source="train")
            x_tr[f"{col}_freq"] = freq_tr.values
            x_va[f"{col}_freq"] = freq_va.values

        # Add OOF target encoding within fold
        train_te, val_te = target_encode_oof(
            X_train, X_val, ALL_FEATURES, y_train)
        for col in ALL_FEATURES:
            te_col = f"{col}_te"
            x_tr[te_col] = train_te[te_col].values
            x_va[te_col] = val_te[te_col].values

        return _to_array(x_tr), _to_array(x_va)

    raise ValueError(f"Unknown feature pipeline: {pipeline_name}")


# -----------------------------------------------------------------------
# OOF collection
# -----------------------------------------------------------------------

def collect_oofs_s5a(dataset_name, X, y):
    """Train all S5a base models and collect OOF predictions.

    Each model uses its own feature pipeline and hyperparameters from
    the competition config. Returns a dict mapping model name to OOF
    prediction arrays.
    """
    cv = get_cv_splitter(dataset_name)
    oofs = {}

    for model_spec in S5A_BASE_MODELS:
        name = model_spec["name"]
        family = model_spec["family"]
        pipeline = model_spec["feature_pipeline"]
        config_key = model_spec["config_key"]

        logger.info("  Training %s (%s, %s)...", name, family, pipeline)

        # Get competition-verified params and apply overrides
        base_params = dict(MODEL_CONFIGS.get(config_key, {}).get("params", {}))
        base_params.update(model_spec["param_overrides"])
        base_params.update(NATIVE_BALANCE.get(family, {}))

        trainer = FOLD_TRAINERS.get(family)
        if trainer is None:
            logger.warning("  No fold trainer for %s, skipping %s", family, name)
            continue

        oof_preds = np.zeros(len(y), dtype=np.float64)
        oof_counts = np.zeros(len(y), dtype=np.int32)
        t0 = time.perf_counter()

        # Need target column in dataframes for feature builders
        X_with_target = X.copy()
        X_with_target["target"] = y

        for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X.values, y)):
            X_tr_df = X_with_target.iloc[train_idx]
            X_va_df = X_with_target.iloc[val_idx]
            y_tr = y[train_idx]
            y_va = y[val_idx]

            # Build features for this fold
            x_tr_feat, x_va_feat = _build_features_for_uci(
                pipeline, X_tr_df, X_va_df, y_tr, train_idx, val_idx)

            # Scale if needed
            if family in NEEDS_SCALING:
                x_tr_feat, x_va_feat = scale_fold(x_tr_feat, x_va_feat)

            config = {"params": dict(base_params), "family": family}
            val_scores, _, _ = trainer(
                x_tr_feat, y_tr, x_va_feat, y_va,
                config=config, seed=42, fold_idx=fold_idx)

            oof_preds[val_idx] += val_scores
            oof_counts[val_idx] += 1

            del val_scores
            gc.collect()

        valid = oof_counts > 0
        oof_preds[valid] /= oof_counts[valid]

        auc = roc_auc_score(y[valid], oof_preds[valid]) if len(np.unique(y[valid])) > 1 else float("nan")
        elapsed = time.perf_counter() - t0
        logger.info("    %s AUC=%.4f (%.1fs)", name, auc, elapsed)

        oofs[name] = oof_preds

    return oofs


# -----------------------------------------------------------------------
# Ensemble methods
# -----------------------------------------------------------------------

def rank_normalize(arr):
    """Rank-normalize an array to [0, 1] range."""
    return rankdata(arr) / len(arr)


def ensemble_rank_blend(oofs, y):
    """Equal-weight rank blend of all OOF predictions."""
    names = sorted(oofs.keys())
    blend = np.mean([rank_normalize(oofs[n]) for n in names], axis=0)
    auc = roc_auc_score(y, blend) if len(np.unique(y)) > 1 else float("nan")
    pr_auc = average_precision_score(y, blend) if len(np.unique(y)) > 1 else float("nan")
    return blend, auc, pr_auc


def ensemble_hillclimb(oofs, y, max_rounds=50):
    """Greedy forward selection: add the model that most improves AUC.

    Starts from the best single model and iteratively adds components,
    weighting by selection count. Stops when no improvement is found.
    """
    names = sorted(oofs.keys())
    ranked = {n: rank_normalize(oofs[n]) for n in names}

    # Start with best single
    best_name = max(names, key=lambda n: roc_auc_score(y, oofs[n]))
    selected = [best_name]
    best_blend = ranked[best_name].copy()
    best_auc = roc_auc_score(y, best_blend)

    for _ in range(max_rounds):
        improved = False
        for name in names:
            trial_blend = (best_blend * len(selected) + ranked[name]) / (len(selected) + 1)
            trial_auc = roc_auc_score(y, trial_blend)
            if trial_auc > best_auc + 1e-6:
                best_auc = trial_auc
                best_blend = trial_blend
                selected.append(name)
                improved = True
                break
        if not improved:
            break

    pr_auc = average_precision_score(y, best_blend)

    # Compute weights from selection counts
    weights = {}
    for n in names:
        weights[n] = selected.count(n) / len(selected)

    return best_blend, best_auc, pr_auc, weights


def ensemble_pfe(oofs, y, dataset_name):
    """PFE-style meta-classifier: RF trained on rank-normalized OOFs.

    Uses nested CV to avoid overfitting the meta-classifier. Final
    prediction blends 85% rank blend with 15% RF meta-predictions,
    matching the competition's PFE ratio.
    """
    names = sorted(oofs.keys())
    cv = get_cv_splitter(dataset_name)
    ranked = {n: rank_normalize(oofs[n]) for n in names}
    X_meta = np.column_stack([ranked[n] for n in names])

    meta_preds = np.zeros(len(y), dtype=np.float64)
    meta_counts = np.zeros(len(y), dtype=np.int32)

    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(np.zeros(len(y)), y)):
        rf = RandomForestClassifier(
            n_estimators=200, max_depth=3,
            random_state=42 + fold_idx, n_jobs=-1)
        rf.fit(X_meta[train_idx], y[train_idx])
        meta_preds[val_idx] += rf.predict_proba(X_meta[val_idx])[:, 1]
        meta_counts[val_idx] += 1

    valid = meta_counts > 0
    meta_preds[valid] /= meta_counts[valid]

    # Blend: 85% equal rank blend + 15% RF meta
    equal_blend, _, _ = ensemble_rank_blend(oofs, y)
    final = 0.85 * rank_normalize(equal_blend) + 0.15 * rank_normalize(meta_preds)

    auc = roc_auc_score(y[valid], final[valid]) if len(np.unique(y[valid])) > 1 else float("nan")
    pr_auc = average_precision_score(y[valid], final[valid]) if len(np.unique(y[valid])) > 1 else float("nan")
    return final, auc, pr_auc


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="S5a/S5b: Ensemble transfer to UCI datasets.")
    parser.add_argument("--scenario", required=True, choices=["5a", "5b"])
    parser.add_argument("--datasets", nargs="+", default=EXTERNAL_DATASETS,
                        choices=EXTERNAL_DATASETS)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def run_s5a(datasets):
    """Run S5a: competition ensemble replica on UCI datasets."""
    project_root = Path(__file__).resolve().parent.parent.parent
    output_dir = project_root / "results" / "external"
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = []

    for ds_name in datasets:
        logger.info("\n%s", "=" * 60)
        logger.info("S5a: %s", ds_name)
        logger.info("=" * 60)

        X, y, meta = load_uci_dataset(ds_name)
        logger.info("  n=%d, prevalence=%.1f%%", meta["n_samples"], meta["prevalence"])

        # Collect OOFs from all 7 base models
        oofs = collect_oofs_s5a(ds_name, X, y)

        if len(oofs) < 2:
            logger.warning("  Only %d models produced OOFs, skipping ensembles", len(oofs))
            continue

        # Individual model results
        for name, oof in oofs.items():
            auc = roc_auc_score(y, oof) if len(np.unique(y)) > 1 else float("nan")
            pr_auc = average_precision_score(y, oof) if len(np.unique(y)) > 1 else float("nan")
            all_results.append({
                "dataset": ds_name, "experiment": "competition_replica",
                "method": f"single_{name}", "ensemble_type": "single",
                "auc": round(float(auc), 6), "pr_auc": round(float(pr_auc), 4),
                "n_models": 1,
            })
            logger.info("    single_%s: AUC=%.4f", name, auc)

        # Equal rank blend
        _, rb_auc, rb_pr = ensemble_rank_blend(oofs, y)
        all_results.append({
            "dataset": ds_name, "experiment": "competition_replica",
            "method": "rank_blend_equal", "ensemble_type": "rank_blend",
            "auc": round(float(rb_auc), 6), "pr_auc": round(float(rb_pr), 4),
            "n_models": len(oofs),
        })
        logger.info("    rank_blend: AUC=%.4f", rb_auc)

        # Hillclimb
        _, hc_auc, hc_pr, hc_weights = ensemble_hillclimb(oofs, y)
        all_results.append({
            "dataset": ds_name, "experiment": "competition_replica",
            "method": "hillclimb_blend", "ensemble_type": "hillclimb",
            "auc": round(float(hc_auc), 6), "pr_auc": round(float(hc_pr), 4),
            "n_models": len(oofs),
        })
        logger.info("    hillclimb: AUC=%.4f", hc_auc)

        # PFE meta-classifier
        _, pfe_auc, pfe_pr = ensemble_pfe(oofs, y, ds_name)
        all_results.append({
            "dataset": ds_name, "experiment": "competition_replica",
            "method": "pfe_meta_classifier", "ensemble_type": "pfe",
            "auc": round(float(pfe_auc), 6), "pr_auc": round(float(pfe_pr), 4),
            "n_models": len(oofs),
        })
        logger.info("    pfe: AUC=%.4f", pfe_auc)

    # Save results
    json_path = output_dir / "ensemble_competition_replica.json"
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    logger.info("Saved %d results to %s", len(all_results), json_path)

    csv_rows = [{k: v for k, v in r.items()} for r in all_results]
    csv_path = output_dir / "ensemble_competition_replica.csv"
    pd.DataFrame(csv_rows).to_csv(csv_path, index=False)
    logger.info("Saved CSV to %s", csv_path)


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s")

    if args.scenario == "5a":
        if args.dry_run:
            print(f"S5a: {len(S5A_BASE_MODELS)} base models × {len(args.datasets)} datasets × 3 ensembles")
            return
        run_s5a(args.datasets)

    elif args.scenario == "5b":
        logger.info("S5b not yet implemented, needs S3+S4 results first")
        return

    logger.info("Done.")


if __name__ == "__main__":
    main()
