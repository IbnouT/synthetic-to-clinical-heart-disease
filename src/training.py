"""Cross-validation training loop.

Trains a model configuration with stratified K-fold CV and saves:
  - OOF predictions (one score per training row, from its held-out fold)
  - Averaged test predictions (mean of per-fold test scores)
  - Metrics summary (AUC, accuracy, precision, recall, F1)

Usage from experiment scripts:
    from src.training import run_cv, run_multi_seed_cv
    result = run_cv("02_cb_raw", MODEL_CONFIGS["02_cb_raw"])
    result = run_multi_seed_cv("xgb39_3seed", config, seeds=[42, 123, 456])
"""

import time
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from src.config import ALL_FEATURES, CAT_FEATURES
from src.config import N_FOLDS, OOF_DIR, PRIMARY_SEED
from src.config import RESULTS_DIR, TEST_PREDS_DIR, SAVED_MODELS_DIR
from src.data import load_folds, load_original_uci, load_train_test
from src.features import FEATURE_BUILDERS
from src.metrics import compute_metrics
from src.utils import save_metrics, save_oof, save_test_preds

import torch  # noqa: F401 — must be imported before CatBoost on Python 3.14

# Fold trainers from each model family
from src.models.boosting import FOLD_TRAINERS as BOOSTING_TRAINERS
from src.models.trees import FOLD_TRAINERS as TREE_TRAINERS
from src.models.linear import FOLD_TRAINERS as LINEAR_TRAINERS
from src.models.neighbors import FOLD_TRAINERS as NEIGHBOR_TRAINERS
from src.models.neural import FOLD_TRAINERS as NEURAL_TRAINERS
from src.models.tabular_dl import FOLD_TRAINERS as TABULAR_DL_TRAINERS

ALL_FOLD_TRAINERS = {}
ALL_FOLD_TRAINERS.update(BOOSTING_TRAINERS)
ALL_FOLD_TRAINERS.update(TREE_TRAINERS)
ALL_FOLD_TRAINERS.update(LINEAR_TRAINERS)
ALL_FOLD_TRAINERS.update(NEIGHBOR_TRAINERS)
ALL_FOLD_TRAINERS.update(NEURAL_TRAINERS)
ALL_FOLD_TRAINERS.update(TABULAR_DL_TRAINERS)


# ---------------------------------------------------------------------------
# Fold-level data transforms
# ---------------------------------------------------------------------------

def _add_target_encoding(train_df, x_tr, x_va, x_te,
                         y_fold_train, train_idx, val_idx, alpha):
    """
    Add target encoding, UCI target means, and frequency columns per fold.

    Adds three groups of columns:
      1. Target encoding (13 cols): alpha-smoothed target means via inner
         5-fold KFold. Validation/test use full training fold means.
      2. UCI target means (13 cols): per-value target prevalence from the
         original 270-row UCI dataset.
      3. Frequency encoding (8 cols): normalized value counts for
         categorical features from the training fold.
    """
    x_tr = x_tr.copy()
    x_va = x_va.copy()
    x_te = x_te.copy()

    global_mean = float(y_fold_train.mean())
    inner_kf = KFold(n_splits=5, shuffle=True, random_state=42)

    train_fold_df = train_df.iloc[train_idx]
    val_fold_df = train_df.iloc[val_idx]

    # Alpha-smoothed target encoding: rare categories get pulled toward the
    # global mean. Formula: (count * mean + alpha * global) / (count + alpha).
    for col in ALL_FEATURES:
        x_tr[f"{col}_te"] = global_mean

        # Inner KFold: compute smoothed mean from inner train, apply to inner val.
        for inner_tr_idx, inner_va_idx in inner_kf.split(train_fold_df):
            inner_train = train_fold_df.iloc[inner_tr_idx]
            inner_target = y_fold_train[inner_tr_idx]
            stats = inner_train.assign(_t=inner_target).groupby(col)["_t"].agg(["mean", "count"])
            smoothed = (stats["count"] * stats["mean"] + alpha * global_mean) / (stats["count"] + alpha)
            mapped = train_fold_df.iloc[inner_va_idx][col].map(smoothed).fillna(global_mean)
            x_tr.iloc[inner_va_idx, x_tr.columns.get_loc(f"{col}_te")] = mapped.values

        # Validation and test use smoothed means from the full training fold.
        full_stats = train_fold_df.assign(_t=y_fold_train).groupby(col)["_t"].agg(["mean", "count"])
        full_smoothed = (full_stats["count"] * full_stats["mean"] + alpha * global_mean) / (full_stats["count"] + alpha)
        x_va[f"{col}_te"] = val_fold_df[col].map(full_smoothed).fillna(global_mean)
        x_te[f"{col}_te"] = x_te[col].map(full_smoothed).fillna(global_mean)

    # UCI target mean per feature value.
    original_df = load_original_uci()
    for col in ALL_FEATURES:
        if col not in original_df.columns:
            continue
        stats_map = original_df.groupby(col)["target"].mean().to_dict()
        x_tr[f"{col}_orig"] = train_fold_df[col].map(stats_map).fillna(0.5)
        x_va[f"{col}_orig"] = val_fold_df[col].map(stats_map).fillna(0.5)
        x_te[f"{col}_orig"] = x_te[col].map(stats_map).fillna(0.5)

    # Frequency encoding for categorical features.
    for col in CAT_FEATURES:
        freq = train_fold_df[col].value_counts(normalize=True)
        x_tr[f"{col}_freq"] = train_fold_df[col].map(freq)
        x_va[f"{col}_freq"] = val_fold_df[col].map(freq).fillna(0)
        x_te[f"{col}_freq"] = x_te[col].map(freq).fillna(0)

    return x_tr, x_va, x_te


def _add_cross_stack_oof(x_tr, x_va, x_te, model_key, train_idx, val_idx):
    """
    Add another model's OOF predictions as an extra numeric feature.

    Loads OOF and test predictions from disk and appends them as a
    column. Works with both DataFrame and numpy array inputs.
    """
    # Load the source model's OOF and test predictions.
    oof_path = OOF_DIR / f"{model_key}_oof.npy"
    test_path = TEST_PREDS_DIR / f"{model_key}_test.npy"

    # Fall back to ensemble_deps if not in the main OOF directory.
    if not oof_path.exists():
        oof_path = RESULTS_DIR / "ensemble_deps" / f"{model_key}_oof.npy"
        test_path = RESULTS_DIR / "ensemble_deps" / f"{model_key}_test.npy"

    source_oof = np.load(oof_path)
    source_test = np.load(test_path)

    pred_col = f"{model_key}_pred"

    # Add the OOF predictions as a column to train and validation sets.
    if isinstance(x_tr, pd.DataFrame):
        x_tr = x_tr.copy()
        x_va = x_va.copy()
        x_te = x_te.copy()
        x_tr[pred_col] = source_oof[train_idx]
        x_va[pred_col] = source_oof[val_idx]
        x_te[pred_col] = source_test
    else:
        x_tr = np.column_stack([x_tr, source_oof[train_idx]])
        x_va = np.column_stack([x_va, source_oof[val_idx]])
        x_te = np.column_stack([x_te, source_test])

    return x_tr, x_va, x_te


def _to_numpy(x):
    """Convert DataFrame to numpy array; pass through if already an array."""
    return x.values if isinstance(x, pd.DataFrame) else np.asarray(x)


def _scale_fold(x_tr, x_va, x_te):
    """Standardize features per fold (fit on train only). Returns numpy arrays."""
    scaler = StandardScaler()
    x_tr = scaler.fit_transform(_to_numpy(x_tr))
    x_va = scaler.transform(_to_numpy(x_va))
    x_te = scaler.transform(_to_numpy(x_te))
    return x_tr, x_va, x_te


def _scale_full_train(x_train, x_test):
    """Standardize on the full training set before fold splitting. Returns numpy arrays."""
    scaler = StandardScaler()
    x_train = scaler.fit_transform(_to_numpy(x_train))
    x_test = scaler.transform(_to_numpy(x_test))
    return x_train, x_test


# ---------------------------------------------------------------------------
# Main CV loop
# ---------------------------------------------------------------------------

def run_cv(experiment_id, config, seed=PRIMARY_SEED, n_folds=N_FOLDS):
    """
    Train one model with stratified K-fold CV.

    Parameters
    ----------
    experiment_id : str
        Canonical name for this run (used in filenames).
    config : dict
        Entry from MODEL_CONFIGS with keys: label, features, family, params.
    seed : int
        Random seed for fold split reproducibility.
    n_folds : int
        Number of CV folds.

    Returns
    -------
    dict with keys: id, metrics, metadata.
    """
    start_time = time.time()

    # Load data and build features for this model's pipeline.
    train_df, test_df = load_train_test()
    y = train_df["target"].to_numpy()

    # Some models use fewer folds (e.g. RealMLP uses 3-fold).
    n_folds = config.get("n_folds", n_folds)

    x_train, x_test = FEATURE_BUILDERS[config["features"]](train_df, test_df)
    folds = load_folds(seed=seed, n_folds=n_folds)

    family = config["family"]
    trainer = ALL_FOLD_TRAINERS[family]

    te_alpha = config.get("te_alpha", None)
    per_fold_scaling = config.get("per_fold_scaling", False)

    # KNN needs consistent scale across folds, so scale before splitting.
    if config.get("full_train_scaling", False):
        x_train, x_test = _scale_full_train(x_train, x_test)

    # OOF: each row gets one prediction from the fold where it was held out.
    # Test preds: averaged across all folds.
    oof = np.zeros(len(train_df))
    test_preds = np.zeros(len(test_df))
    fold_aucs = []

    for fold_i, (train_idx, val_idx) in enumerate(folds):
        # Split features into train/val for this fold.
        if isinstance(x_train, pd.DataFrame):
            x_tr = x_train.iloc[train_idx]
            x_va = x_train.iloc[val_idx]
        else:
            x_tr = x_train[train_idx]
            x_va = x_train[val_idx]

        x_te = x_test

        # Add per-fold target encoding + UCI stats + frequency columns.
        if te_alpha is not None:
            x_tr, x_va, x_te = _add_target_encoding(
                train_df, x_tr, x_va, x_te.copy(),
                y[train_idx], train_idx, val_idx, te_alpha,
            )

        # Cross-stacking: inject another model's OOF predictions as a feature.
        cross_stack_key = config.get("cross_stack_oof", None)
        if cross_stack_key is not None:
            x_tr, x_va, x_te = _add_cross_stack_oof(
                x_tr, x_va, x_te, cross_stack_key, train_idx, val_idx,
            )

        # Standardize features within this fold (SVM).
        if per_fold_scaling:
            x_tr, x_va, x_te = _scale_fold(x_tr, x_va, x_te)

        # Per-fold seed adds fold index to the base seed for fold-level diversity.
        fold_seed = seed + fold_i if config.get("per_fold_seed", False) else seed

        # Train model and get predictions for this fold.
        val_scores, test_scores, fold_model = trainer(
            x_tr, y[train_idx], x_va, y[val_idx], x_te, config, fold_seed,
            fold_idx=fold_i,
        )

        # Save trained model for this fold.
        model_dir = SAVED_MODELS_DIR / experiment_id
        model_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(fold_model, model_dir / f"fold_{fold_i}.joblib")

        oof[val_idx] = val_scores
        test_preds += test_scores / n_folds

        fold_auc = roc_auc_score(y[val_idx], val_scores)
        fold_aucs.append(float(fold_auc))
        print(f"  Fold {fold_i + 1}/{n_folds}  AUC: {fold_auc:.6f}")

    # Overall metrics computed on the full OOF vector.
    metrics = compute_metrics(y, oof)
    elapsed = round(time.time() - start_time, 2)

    metadata = {
        "label": config["label"],
        "features": config["features"],
        "family": family,
        "seed": seed,
        "n_folds": n_folds,
        "n_features": int(x_train.shape[1]),
        "fold_aucs": fold_aucs,
        "runtime_seconds": elapsed,
    }

    save_oof(experiment_id, oof)
    save_test_preds(experiment_id, test_preds)
    save_metrics(experiment_id, metrics, metadata)

    print(f"  OOF AUC: {metrics['auc']:.6f}  ({elapsed}s)")
    return {"id": experiment_id, "metrics": metrics, "metadata": metadata}


def run_multi_seed_cv(experiment_id, config, seeds, n_folds=N_FOLDS):
    """
    Train with multiple random seeds and save per-seed + averaged predictions.

    Some multi-seed experiments use different fold splits per seed (the seed
    controls both model randomness and fold assignment). Others keep folds
    fixed at PRIMARY_SEED and only vary the model's random state. The config
    key ``per_seed_folds`` controls which mode is used.

    For cross-stacking models (config key ``cross_stack_oof``), the OOF
    predictions from another model are injected as an extra feature column
    per fold (each row's value comes from a fold where it was held out).

    Parameters
    ----------
    experiment_id : str
        Canonical name for this run (used in filenames).
    config : dict
        Entry from MODEL_CONFIGS with keys: label, features, family, params.
    seeds : list of int
        Random seeds to train with.
    n_folds : int
        Number of CV folds.
    """
    start_time = time.time()

    # Load data and build features.
    train_df, test_df = load_train_test()
    y = train_df["target"].to_numpy()

    x_train, x_test = FEATURE_BUILDERS[config["features"]](train_df, test_df)

    # When per_seed_folds is True, each seed uses its own fold split file.
    # Otherwise all seeds share the same folds (only model seed varies).
    per_seed_folds = config.get("per_seed_folds", False)

    # Cross-stacking: load the donor model's OOF predictions to inject
    # as an extra feature per fold (held-out OOF values are safe to use).
    cross_stack_key = config.get("cross_stack_oof", None)
    cross_stack_oof = None
    cross_stack_test = None
    if cross_stack_key is not None:
        cross_stack_oof = np.load(OOF_DIR / f"{cross_stack_key}_oof.npy")
        cross_stack_test = np.load(TEST_PREDS_DIR / f"{cross_stack_key}_test.npy")

    family = config["family"]
    trainer = ALL_FOLD_TRAINERS[family]

    # Seed-averaged accumulators.
    blend_oof = np.zeros(len(train_df))
    blend_test = np.zeros(len(test_df))
    seed_fold_aucs = {}
    all_seed_oofs = []
    all_seed_tests = []

    for seed in seeds:
        # Choose folds: per-seed or fixed at PRIMARY_SEED.
        fold_seed = seed if per_seed_folds else PRIMARY_SEED
        folds = load_folds(seed=fold_seed, n_folds=n_folds)

        # Per-seed accumulators (averaged into blend after all folds).
        seed_oof = np.zeros(len(train_df))
        seed_test = np.zeros(len(test_df))
        fold_aucs = []

        for fold_i, (train_idx, val_idx) in enumerate(folds):
            # Split features for this fold.
            if isinstance(x_train, pd.DataFrame):
                x_tr = x_train.iloc[train_idx].copy()
                x_va = x_train.iloc[val_idx].copy()
                x_te = x_test.copy()
            else:
                x_tr = x_train[train_idx].copy()
                x_va = x_train[val_idx].copy()
                x_te = x_test.copy()

            # Inject cross-stacked OOF as extra feature column.
            if cross_stack_oof is not None:
                _inject_cross_stack(x_tr, x_va, x_te, cross_stack_oof,
                                    cross_stack_test, train_idx, val_idx)

            # Train and predict for this seed + fold combination.
            val_scores, test_scores, fold_model = trainer(
                x_tr, y[train_idx], x_va, y[val_idx], x_te, config, seed,
                fold_idx=fold_i,
            )

            # Save trained model for this seed + fold.
            model_dir = SAVED_MODELS_DIR / f"{experiment_id}_s{seed}"
            model_dir.mkdir(parents=True, exist_ok=True)
            joblib.dump(fold_model, model_dir / f"fold_{fold_i}.joblib")

            # Accumulate OOF and test predictions for this seed.
            seed_oof[val_idx] = val_scores
            seed_test += test_scores / n_folds

            fold_auc = roc_auc_score(y[val_idx], val_scores)
            fold_aucs.append(float(fold_auc))
            print(f"  Seed {seed}  Fold {fold_i + 1}/{n_folds}  AUC: {fold_auc:.6f}")

        # Log this seed's overall AUC.
        seed_auc = roc_auc_score(y, seed_oof)
        seed_fold_aucs[str(seed)] = fold_aucs
        print(f"  Seed {seed} OOF AUC: {seed_auc:.6f}")

        # Save per-seed OOF and test predictions for the ensemble blend.
        save_oof(f"{experiment_id}_s{seed}", seed_oof)
        save_test_preds(f"{experiment_id}_s{seed}", seed_test)
        all_seed_oofs.append(seed_oof)
        all_seed_tests.append(seed_test)

        # Average this seed's predictions into the blend.
        blend_oof += seed_oof / len(seeds)
        blend_test += seed_test / len(seeds)

    # Metrics on the final seed-averaged OOF.
    metrics = compute_metrics(y, blend_oof)
    elapsed = round(time.time() - start_time, 2)

    # Bundle run metadata.
    metadata = {
        "label": config["label"],
        "features": config["features"],
        "family": family,
        "seeds": seeds,
        "n_folds": n_folds,
        "n_features": int(x_train.shape[1]),
        "per_seed_folds": per_seed_folds,
        "seed_fold_aucs": seed_fold_aucs,
        "runtime_seconds": elapsed,
    }

    # Save blended predictions and metrics.
    save_oof(experiment_id, blend_oof)
    save_test_preds(experiment_id, blend_test)
    save_metrics(experiment_id, metrics, metadata)

    print(f"  Blended OOF AUC: {metrics['auc']:.6f}  ({elapsed}s)")
    return {"id": experiment_id, "metrics": metrics, "metadata": metadata}


def _inject_cross_stack(x_tr, x_va, x_te, donor_oof, donor_test,
                        train_idx, val_idx):
    """Add a donor model's OOF predictions as an extra feature column."""
    col_name = "stack_pred"
    if isinstance(x_tr, pd.DataFrame):
        x_tr[col_name] = donor_oof[train_idx]
        x_va[col_name] = donor_oof[val_idx]
        x_te[col_name] = donor_test
    else:
        x_tr = np.column_stack([x_tr, donor_oof[train_idx]])
        x_va = np.column_stack([x_va, donor_oof[val_idx]])
        x_te = np.column_stack([x_te, donor_test])
