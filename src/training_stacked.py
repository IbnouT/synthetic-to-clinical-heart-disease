"""Per-fold stacked training for multi-model ensemble pipelines.

Implements the stacking pattern used by cb_origstats_f10 and top_pipe_f10
models. Within each CV fold, three models are trained and combined:

  1. CatBoost (depth=2, balanced, Bernoulli bootstrap)
  2. XGBoost (depth=2, column subsampling)
  3. RandomForest on rank-space meta-features (residual correction)

The fold-level prediction is a weighted combination:
  final = 0.85 * rank_blend(CB, XGB) + 0.15 * RF_residual

This stacking pattern is based on Mahajan's (2026) CB+XGB residual RF
notebook. The RF component captures nonlinear patterns in the
agreement/disagreement region between the two gradient boosters.

The shallow depth (2) across CB and XGB prevents overfitting on the
630K synthetic dataset while the stacking captures complementary signals.
"""

import gc
import joblib
import numpy as np
from scipy.stats import rankdata
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from catboost import CatBoostClassifier
from xgboost import XGBClassifier

from src.config import SAVED_MODELS_DIR
from src.metrics import compute_metrics
from src.utils import save_metrics, save_oof, save_test_preds


# ---------------------------------------------------------------------------
# Default hyperparameters for the stacking components.
# Shallow trees (depth=2) are deliberate: the 630K synthetic dataset
# rewards regularization over expressiveness.
# ---------------------------------------------------------------------------

DEFAULT_CB_PARAMS = {
    "iterations": 10000,
    "learning_rate": 0.01,
    "depth": 2,
    "loss_function": "Logloss",
    "eval_metric": "AUC",
    "auto_class_weights": "Balanced",
    "bootstrap_type": "Bernoulli",
    "subsample": 0.9,
    "l2_leaf_reg": 12,
    "random_strength": 1.2,
    "task_type": "CPU",
    "verbose": 0,
}

DEFAULT_XGB_PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "learning_rate": 0.01,
    "max_depth": 2,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "n_estimators": 10000,
    "n_jobs": -1,
    "verbosity": 0,
}

DEFAULT_RF_PARAMS = {
    "n_estimators": 1000,
    "max_depth": 8,
    "min_samples_leaf": 80,
    "min_samples_split": 150,
    "max_features": 0.7,
    "bootstrap": True,
    "class_weight": "balanced",
    "n_jobs": -1,
}

# Blending weight for rank average vs RF residual correction.
# 0.85/0.15 was empirically optimal during hyperparameter search.
RANK_WEIGHT = 0.85
RF_WEIGHT = 0.15


def run_stacked_cv(model_id, x_train, x_test, y, n_folds=10, seed=42,
                   cb_params=None, xgb_params=None, rf_params=None):
    """
    Train a per-fold stacked ensemble and produce OOF + test predictions.

    Within each fold the procedure is:
      1. Train CatBoost on the training split, predict validation and test.
      2. Train XGBoost on the same split, predict validation and test.
      3. Rank-normalize both sets of predictions and average them.
      4. Build 4 meta-features from the rank predictions:
         [rank_cb, rank_xgb, rank_avg, |rank_cb - rank_xgb|]
      5. Train a RandomForest on these meta-features using the validation
         labels (captures where the two boosters disagree).
      6. Combine: 0.85 * rank_avg + 0.15 * RF prediction.

    The test prediction is the average of per-fold test predictions,
    each produced by the same stacking procedure.

    Parameters
    ----------
    model_id : str
        Identifier for saving results (e.g. 'cb_origstats_f10_s42').
    x_train : DataFrame or ndarray
        Training features, shape (n_train, n_features).
    x_test : DataFrame or ndarray
        Test features, shape (n_test, n_features).
    y : ndarray
        Binary target labels for training data.
    n_folds : int
        Number of stratified CV folds.
    seed : int
        Random seed for fold splitting.
    cb_params, xgb_params, rf_params : dict, optional
        Override default hyperparameters for each model component.

    Returns
    -------
    dict with keys 'oof', 'test_pred', 'metrics', 'fold_aucs'.
    """
    cb_p = {**DEFAULT_CB_PARAMS, **(cb_params or {})}
    xgb_p = {**DEFAULT_XGB_PARAMS, **(xgb_params or {})}
    rf_p = {**DEFAULT_RF_PARAMS, **(rf_params or {})}

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    oof = np.zeros(len(y))
    test_pred = np.zeros(len(x_test))
    fold_aucs = []

    for fold_idx, (tr_idx, val_idx) in enumerate(skf.split(x_train, y)):
        fold_seed = 42 + fold_idx

        # --- CatBoost ---
        cb = CatBoostClassifier(**cb_p, random_seed=fold_seed)
        cb.fit(x_train.iloc[tr_idx], y[tr_idx])
        val_cb = cb.predict_proba(x_train.iloc[val_idx])[:, 1]
        te_cb = cb.predict_proba(x_test)[:, 1]

        # --- XGBoost ---
        xgb_model = XGBClassifier(**xgb_p, random_state=fold_seed)
        xgb_model.fit(x_train.iloc[tr_idx], y[tr_idx])
        val_xgb = xgb_model.predict_proba(x_train.iloc[val_idx])[:, 1]
        te_xgb = xgb_model.predict_proba(x_test)[:, 1]

        # --- Rank normalization and averaging ---
        # Rank normalization maps heterogeneous probability scales to a
        # uniform [0,1] space, making the average meaningful even when
        # CB and XGB have different calibration characteristics.
        rank_cb_val = rankdata(val_cb) / len(val_cb)
        rank_xgb_val = rankdata(val_xgb) / len(val_xgb)
        rank_cb_te = rankdata(te_cb) / len(te_cb)
        rank_xgb_te = rankdata(te_xgb) / len(te_xgb)

        val_rank_avg = (rank_cb_val + rank_xgb_val) / 2
        te_rank_avg = (rank_cb_te + rank_xgb_te) / 2

        # --- RF residual correction ---
        # The meta-features capture where the two boosters agree (small
        # difference) vs disagree (large difference). The RF learns to
        # exploit these disagreement patterns using the actual labels.
        val_meta = np.column_stack([
            rank_cb_val, rank_xgb_val, val_rank_avg,
            np.abs(rank_cb_val - rank_xgb_val),
        ])
        te_meta = np.column_stack([
            rank_cb_te, rank_xgb_te, te_rank_avg,
            np.abs(rank_cb_te - rank_xgb_te),
        ])

        rf = RandomForestClassifier(**rf_p, random_state=fold_seed)
        rf.fit(val_meta, y[val_idx])
        val_rf = rf.predict_proba(val_meta)[:, 1]
        te_rf = rf.predict_proba(te_meta)[:, 1]

        # Weighted combination: rank average dominates, RF provides
        # a correction signal in ambiguous regions.
        val_final = RANK_WEIGHT * val_rank_avg + RF_WEIGHT * val_rf
        te_final = RANK_WEIGHT * te_rank_avg + RF_WEIGHT * te_rf

        oof[val_idx] = val_final
        test_pred += te_final / n_folds

        fold_auc = roc_auc_score(y[val_idx], val_final)
        fold_aucs.append(fold_auc)
        print(f"  Fold {fold_idx}: AUC={fold_auc:.6f}", flush=True)

        # Save the three component models for this fold.
        fold_dir = SAVED_MODELS_DIR / model_id / f"fold_{fold_idx}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(cb, fold_dir / "catboost.joblib")
        joblib.dump(xgb_model, fold_dir / "xgboost.joblib")
        joblib.dump(rf, fold_dir / "rf.joblib")

        del cb, xgb_model, rf
        gc.collect()

    # Compute and save full metrics suite.
    metrics = compute_metrics(y, oof)
    save_metrics(model_id, metrics, {"label": model_id, "n_folds": n_folds, "seed": seed})
    save_oof(model_id, oof)
    save_test_preds(model_id, test_pred)

    print(f"  {model_id} OOF AUC: {metrics['auc']:.6f}", flush=True)

    return {
        "oof": oof,
        "test_pred": test_pred,
        "metrics": metrics,
        "fold_aucs": fold_aucs,
    }
