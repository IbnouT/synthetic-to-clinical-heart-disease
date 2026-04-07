"""Cross-validation utilities for UCI external validation experiments.

Provides adaptive CV splitting, SMOTE oversampling, per-fold feature
scaling, and native class balancing for the within-dataset evaluation
grid. All functions operate inside the fold loop and follow the same
leakage-safe design: fit on training data only, transform validation.
"""

import numpy as np

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold

from imblearn.over_sampling import SMOTE


# ---------------------------------------------------------------------------
# CV splitting
# ---------------------------------------------------------------------------

def get_cv_splitter(dataset_name, seed=42):
    """Choose CV strategy based on dataset size.

    Cleveland and Hungarian (n ~ 300) use standard 10-fold stratified CV.
    Switzerland and VA Long Beach (n <= 200) use 5-fold x 3 repeats for
    more stable AUC estimates on small samples.
    """
    if dataset_name in ("switzerland", "va_longbeach"):
        return RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=seed)
    return StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)


# ---------------------------------------------------------------------------
# SMOTE oversampling
# ---------------------------------------------------------------------------

# Switzerland has only 8 minority samples, so the default k=5 would fail.
_SMOTE_K = {
    "switzerland": 2,
}
_SMOTE_K_DEFAULT = 5


def apply_smote(x_train, y_train, dataset_name, seed):
    """Apply SMOTE oversampling to the training portion of a fold.

    Uses k_neighbors adapted to dataset size (k=2 for Switzerland where
    the minority class has very few samples). Skips oversampling if the
    minority class has fewer samples than k_neighbors requires.

    The caller provides the random seed directly (typically 42 + fold_idx).
    """
    k_configured = _SMOTE_K.get(dataset_name, _SMOTE_K_DEFAULT)
    min_class_count = min(np.bincount(y_train.astype(int)))
    k = min(k_configured, min_class_count - 1)

    if k < 1:
        return x_train, y_train

    sm = SMOTE(random_state=seed, k_neighbors=k)
    return sm.fit_resample(x_train, y_train)


# ---------------------------------------------------------------------------
# Feature scaling
# ---------------------------------------------------------------------------

def scale_fold(x_tr, x_va):
    """Standardize features per fold (fit on training only).

    Returns numpy arrays. Used for models that need scaled input
    (SVM, KNN, linear models when not using Pipeline-based scaling).
    """
    scaler = StandardScaler()
    x_tr_scaled = scaler.fit_transform(
        x_tr.values if hasattr(x_tr, "values") else np.asarray(x_tr))
    x_va_scaled = scaler.transform(
        x_va.values if hasattr(x_va, "values") else np.asarray(x_va))
    return x_tr_scaled, x_va_scaled


# ---------------------------------------------------------------------------
# Native class balancing
# ---------------------------------------------------------------------------

# Per-family params that enable model-level class balancing.
# Applied when imbalance mode is "native" (as opposed to SMOTE).
NATIVE_BALANCE_PARAMS = {
    "catboost":             {"auto_class_weights": "Balanced"},
    "xgboost":              {},  # scale_pos_weight computed dynamically below
    "lightgbm":             {"is_unbalance": True},
    "random_forest":        {"class_weight": "balanced"},
    "extra_trees":          {"class_weight": "balanced"},
    "logistic_regression":  {"class_weight": "balanced"},
    "ridge":                {"class_weight": "balanced"},
    "lasso":                {},
    "elastic_net":          {},
    "svc":                  {"class_weight": "balanced"},
    "svm":                  {"class_weight": "balanced"},
    "knn":                  {},
    "pytorch_mlp":          {},
    "tabpfn":               {},
    "realmlp":              {},
}


def apply_native_balance(params, family, y_train):
    """Merge native class balancing params into model params.

    For XGBoost, scale_pos_weight is computed from the class ratio
    in the current fold's training data.
    """
    params = dict(params)
    params.update(NATIVE_BALANCE_PARAMS.get(family, {}))

    if family == "xgboost":
        n_neg = int(np.sum(y_train == 0))
        n_pos = int(np.sum(y_train == 1))
        if n_pos > 0:
            params["scale_pos_weight"] = n_neg / n_pos

    return params
