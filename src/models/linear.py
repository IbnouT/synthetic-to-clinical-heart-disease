"""Linear model training (Logistic Regression, Ridge, SVM).

Each function trains one CV fold and returns validation and test scores.
When x_va is None (e.g. zero-shot transfer), validation predictions are
skipped. When x_te is None (e.g. UCI within-dataset CV), test predictions
are skipped.
"""

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC


def train_fold_logistic(x_tr, y_tr, x_va=None, y_va=None, x_te=None, config=None, seed=42, fold_idx=0):
    """Train logistic regression on one fold."""
    model = LogisticRegression(random_state=seed, **config["params"])
    model.fit(x_tr, y_tr)

    val_scores = model.predict_proba(x_va)[:, 1] if x_va is not None else None
    test_scores = model.predict_proba(x_te)[:, 1] if x_te is not None else None
    return val_scores, test_scores, model


def train_fold_ridge(x_tr, y_tr, x_va=None, y_va=None, x_te=None, config=None, seed=42, fold_idx=0):
    """
    Train Ridge classifier on one fold.

    Ridge has no predict_proba method, so we use the raw decision_function
    output as continuous scores. Higher values indicate higher confidence
    in the positive class, which works correctly with AUC.
    """
    model = RidgeClassifier(**config["params"])
    model.fit(x_tr, y_tr)

    val_scores = model.decision_function(x_va) if x_va is not None else None
    test_scores = model.decision_function(x_te) if x_te is not None else None
    return val_scores, test_scores, model


def train_fold_svm(x_tr, y_tr, x_va=None, y_va=None, x_te=None, config=None, seed=42, fold_idx=0):
    """
    Train LinearSVC with isotonic calibration on one fold.

    LinearSVC outputs decision values, not probabilities.
    CalibratedClassifierCV wraps it with isotonic regression (3-fold
    internal CV) to produce calibrated probability scores.
    """
    base = LinearSVC(random_state=seed, **config["params"])
    model = CalibratedClassifierCV(base, cv=3, method="isotonic")
    model.fit(x_tr, y_tr)

    val_scores = model.predict_proba(x_va)[:, 1] if x_va is not None else None
    test_scores = model.predict_proba(x_te)[:, 1] if x_te is not None else None
    return val_scores, test_scores, model


def train_fold_svc(x_tr, y_tr, x_va=None, y_va=None, x_te=None, config=None, seed=42, fold_idx=0):
    """
    Train SVC with flexible kernel selection on one fold.

    Unlike LinearSVC, SVC supports rbf, poly, and sigmoid kernels.
    Probability estimates use Platt scaling (probability=True).

    For large training sets (>50K rows), subsamples to 50K with a
    fixed seed for reproducibility. SVC has O(n^2) time complexity
    so this guard prevents multi-hour training on competition data.
    """
    params = dict(config["params"])
    params["probability"] = True

    # Subsample guard for large datasets (SVC is O(n^2)).
    max_train = 50000
    n = x_tr.shape[0]
    if n > max_train:
        rng = np.random.RandomState(42)
        idx = rng.choice(n, size=max_train, replace=False)
        idx.sort()
        x_tr = x_tr.iloc[idx] if hasattr(x_tr, "iloc") else x_tr[idx]
        y_tr = y_tr[idx]

    # Standardize inputs before SVC for consistent kernel behavior.
    model = Pipeline(
        steps=[
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("model", SVC(random_state=seed, **params)),
        ]
    )
    model.fit(x_tr, y_tr)

    val_scores = model.predict_proba(x_va)[:, 1] if x_va is not None else None
    test_scores = model.predict_proba(x_te)[:, 1] if x_te is not None else None
    return val_scores, test_scores, model


FOLD_TRAINERS = {
    "logistic_regression": train_fold_logistic,
    "ridge": train_fold_ridge,
    "svm": train_fold_svm,
    "svc": train_fold_svc,
}
