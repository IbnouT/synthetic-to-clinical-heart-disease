"""K-nearest neighbors model training.

KNN is distance-based, so feature scaling matters. The full_train_scaling
flag in model config triggers StandardScaler on the full training set
before the fold loop, ensuring consistent distance computations.
"""

from sklearn.neighbors import KNeighborsClassifier


def train_fold_knn(x_tr, y_tr, x_va=None, y_va=None, x_te=None, config=None, seed=42, fold_idx=0):
    """Fit KNN on one fold and return val/test predictions."""
    model = KNeighborsClassifier(**config["params"])
    model.fit(x_tr, y_tr)

    val_scores = model.predict_proba(x_va)[:, 1] if x_va is not None else None
    test_scores = model.predict_proba(x_te)[:, 1] if x_te is not None else None
    return val_scores, test_scores, model


FOLD_TRAINERS = {
    "knn": train_fold_knn,
}
