"""Tabular deep learning model training (RealMLP, TabPFN).

RealMLP is a tuned feed-forward network from pytabkit with learned
categorical embeddings and validation-based early stopping. TabPFN is
a pretrained tabular transformer that uses in-context learning and
subsample ensembling for datasets larger than its context window.
For small datasets (under the subsample threshold), TabPFN fits on
the full training fold directly.
"""

import gc
import numpy as np
import pandas as pd
import torch

from pytabkit.models.sklearn.sklearn_interfaces import RealMLP_TD_Classifier
from tabpfn import TabPFNClassifier

_SUBSAMPLE_THRESHOLD = 10_000


def train_fold_realmlp(x_tr, y_tr, x_va=None, y_va=None, x_te=None, config=None, seed=42, fold_idx=0):
    """
    Train RealMLP on one fold with optional validation-based early stopping.

    When x_va is provided, uses it for early stopping during training.
    Otherwise trains for the configured number of epochs.
    """
    params = dict(config["params"])
    params["random_state"] = seed

    model = RealMLP_TD_Classifier(**params)

    if x_va is not None and y_va is not None:
        model.fit(x_tr, y_tr, x_va, y_va)
    else:
        model.fit(x_tr, y_tr)

    def _extract_proba(probs):
        return probs[:, 1] if probs.ndim == 2 else probs

    val_probs = _extract_proba(model.predict_proba(x_va)) if x_va is not None else None
    test_probs = _extract_proba(model.predict_proba(x_te)) if x_te is not None else None

    return val_probs, test_probs


def train_fold_tabpfn(x_tr, y_tr, x_va=None, y_va=None, x_te=None, config=None, seed=42, fold_idx=0):
    """
    Train TabPFN on one fold.

    For small datasets (training fold under the subsample threshold),
    TabPFN fits directly on the full training fold with no subsampling.
    For large datasets, it fits on random subsamples and averages
    predictions. Seeds vary by fold and subsample index.
    """
    p = config["params"]
    n_estimators = p["n_estimators"]
    device = p.get("device", "cpu")

    # Convert DataFrames to numpy if needed.
    _v = lambda x: x.values if isinstance(x, pd.DataFrame) else np.asarray(x)
    x_tr_np = _v(x_tr)
    x_va_np = _v(x_va) if x_va is not None else None
    x_te_np = _v(x_te) if x_te is not None else None

    cat_indices = config.get("cat_indices", [])

    # Small dataset: fit directly on the full training fold.
    if len(x_tr_np) <= _SUBSAMPLE_THRESHOLD:
        model = TabPFNClassifier(
            n_estimators=n_estimators,
            device="cpu",  # CPU is faster than MPS for small N
            categorical_features_indices=cat_indices,
            ignore_pretraining_limits=True,
            random_state=seed + fold_idx,
        )
        model.fit(x_tr_np, y_tr)

        va_preds = model.predict_proba(x_va_np)[:, 1] if x_va_np is not None else None
        te_preds = model.predict_proba(x_te_np)[:, 1] if x_te_np is not None else None
        return va_preds, te_preds, model

    # Large dataset: subsample ensembling.
    sub_size = p["sub_size"]
    n_sub = p["n_sub"]

    va_preds = np.zeros(len(x_va_np)) if x_va_np is not None else None
    te_preds = np.zeros(len(x_te_np)) if x_te_np is not None else None
    n_ok = 0

    for si in range(n_sub):
        subseed = seed + fold_idx * 100 + si
        rng = np.random.RandomState(subseed)
        sub_idx = rng.choice(len(x_tr_np), min(sub_size, len(x_tr_np)), replace=False)

        model = TabPFNClassifier(
            n_estimators=n_estimators,
            device=device,
            categorical_features_indices=cat_indices,
            ignore_pretraining_limits=True,
            random_state=subseed,
        )
        model.fit(x_tr_np[sub_idx], y_tr[sub_idx])

        if va_preds is not None:
            va_preds += _batch_predict(model, x_va_np)
        if te_preds is not None:
            te_preds += _batch_predict(model, x_te_np)
        n_ok += 1

        del model
        gc.collect()
        if device == "mps" and torch.backends.mps.is_available():
            torch.mps.empty_cache()

    if va_preds is not None:
        va_preds /= n_ok
    if te_preds is not None:
        te_preds /= n_ok

    return va_preds, te_preds, model


def _batch_predict(model, x, batch_size=5000):
    """Predict in batches to avoid memory issues with large arrays."""
    preds = []
    for i in range(0, len(x), batch_size):
        p = model.predict_proba(x[i:i + batch_size])[:, 1]
        preds.append(p)
    return np.concatenate(preds)


FOLD_TRAINERS = {
    "realmlp": train_fold_realmlp,
    "tabpfn": train_fold_tabpfn,
}
