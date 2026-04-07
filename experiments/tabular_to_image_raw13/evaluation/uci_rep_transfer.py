"""Representation transfer UCI evaluation.

Freezes the trained backbone, extracts representations from UCI data,
then trains lightweight classifiers (LogReg, LightGBM, CatBoost, XGBoost)
on those representations using UCI labels with cross-validation.

This tests whether the backbone learned useful features that transfer
to clinical data, independent of the competition-trained classification head.
"""

import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
from catboost import CatBoostClassifier
from xgboost import XGBClassifier

from ..training.utils import extract_reps
from ..config import UCI_CV, DOWNSTREAM_CLASSIFIERS, RANDOM_SEED


def _make_clf(name):
    """Instantiate a downstream classifier by name."""
    s = RANDOM_SEED
    if name == "LogReg":
        return LogisticRegression(max_iter=1000, C=1.0, random_state=s)
    if name == "LightGBM":
        return lgb.LGBMClassifier(n_estimators=100, max_depth=3, learning_rate=0.1,
                                   num_leaves=8, random_state=s, verbose=-1, n_jobs=-1)
    if name == "CatBoost":
        return CatBoostClassifier(iterations=200, depth=4, learning_rate=0.05,
                                   random_seed=s, verbose=0, allow_writing_files=False)
    if name == "XGBoost":
        return XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05,
                              random_state=s, verbosity=0, eval_metric='logloss')


def _run_cv(X, y, n_splits, n_repeats):
    """Train all downstream classifiers with stratified CV, return per-fold metrics."""
    if n_repeats > 1:
        cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=RANDOM_SEED)
    else:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)

    results = {}
    for clf_name in DOWNSTREAM_CLASSIFIERS:
        folds = {"AUC": [], "Accuracy": [], "F1": [], "Precision": [], "Recall": []}

        for tr_idx, te_idx in cv.split(X, y):
            X_tr, X_te = X[tr_idx], X[te_idx]
            y_tr, y_te = y[tr_idx], y[te_idx]

            # Scale representations per fold to avoid leakage
            sc = StandardScaler()
            X_tr = sc.fit_transform(X_tr)
            X_te = sc.transform(X_te)

            clf = _make_clf(clf_name)
            if clf_name == "CatBoost":
                clf.fit(X_tr, y_tr, eval_set=(X_te, y_te), early_stopping_rounds=50)
            elif clf_name == "XGBoost":
                clf.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False)
            else:
                clf.fit(X_tr, y_tr)

            probs = clf.predict_proba(X_te)[:, 1]
            preds = (probs > 0.5).astype(int)
            auc = roc_auc_score(y_te, probs)

            # Handle inverted predictions on small folds
            if auc < 0.5:
                auc = 1 - auc
                preds = 1 - preds

            folds["AUC"].append(float(auc))
            folds["Accuracy"].append(float(accuracy_score(y_te, preds)))
            folds["F1"].append(float(f1_score(y_te, preds, zero_division=0)))
            folds["Precision"].append(float(precision_score(y_te, preds, zero_division=0)))
            folds["Recall"].append(float(recall_score(y_te, preds, zero_division=0)))

        # Aggregate fold results
        results[clf_name] = {
            metric: {"mean": float(np.mean(vals)), "std": float(np.std(vals)), "folds": vals}
            for metric, vals in folds.items()
        }
    return results


def evaluate_rep_transfer(model, uci_datasets, device):
    """Extract backbone reps, train classifiers on UCI, return results per dataset."""
    results = {}
    for name, data in uci_datasets.items():
        cv_cfg = UCI_CV[name]
        reps = extract_reps(model, data["X"], device)
        print(f"    {name}: reps {reps.shape}")

        clf_results = _run_cv(reps, data["y"], cv_cfg["n_splits"], cv_cfg["n_repeats"])
        results[name] = {"n": data["n"], "rep_dim": int(reps.shape[1]), "classifiers": clf_results}

        for clf_name in DOWNSTREAM_CLASSIFIERS:
            auc = clf_results[clf_name]["AUC"]["mean"]
            acc = clf_results[clf_name]["Accuracy"]["mean"]
            print(f"      {clf_name}: AUC={auc:.4f}  Acc={acc:.3f}")

    return results
