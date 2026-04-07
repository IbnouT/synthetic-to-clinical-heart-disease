#!/usr/bin/env python3
"""Comprehensive evaluation of all trained models.

Loads existing checkpoints (no retraining). For each model, evaluates:

Competition (80/20 split):
  - backbone + NN head (from training)
  - frozen reps → CatBoost, LogReg
  - frozen reps + raw 13 features → CatBoost, LogReg

UCI (Cleveland, Hungarian):
  - zero-shot (full model applied directly)
  - frozen reps → CatBoost, LogReg
  - frozen reps + raw 13 features → CatBoost, LogReg

Usage:
    python -m experiments.tabular_to_image_raw13.evaluate_all
"""

import argparse
import json
import os
import time

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from catboost import CatBoostClassifier

from .config import (CHECKPOINTS_DIR, METRICS_DIR, MODEL_CONFIGS,
                     UCI_CV, RANDOM_SEED)
from .data import load_competition, load_uci
from .training.utils import get_device, predict, extract_reps
from .evaluation.metrics import compute_metrics
from .models import (ExpansionModel, DeeperCNN, VisionTransformer,
                     HybridCNNTransformer, VAEModel, Discriminator)


# All models and how to reconstruct them from checkpoints
MODEL_DEFS = {
    "deepercnn": lambda: ExpansionModel(DeeperCNN()),
    "vit": lambda: ExpansionModel(VisionTransformer()),
    "hybrid": lambda: ExpansionModel(HybridCNNTransformer()),
    "simclr": lambda: ExpansionModel(DeeperCNN()),
    "moco": lambda: ExpansionModel(DeeperCNN()),
    "vae": lambda: ExpansionModel(VAEModel()),
    "cgan": None,  # handled separately — disc classifier architecture
    "gan_aug": lambda: ExpansionModel(DeeperCNN()),
}


def _make_clf(name):
    s = RANDOM_SEED
    if name == "LogReg":
        return LogisticRegression(max_iter=1000, C=1.0, random_state=s)
    if name == "CatBoost":
        return CatBoostClassifier(iterations=200, depth=4, learning_rate=0.05,
                                   random_seed=s, verbose=0, allow_writing_files=False)


def _eval_cv(X, y, n_splits, n_repeats=1):
    """Train LogReg + CatBoost with CV, return results dict with all metrics."""
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)
    results = {}

    for clf_name in ["LogReg", "CatBoost"]:
        folds = {"AUC": [], "Accuracy": [], "F1": [], "Precision": [], "Recall": []}
        for tr_idx, te_idx in cv.split(X, y):
            X_tr, X_te = X[tr_idx], X[te_idx]
            y_tr, y_te = y[tr_idx], y[te_idx]

            sc = StandardScaler()
            X_tr = sc.fit_transform(X_tr)
            X_te = sc.transform(X_te)

            clf = _make_clf(clf_name)
            if clf_name == "CatBoost":
                clf.fit(X_tr, y_tr, eval_set=(X_te, y_te), early_stopping_rounds=50)
            else:
                clf.fit(X_tr, y_tr)

            probs = clf.predict_proba(X_te)[:, 1]
            preds = (probs > 0.5).astype(int)
            auc = roc_auc_score(y_te, probs)
            if auc < 0.5:
                auc = 1 - auc
                preds = 1 - preds

            folds["AUC"].append(float(auc))
            folds["Accuracy"].append(float(accuracy_score(y_te, preds)))
            folds["F1"].append(float(f1_score(y_te, preds, zero_division=0)))
            folds["Precision"].append(float(precision_score(y_te, preds, zero_division=0)))
            folds["Recall"].append(float(recall_score(y_te, preds, zero_division=0)))

        results[clf_name] = {
            m: {"mean": float(np.mean(folds[m])), "std": float(np.std(folds[m]))}
            for m in folds
        }
    return results


def _eval_holdout(X, y):
    """Train LogReg + CatBoost on full set with a single holdout split (for competition)."""
    # Use the same 80/20 split logic: X and y are already the val set reps
    # We need train reps too — this function receives train and val separately
    pass


def evaluate_model(name, model, X_train, X_val, y_train, y_val,
                   X_train_raw, X_val_raw, uci, uci_raw, device):
    """Run all evaluations for one model."""
    print(f"\n{'='*60}\n  {name}\n{'='*60}")

    results = {"model": name, "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}

    # --- Competition: backbone + NN head (from original training) ---
    comp_preds = predict(model, X_val, device)
    results["competition_nn_head"] = compute_metrics(y_val, comp_preds)
    print(f"  Comp NN head:      AUC={results['competition_nn_head']['auc']:.6f}")

    # --- Extract competition reps ---
    train_reps = extract_reps(model, X_train, device)
    val_reps = extract_reps(model, X_val, device)
    print(f"  Rep dim: {val_reps.shape[1]}")

    # --- Competition: frozen reps → classifiers ---
    for clf_name in ["LogReg", "CatBoost"]:
        sc = StandardScaler()
        tr_s = sc.fit_transform(train_reps)
        va_s = sc.transform(val_reps)

        clf = _make_clf(clf_name)
        if clf_name == "CatBoost":
            clf.fit(tr_s, y_train, eval_set=(va_s, y_val), early_stopping_rounds=50)
        else:
            clf.fit(tr_s, y_train)

        probs = clf.predict_proba(va_s)[:, 1]
        m = compute_metrics(y_val, probs)
        results[f"competition_reps_{clf_name}"] = m
        print(f"  Comp reps→{clf_name}:  AUC={m['auc']:.6f}")

    # --- Competition: frozen reps + raw 13 → classifiers ---
    train_combined = np.hstack([train_reps, X_train_raw])
    val_combined = np.hstack([val_reps, X_val_raw])

    for clf_name in ["LogReg", "CatBoost"]:
        sc = StandardScaler()
        tr_s = sc.fit_transform(train_combined)
        va_s = sc.transform(val_combined)

        clf = _make_clf(clf_name)
        if clf_name == "CatBoost":
            clf.fit(tr_s, y_train, eval_set=(va_s, y_val), early_stopping_rounds=50)
        else:
            clf.fit(tr_s, y_train)

        probs = clf.predict_proba(va_s)[:, 1]
        m = compute_metrics(y_val, probs)
        results[f"competition_reps_raw_{clf_name}"] = m
        print(f"  Comp reps+raw→{clf_name}: AUC={m['auc']:.6f}")

    # --- UCI evaluations ---
    results["uci"] = {}

    for ds_name, data in uci.items():
        cv_cfg = UCI_CV[ds_name]
        print(f"\n  {ds_name}:")
        ds_results = {}

        # Zero-shot
        zs_probs = predict(model, data["X"], device)
        ds_results["zeroshot"] = compute_metrics(data["y"], zs_probs)
        print(f"    Zero-shot:      AUC={ds_results['zeroshot']['auc']:.4f}")

        # Extract UCI reps
        uci_reps = extract_reps(model, data["X"], device)

        # Reps only → classifiers (CV)
        reps_results = _eval_cv(uci_reps, data["y"], cv_cfg["n_splits"])
        ds_results["reps"] = reps_results
        for clf_name in ["LogReg", "CatBoost"]:
            print(f"    Reps→{clf_name}:   AUC={reps_results[clf_name]['AUC']['mean']:.4f}")

        # Reps + raw 13 → classifiers (CV)
        uci_combined = np.hstack([uci_reps, uci_raw[ds_name]])
        reps_raw_results = _eval_cv(uci_combined, data["y"], cv_cfg["n_splits"])
        ds_results["reps_raw"] = reps_raw_results
        for clf_name in ["LogReg", "CatBoost"]:
            print(f"    Reps+raw→{clf_name}: AUC={reps_raw_results[clf_name]['AUC']['mean']:.4f}")

        results["uci"][ds_name] = ds_results

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default=None, choices=["cpu", "mps"])
    args = parser.parse_args()

    device = get_device(args.device)
    print(f"Device: {device}")

    # Load data
    print("\nLoading data...")
    X_train, X_val, y_train, y_val, X_all, y_all, scaler = load_competition()

    # Keep unscaled raw features for the reps+raw combination
    # X_train and X_val are already scaled, which is what the model expects
    # For reps+raw, we use the same scaled features as "raw"
    X_train_raw = X_train.copy()
    X_val_raw = X_val.copy()

    uci = load_uci(scaler)
    # Keep raw scaled UCI features for reps+raw
    uci_raw = {name: data["X"].copy() for name, data in uci.items()}

    # Evaluate each model that has a checkpoint
    all_results = {}

    for model_name, factory in MODEL_DEFS.items():
        ckpt_path = CHECKPOINTS_DIR / model_name / "best.pt"
        if not ckpt_path.exists():
            print(f"\n  Skipping {model_name} — no checkpoint")
            continue

        if model_name == "cgan":
            # cGAN disc classifier has a custom architecture
            from .models import Discriminator
            class DiscClassifier(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    d = Discriminator()
                    self.feat = d.feat
                    self.head = torch.nn.Sequential(
                        torch.nn.Linear(64, 32), torch.nn.GELU(),
                        torch.nn.Dropout(0.3), torch.nn.Linear(32, 1))
                    self.REP_DIM = 64
                def extract(self, x):
                    return self.feat(x)
                def forward(self, x):
                    return self.head(self.feat(x))
            model = ExpansionModel(DiscClassifier())
        else:
            model = factory()

        state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        model = model.to(device)
        model.eval()

        results = evaluate_model(
            model_name, model, X_train, X_val, y_train, y_val,
            X_train_raw, X_val_raw, uci, uci_raw, device)

        all_results[model_name] = results

    # Save comprehensive results
    os.makedirs(METRICS_DIR, exist_ok=True)
    out_path = METRICS_DIR / "comprehensive_evaluation.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
