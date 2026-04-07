"""Evaluate pretrained SSL encoders on competition and UCI data.

Loads any combination of pretrained checkpoints, extracts latent
representations, and measures downstream classification performance
using LogReg, LightGBM, and CatBoost with stratified cross-validation.

Feature configurations tested for each encoder:
  - Raw features only (13-dim baseline)
  - SSL representations only (64-dim)
  - Raw + SSL representations concatenated (77-dim)

For SemiMAE models (14-dim input), inference pads the label column
with the learned mask token so the encoder never sees true labels.

Usage (from code/ directory):
    python -m experiments.ssl.evaluate                           # evaluate all checkpoints found
    python -m experiments.ssl.evaluate --methods mae scarf       # specific methods only
    python -m experiments.ssl.evaluate --skip-competition        # UCI transfer only
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, brier_score_loss, f1_score,
    precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
import lightgbm as lgb
from catboost import CatBoostClassifier
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiments.ssl.data import load_competition_data, load_uci_datasets, CV_CONFIG
from experiments.ssl.models import SCARF, MAE

# Auto-detect device
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

_SSL_DIR = Path(__file__).resolve().parent
CHECKPOINT_DIR = str(_SSL_DIR / "results" / "checkpoints")
METRICS_DIR = str(_SSL_DIR / "results" / "metrics")
FIGURES_DIR = str(_SSL_DIR / "results" / "figures")


# -----------------------------------------------------------------------
# Checkpoint loading
# -----------------------------------------------------------------------

def load_checkpoint(method_name):
    """Load a pretrained model from its checkpoint file.

    Returns (model, config) or (None, None) if the checkpoint is missing.
    """
    path = os.path.join(CHECKPOINT_DIR, f"{method_name}_pretrained.pt")
    if not os.path.exists(path):
        return None, None

    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
    cfg = ckpt["config"]

    # Determine model type from checkpoint metadata
    label_mask_mode = ckpt.get("label_mask_mode", None)

    if "scarf" in method_name:
        model = SCARF(
            input_dim=13,
            hidden_dim=cfg["hidden_dim"],
            latent_dim=cfg["latent_dim"],
            proj_dim=cfg.get("proj_dim", 32),
        ).to(DEVICE)
    elif label_mask_mode is not None:
        # SemiMAE: 14-dim input
        model = MAE(
            input_dim=14,
            hidden_dim=cfg["hidden_dim"],
            latent_dim=cfg["latent_dim"],
            label_mask_mode=label_mask_mode,
        ).to(DEVICE)
    else:
        # Standard MAE: 13-dim input
        model = MAE(
            input_dim=13,
            hidden_dim=cfg["hidden_dim"],
            latent_dim=cfg["latent_dim"],
        ).to(DEVICE)

    model.load_state_dict(ckpt["full_model_state_dict"])
    model.eval()
    print(f"  Loaded {method_name} from {path}")

    return model, cfg


@torch.no_grad()
def extract_representations(model, X_scaled, is_semi=False):
    """Extract encoder representations from scaled features.

    For SemiMAE models (is_semi=True), pads with the learned mask token
    for the label column. For SCARF/MAE, feeds the 13-dim input directly.
    """
    model.eval()
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(DEVICE)

    if is_semi:
        reps = model.encode_without_label(X_tensor)
    else:
        reps = model.encode(X_tensor)

    return reps.cpu().numpy()


# -----------------------------------------------------------------------
# Downstream evaluation
# -----------------------------------------------------------------------

def evaluate_features(X_feat, y, n_splits=10, n_repeats=1, seed=42):
    """Run stratified CV with three classifiers and compute all metrics.

    Returns a nested dict: {classifier: {metric: {"mean", "std"}}}.
    """
    metric_names = ["AUC", "Accuracy", "Precision", "Recall", "F1", "Brier"]

    classifiers = {
        "LogReg": lambda: LogisticRegression(max_iter=1000, C=1.0, random_state=seed),
        "LightGBM": lambda: lgb.LGBMClassifier(
            n_estimators=300, learning_rate=0.05, num_leaves=15,
            subsample=0.8, colsample_bytree=0.8, verbose=-1,
            random_state=seed, min_child_samples=5),
        "CatBoost": lambda: CatBoostClassifier(
            iterations=300, learning_rate=0.05, depth=4,
            verbose=0, random_seed=seed, min_data_in_leaf=5),
    }

    results = {clf: {m: [] for m in metric_names} for clf in classifiers}

    if n_repeats > 1:
        cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats,
                                     random_state=seed)
    else:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    for train_idx, val_idx in cv.split(X_feat, y):
        X_tr, X_va = X_feat[train_idx], X_feat[val_idx]
        y_tr, y_va = y[train_idx], y[val_idx]

        for clf_name, clf_factory in classifiers.items():
            model = clf_factory()

            if clf_name == "LightGBM":
                model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)],
                          callbacks=[lgb.early_stopping(50, verbose=False)])
            elif clf_name == "CatBoost":
                model.fit(X_tr, y_tr, eval_set=(X_va, y_va), early_stopping_rounds=50)
            else:
                model.fit(X_tr, y_tr)

            proba = model.predict_proba(X_va)[:, 1]
            preds = (proba >= 0.5).astype(int)

            results[clf_name]["AUC"].append(roc_auc_score(y_va, proba))
            results[clf_name]["Accuracy"].append(accuracy_score(y_va, preds))
            results[clf_name]["Precision"].append(precision_score(y_va, preds, zero_division=0))
            results[clf_name]["Recall"].append(recall_score(y_va, preds, zero_division=0))
            results[clf_name]["F1"].append(f1_score(y_va, preds, zero_division=0))
            results[clf_name]["Brier"].append(brier_score_loss(y_va, proba))

    # Aggregate
    summary = {}
    for clf_name in classifiers:
        summary[clf_name] = {}
        for metric in metric_names:
            vals = results[clf_name][metric]
            summary[clf_name][metric] = {
                "mean": round(float(np.mean(vals)), 5),
                "std": round(float(np.std(vals)), 5),
            }

    return summary


# -----------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------

def plot_comparison(all_results, title, save_name):
    """Bar chart comparing AUC across methods and classifiers."""
    methods = list(all_results.keys())
    clf_names = ["LogReg", "LightGBM", "CatBoost"]
    colors = ["#2196F3", "#4CAF50", "#FF9800"]

    x = np.arange(len(methods))
    width = 0.25

    fig, ax = plt.subplots(figsize=(max(10, len(methods) * 1.5), 6))
    for i, clf in enumerate(clf_names):
        means = [all_results[m][clf]["AUC"]["mean"] for m in methods]
        stds = [all_results[m][clf]["AUC"]["std"] for m in methods]
        ax.bar(x + i * width, means, width, yerr=stds, label=clf,
               color=colors[i], alpha=0.85, capsize=3)

    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_ylabel("AUC")
    ax.set_xticks(x + width)
    ax.set_xticklabels(methods, rotation=25, ha="right", fontsize=9)
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)

    # Tight y-axis around the data range
    all_means = [all_results[m][c]["AUC"]["mean"]
                 for m in methods for c in clf_names]
    if all_means:
        ax.set_ylim(max(0.5, min(all_means) - 0.02), min(1.0, max(all_means) + 0.015))

    plt.tight_layout()
    os.makedirs(FIGURES_DIR, exist_ok=True)
    path = os.path.join(FIGURES_DIR, save_name)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate SSL encoders")
    parser.add_argument("--methods", nargs="+", default=None,
                        help="Methods to evaluate (default: all found checkpoints)")
    parser.add_argument("--skip-competition", action="store_true",
                        help="Skip competition data evaluation")
    parser.add_argument("--skip-uci", action="store_true",
                        help="Skip UCI transfer evaluation")
    args = parser.parse_args()

    # Discover available checkpoints
    all_methods = {
        "scarf": False,           # is_semi=False
        "mae": False,
        "semi_mae_random": True,  # is_semi=True
        "semi_mae_always": True,
    }

    if args.methods:
        # Filter to requested methods
        all_methods = {k: v for k, v in all_methods.items() if k in args.methods}

    # Load models
    models = {}
    for method_name, is_semi in all_methods.items():
        model, cfg = load_checkpoint(method_name)
        if model is not None:
            models[method_name] = {"model": model, "config": cfg, "is_semi": is_semi}

    if not models:
        print("No checkpoints found. Run pretrain.py first.")
        return

    # Load data
    print("\nLoading competition data...")
    X_scaled, y_comp, scaler = load_competition_data()
    print(f"  {len(X_scaled):,} samples")

    # Display names for cleaner output
    display_names = {
        "scarf": "SCARF", "mae": "MAE",
        "semi_mae_random": "SemiMAE-random", "semi_mae_always": "SemiMAE-always",
    }

    # ------------------------------------------------------------------
    # Competition data evaluation
    # ------------------------------------------------------------------
    if not args.skip_competition:
        print("\n" + "=" * 60)
        print("COMPETITION DATA EVALUATION")
        print("=" * 60)

        comp_results = {}

        # Raw baseline
        print("\n  Raw features (13-dim)...")
        comp_results["Raw"] = evaluate_features(X_scaled, y_comp)

        for method_name, info in models.items():
            dname = display_names.get(method_name, method_name)
            reps = extract_representations(info["model"], X_scaled, info["is_semi"])

            # Representations only
            print(f"\n  {dname} reps ({reps.shape[1]}-dim)...")
            comp_results[f"{dname} reps"] = evaluate_features(reps, y_comp)

            # Raw + representations
            combined = np.hstack([X_scaled, reps])
            print(f"  Raw+{dname} ({combined.shape[1]}-dim)...")
            comp_results[f"Raw+{dname}"] = evaluate_features(combined, y_comp)

        # Print summary
        print(f"\n{'Method':<25} {'LogReg':>10} {'LightGBM':>10} {'CatBoost':>10}")
        print("-" * 60)
        for name, res in comp_results.items():
            lr = f"{res['LogReg']['AUC']['mean']:.4f}"
            lgb_s = f"{res['LightGBM']['AUC']['mean']:.4f}"
            cb = f"{res['CatBoost']['AUC']['mean']:.4f}"
            print(f"{name:<25} {lr:>10} {lgb_s:>10} {cb:>10}")

        plot_comparison(comp_results, "Competition Data - SSL Comparison", "competition_comparison.png")

        # Save
        os.makedirs(METRICS_DIR, exist_ok=True)
        with open(os.path.join(METRICS_DIR, "competition_results.json"), "w") as f:
            json.dump(comp_results, f, indent=2)

    # ------------------------------------------------------------------
    # UCI transfer evaluation
    # ------------------------------------------------------------------
    if not args.skip_uci:
        print("\n" + "=" * 60)
        print("UCI TRANSFER EVALUATION")
        print("=" * 60)

        print("\nLoading UCI datasets...")
        uci_datasets = load_uci_datasets()

        uci_results = {}  # {method: {dataset: eval_results}}

        for ds_name, data in uci_datasets.items():
            print(f"\n--- {ds_name} (n={data['n']}) ---")
            X_uci_scaled = scaler.transform(data["X"])
            y_uci = data["y"]
            cv = CV_CONFIG.get(ds_name, {"n_splits": 5, "n_repeats": 3})

            # Raw baseline
            key = "Raw"
            uci_results.setdefault(key, {})
            print(f"  {key}...")
            uci_results[key][ds_name] = evaluate_features(
                X_uci_scaled, y_uci, cv["n_splits"], cv["n_repeats"])

            for method_name, info in models.items():
                dname = display_names.get(method_name, method_name)
                reps = extract_representations(info["model"], X_uci_scaled, info["is_semi"])

                # Representations only
                key = f"{dname} reps"
                uci_results.setdefault(key, {})
                print(f"  {key} ({reps.shape[1]}-dim)...")
                uci_results[key][ds_name] = evaluate_features(
                    reps, y_uci, cv["n_splits"], cv["n_repeats"])

                # Raw + representations
                combined = np.hstack([X_uci_scaled, reps])
                key = f"Raw+{dname}"
                uci_results.setdefault(key, {})
                print(f"  {key} ({combined.shape[1]}-dim)...")
                uci_results[key][ds_name] = evaluate_features(
                    combined, y_uci, cv["n_splits"], cv["n_repeats"])

        # Print UCI summary
        ds_names = list(uci_datasets.keys())
        print(f"\n{'Method':<25}" + "".join(f"{d:>16}" for d in ds_names))
        print("-" * (25 + 16 * len(ds_names)))
        for method_key in uci_results:
            row = f"{method_key:<25}"
            for ds_name in ds_names:
                if ds_name in uci_results[method_key]:
                    res = uci_results[method_key][ds_name]
                    best_clf = max(["LogReg", "LightGBM", "CatBoost"],
                                   key=lambda c: res[c]["AUC"]["mean"])
                    auc = res[best_clf]["AUC"]["mean"]
                    row += f"{auc:>12.3f} ({best_clf[:2]})"
                else:
                    row += f"{'N/A':>16}"
            print(row)

        # Save
        os.makedirs(METRICS_DIR, exist_ok=True)
        with open(os.path.join(METRICS_DIR, "uci_transfer_results.json"), "w") as f:
            json.dump(uci_results, f, indent=2)

    print("\nDone.")


if __name__ == "__main__":
    main()
