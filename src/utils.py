"""Helpers for saving and loading predictions and metric summaries.

Training runs produce three types of output files:
  - OOF predictions (numpy arrays covering every training row)
  - Test predictions (numpy arrays averaged across folds)
  - Metrics JSON files (AUC, accuracy, F1, etc. plus run metadata)

All files are named by the experiment ID so they can be matched back
to the model configuration that produced them.
"""

import json
import numpy as np

from src.config import METRICS_DIR
from src.config import OOF_DIR
from src.config import TEST_PREDS_DIR


# ---------------------------------------------------------------------------
# OOF predictions
# ---------------------------------------------------------------------------

def save_oof(experiment_id, predictions):
    """Save the full out-of-fold prediction vector for one experiment."""
    np.save(OOF_DIR / f"{experiment_id}_oof.npy", predictions)


def load_oof(experiment_id):
    """Load a previously saved out-of-fold prediction vector."""
    return np.load(OOF_DIR / f"{experiment_id}_oof.npy")


# ---------------------------------------------------------------------------
# Test predictions
# ---------------------------------------------------------------------------

def save_test_preds(experiment_id, predictions):
    """Save the fold-averaged test predictions for one experiment."""
    np.save(TEST_PREDS_DIR / f"{experiment_id}_test.npy", predictions)


def load_test_preds(experiment_id):
    """Load previously saved test predictions."""
    return np.load(TEST_PREDS_DIR / f"{experiment_id}_test.npy")


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def save_metrics(experiment_id, metrics, metadata):
    """Save metrics and run metadata together as a single JSON file."""
    payload = {
        "id": experiment_id,
        "metrics": metrics,
        "metadata": metadata,
    }
    path = METRICS_DIR / f"{experiment_id}_metrics.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def load_metrics(experiment_id):
    """Load a previously saved metrics JSON file."""
    path = METRICS_DIR / f"{experiment_id}_metrics.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_model(experiment_id, model, fold_idx):
    """Save a trained model to disk (joblib for sklearn, native for boosters)."""
    import joblib
    from src.config import SAVED_MODELS_DIR
    model_dir = SAVED_MODELS_DIR / experiment_id
    model_dir.mkdir(parents=True, exist_ok=True)
    path = model_dir / f"fold_{fold_idx}.joblib"
    joblib.dump(model, path)
    return path
