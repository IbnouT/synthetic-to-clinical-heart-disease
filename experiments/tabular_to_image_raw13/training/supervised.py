"""Supervised training: expansion + backbone trained end-to-end.

Creates an ExpansionModel (expansion layer + backbone), trains it on
competition data with binary cross-entropy, and returns the trained model
along with competition metrics and timing.
"""

import time

from .utils import train_loop
from ..models.expansion import ExpansionModel


def train_supervised(backbone, X_train, y_train, X_val, y_val, device, config):
    """Train end-to-end and return (model, auc, predictions, elapsed_seconds)."""
    model = ExpansionModel(backbone)
    print(f"    Parameters: {sum(p.numel() for p in model.parameters()):,}")

    t0 = time.time()
    auc, preds, state = train_loop(
        model, X_train, y_train, X_val, y_val, device,
        epochs=config["epochs"], lr=config["lr"])
    elapsed = time.time() - t0

    model.load_state_dict(state)
    print(f"    Best AUC: {auc:.6f} ({elapsed/60:.1f} min)")
    return model, auc, preds, elapsed
