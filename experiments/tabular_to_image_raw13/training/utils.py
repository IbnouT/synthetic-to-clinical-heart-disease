"""Training utilities: device selection, training loop, batched inference."""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score

from ..config import BATCH_SIZE


def get_device(force=None):
    """Auto-detect MPS (Apple Silicon GPU), fallback to CPU."""
    if force == "cpu":
        return torch.device("cpu")
    if force == "mps" or (force is None and torch.backends.mps.is_available()):
        return torch.device("mps")
    return torch.device("cpu")


def train_loop(model, X_train, y_train, X_val, y_val, device,
               epochs=100, lr=1e-3, patience=10):
    """Supervised training with early stopping on validation AUC.

    Stops if validation AUC doesn't improve for `patience` epochs.
    Returns the model state at the best validation AUC, not the last epoch.
    """
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.BCEWithLogitsLoss()

    loader = DataLoader(
        TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                      torch.tensor(y_train, dtype=torch.float32)),
        batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    best_auc, best_preds, best_state = 0.0, None, None
    no_improve = 0

    for epoch in range(epochs):
        # Training pass
        model.train()
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            loss = criterion(model(bx).squeeze(-1), by)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()

        # Validation: check if this is the best epoch so far
        val_preds = predict(model, X_val, device)
        val_auc = roc_auc_score(y_val, val_preds)

        if val_auc > best_auc:
            best_auc = val_auc
            best_preds = val_preds.copy()
            # Save state to CPU so it doesn't hold GPU memory
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch+1}/{epochs}: AUC={val_auc:.6f} (best={best_auc:.6f})")

        if no_improve >= patience:
            print(f"    Early stopping at epoch {epoch+1}")
            break

    # Restore the best model, not the last epoch
    model.load_state_dict(best_state)
    return best_auc, best_preds, best_state


def predict(model, X, device):
    """Batched inference on raw 13-dim features, returns probabilities."""
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(X), BATCH_SIZE):
            chunk = torch.tensor(X[i:i+BATCH_SIZE], dtype=torch.float32).to(device)
            preds.append(torch.sigmoid(model(chunk).squeeze(-1)).cpu().numpy())
    return np.concatenate(preds)


def extract_reps(model, X, device):
    """Batched backbone extraction on raw 13-dim features, returns representations."""
    model.eval()
    reps = []
    with torch.no_grad():
        for i in range(0, len(X), BATCH_SIZE):
            chunk = torch.tensor(X[i:i+BATCH_SIZE], dtype=torch.float32).to(device)
            reps.append(model.extract(chunk).cpu().numpy())
    return np.concatenate(reps)
