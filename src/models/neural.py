"""PyTorch neural network model training.

The MLP uses a simple feed-forward architecture with BatchNorm and
Dropout between each hidden layer. Training uses Adam with cosine
annealing and early stopping on validation AUC (when validation data
is available). When x_va is None, trains for the full epoch count.
"""

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score


class SimpleMLP(nn.Module):
    """Feed-forward network: [Linear -> BatchNorm -> ReLU -> Dropout] x N -> Linear(1)."""

    def __init__(self, input_dim, hidden_dims, dropout=0.3):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.extend([
                nn.Linear(prev, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_fold_mlp(x_tr, y_tr, x_va=None, y_va=None, x_te=None, config=None, seed=42, fold_idx=0):
    """
    Train a SimpleMLP on one fold with optional early stopping.

    Scales features per fold, trains with BCEWithLogitsLoss and Adam.
    When validation data is provided, restores the best model state
    (by validation AUC). Otherwise trains for full epoch count.
    """
    p = config["params"]
    hidden_dims = p["hidden_dims"]
    dropout = p["dropout"]
    lr = p["lr"]
    epochs = p["epochs"]

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # Ensure numpy arrays (features may arrive as DataFrames).
    _v = lambda x: x.values if isinstance(x, pd.DataFrame) else np.asarray(x)
    x_tr_np = _v(x_tr)

    # Scale features for this fold.
    scaler = StandardScaler()
    x_tr_t = torch.tensor(scaler.fit_transform(x_tr_np), dtype=torch.float32).to(device)
    y_tr_t = torch.tensor(y_tr, dtype=torch.float32).to(device)

    # Prepare validation and test tensors when available.
    x_va_t = None
    if x_va is not None:
        x_va_t = torch.tensor(scaler.transform(_v(x_va)), dtype=torch.float32).to(device)
    x_te_t = None
    if x_te is not None:
        x_te_t = torch.tensor(scaler.transform(_v(x_te)), dtype=torch.float32).to(device)

    model = SimpleMLP(x_tr_t.shape[1], hidden_dims, dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.BCEWithLogitsLoss()

    loader = DataLoader(TensorDataset(x_tr_t, y_tr_t), batch_size=4096, shuffle=True)

    best_auc = 0.0
    best_state = None
    no_improve = 0
    patience = 30

    for epoch in range(epochs):
        # Training pass.
        model.train()
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
        scheduler.step()

        # Early stopping on validation AUC (only when validation data exists).
        if x_va_t is not None:
            model.eval()
            with torch.no_grad():
                va_logits = model(x_va_t).cpu().numpy()
                va_probs = 1.0 / (1.0 + np.exp(-va_logits))
            auc = roc_auc_score(y_va, va_probs)

            if auc > best_auc:
                best_auc = auc
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    break

    # Restore best model if early stopping was used.
    if best_state is not None:
        model.load_state_dict(best_state)

    # Get final predictions.
    model.eval()
    with torch.no_grad():
        val_scores = torch.sigmoid(model(x_va_t)).cpu().numpy() if x_va_t is not None else None
        test_scores = torch.sigmoid(model(x_te_t)).cpu().numpy() if x_te_t is not None else None

    return val_scores, test_scores, model


FOLD_TRAINERS = {
    "mlp": train_fold_mlp,
    "pytorch_mlp": train_fold_mlp,  # alternative name used by some scripts
}
