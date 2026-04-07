"""Pretrain SSL encoders on the 630K competition dataset.

Supports four methods via --method:
  scarf         SCARF contrastive learning (NT-Xent loss)
  mae           Standard masked autoencoder (MSE on masked features)
  semi-random   SemiMAE with label as 14th feature, masked randomly at 30%
  semi-always   SemiMAE with label as 14th feature, always masked at 100%

Each method produces a checkpoint in results/checkpoints/ containing
the full model state, encoder state, training config, and loss history.

Usage (from code/ directory):
    python -m experiments.ssl.pretrain --method mae
    python -m experiments.ssl.pretrain --method semi-always --epochs 300
    python -m experiments.ssl.pretrain --method scarf --lr 0.001
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

from experiments.ssl.data import load_competition_data
from experiments.ssl.models import SCARF, MAE

# Reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Auto-detect best available device
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

RESULTS_DIR = str(Path(__file__).resolve().parent / "results" / "checkpoints")


def nt_xent_loss(z1, z2, temperature=0.5):
    """Normalized Temperature-scaled Cross-Entropy loss (NT-Xent).

    Treats (z1[i], z2[i]) as positive pairs and all other combinations
    in the batch as negatives. This is the standard contrastive loss
    used in SimCLR and SCARF.
    """
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    batch_size = z1.shape[0]

    # Cosine similarity matrix between all pairs
    z = torch.cat([z1, z2], dim=0)
    sim = torch.mm(z, z.t()) / temperature

    # Mask out self-similarity on the diagonal
    mask = torch.eye(2 * batch_size, device=z.device).bool()
    sim.masked_fill_(mask, -1e9)

    # Positive pairs: (i, i+N) and (i+N, i)
    pos_idx = torch.arange(batch_size, device=z.device)
    pos_sim = torch.cat([
        sim[pos_idx, pos_idx + batch_size],
        sim[pos_idx + batch_size, pos_idx],
    ])

    # All similarities as negatives (excluding self)
    logits = torch.cat([sim[:batch_size], sim[batch_size:]], dim=0)
    labels = torch.cat([pos_idx + batch_size, pos_idx])

    return F.cross_entropy(logits, labels)


def train_scarf(X_scaled, config):
    """Train SCARF with contrastive loss."""
    print(f"\n  Training SCARF ({config['epochs']} epochs, lr={config['lr']})")

    model = SCARF(
        input_dim=13,
        hidden_dim=config["hidden_dim"],
        latent_dim=config["latent_dim"],
        proj_dim=config["proj_dim"],
        corruption_rate=config["mask_rate"],
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"])

    dataset = TensorDataset(torch.tensor(X_scaled, dtype=torch.float32))
    loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True,
                        drop_last=True, num_workers=0)

    losses = []
    t0 = time.time()

    for epoch in range(config["epochs"]):
        model.train()
        epoch_loss = 0
        n_batches = 0

        for (batch_x,) in loader:
            batch_x = batch_x.to(DEVICE)
            p_orig, p_corrupt = model(batch_x)
            loss = nt_xent_loss(p_orig, p_corrupt)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg = epoch_loss / n_batches
        losses.append(avg)

        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"    Epoch {epoch+1:3d}/{config['epochs']}  loss={avg:.4f}  "
                  f"({time.time() - t0:.0f}s)")

    return model, losses


def train_mae(X_input, config, label_mask_mode=None):
    """Train MAE or SemiMAE with reconstruction loss.

    For standard MAE, X_input is (N, 13) scaled features.
    For SemiMAE, X_input is (N, 14) with the scaled label as the last column.
    """
    input_dim = X_input.shape[1]
    mode_name = f"SemiMAE-{label_mask_mode}" if label_mask_mode else "MAE"
    print(f"\n  Training {mode_name} ({config['epochs']} epochs, "
          f"input_dim={input_dim}, lr={config['lr']})")

    model = MAE(
        input_dim=input_dim,
        hidden_dim=config["hidden_dim"],
        latent_dim=config["latent_dim"],
        mask_rate=config["mask_rate"],
        label_mask_mode=label_mask_mode,
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"])

    dataset = TensorDataset(torch.tensor(X_input, dtype=torch.float32))
    loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True,
                        drop_last=True, num_workers=0)

    losses = []
    t0 = time.time()

    for epoch in range(config["epochs"]):
        model.train()
        epoch_loss = 0
        n_batches = 0

        for (batch_x,) in loader:
            batch_x = batch_x.to(DEVICE)
            x_recon, mask, z = model(batch_x)

            # Reconstruction loss only on masked positions
            loss = F.mse_loss(x_recon[mask], batch_x[mask])

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg = epoch_loss / n_batches
        losses.append(avg)

        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"    Epoch {epoch+1:3d}/{config['epochs']}  MSE={avg:.4f}  "
                  f"({time.time() - t0:.0f}s)")

    return model, losses


def save_checkpoint(model, config, losses, method_name, label_mask_mode=None):
    """Save model checkpoint with full state and training metadata."""
    import os
    os.makedirs(RESULTS_DIR, exist_ok=True)

    save_data = {
        "encoder_state_dict": model.encoder.state_dict(),
        "full_model_state_dict": model.state_dict(),
        "config": config,
        "method": method_name,
        "losses": losses,
    }
    if label_mask_mode:
        save_data["label_mask_mode"] = label_mask_mode

    path = os.path.join(RESULTS_DIR, f"{method_name}_pretrained.pt")
    torch.save(save_data, path)
    print(f"\n  Checkpoint saved: {path}")


def main():
    parser = argparse.ArgumentParser(description="Pretrain SSL encoders")
    parser.add_argument("--method", required=True,
                        choices=["scarf", "mae", "semi-random", "semi-always"],
                        help="SSL method to train")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--mask-rate", type=float, default=0.3)
    args = parser.parse_args()

    config = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "hidden_dim": args.hidden_dim,
        "latent_dim": args.latent_dim,
        "mask_rate": args.mask_rate,
        "proj_dim": 32,  # only used by SCARF
    }

    print(f"Device: {DEVICE}")
    print(f"Method: {args.method}")

    # Load competition data
    print("\nLoading competition data...")
    X_scaled, y, scaler = load_competition_data()
    print(f"  {len(X_scaled):,} samples, {X_scaled.shape[1]} features")

    if args.method == "scarf":
        model, losses = train_scarf(X_scaled, config)
        save_checkpoint(model, config, losses, "scarf")

    elif args.method == "mae":
        model, losses = train_mae(X_scaled, config)
        save_checkpoint(model, config, losses, "mae")

    elif args.method in ("semi-random", "semi-always"):
        # Build 14-dim input: 13 features + scaled label
        label_scaler = StandardScaler()
        y_scaled = label_scaler.fit_transform(y.reshape(-1, 1).astype(np.float32))
        X_with_label = np.hstack([X_scaled, y_scaled]).astype(np.float32)

        label_mask_mode = args.method.split("-")[1]  # "random" or "always"
        method_name = f"semi_mae_{label_mask_mode}"

        model, losses = train_mae(X_with_label, config, label_mask_mode=label_mask_mode)
        save_checkpoint(model, config, losses, method_name, label_mask_mode)

    print("\nDone.")


if __name__ == "__main__":
    main()
