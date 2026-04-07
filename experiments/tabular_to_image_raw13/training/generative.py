"""Generative model training: VAE, cGAN, and GAN-augmented CNN."""

import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from .utils import train_loop
from ..models.expansion import ExpansionLayer, ExpansionModel
from ..config import BATCH_SIZE, IMG_SIZE

N_PIXELS = IMG_SIZE * IMG_SIZE


def train_vae(vae_model, X_all, y_all, X_train, y_train, X_val, y_val,
              device, config):
    """VAE: reconstruction pretrain -> finetune classifier on latent.

    Returns (model, pretrain_state, comp_auc, comp_preds, elapsed).
    """
    expansion = ExpansionLayer().to(device)
    vae_model = vae_model.to(device)

    opt = torch.optim.AdamW(
        list(expansion.parameters()) + list(vae_model.parameters()),
        lr=config["pretrain_lr"], weight_decay=1e-5)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=config["pretrain_epochs"])

    loader = DataLoader(
        TensorDataset(torch.tensor(X_all, dtype=torch.float32)),
        batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    print(f"    VAE pretraining ({config['pretrain_epochs']} epochs)...")
    t0 = time.time()

    for ep in range(config["pretrain_epochs"]):
        expansion.train(); vae_model.train()
        total, n = 0, 0
        # KL weight warmup: gradually increase from near-zero
        kl_w = min(0.01, 0.0001 + ep * 0.01 / config["pretrain_epochs"])
        for (bx,) in loader:
            bx = bx.to(device)
            img = expansion(bx)
            recon, mu, lv = vae_model.forward_vae(img)
            # Reconstruction loss + KL divergence
            loss = F.mse_loss(recon, img) + kl_w * (-0.5 * (1 + lv - mu**2 - lv.exp()).mean())
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item(); n += 1
        sch.step()
        if (ep + 1) % 20 == 0:
            print(f"      Epoch {ep+1}/{config['pretrain_epochs']}: loss={total/n:.4f}")

    pretrain_state = {
        "expansion": {k: v.cpu().clone() for k, v in expansion.state_dict().items()},
        "vae": {k: v.cpu().clone() for k, v in vae_model.state_dict().items()},
    }

    # Finetune classifier head on latent
    print(f"    Finetuning classifier ({config['finetune_epochs']} epochs)...")
    # Reset classifier weights so it learns fresh on the pretrained latent
    for layer in vae_model.classifier:
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()

    model = ExpansionModel(vae_model)
    model.expansion.load_state_dict(pretrain_state["expansion"])

    auc, preds, _ = train_loop(
        model, X_train, y_train, X_val, y_val, device,
        epochs=config["finetune_epochs"], lr=config["finetune_lr"])

    elapsed = time.time() - t0
    print(f"    Best AUC: {auc:.6f} ({elapsed/60:.1f} min total)")

    return model, pretrain_state, auc, preds, elapsed


def train_cgan(generator, discriminator, X_all, y_all, X_train, y_train,
               X_val, y_val, device, config):
    """cGAN: train GAN -> use disc features -> finetune classifier.

    Returns (disc_model, gan_state, comp_auc, comp_preds, elapsed).
    """
    expansion = ExpansionLayer().to(device)
    G = generator.to(device)
    D = discriminator.to(device)

    oG = torch.optim.Adam(
        list(expansion.parameters()) + list(G.parameters()),
        lr=config["gan_lr"], betas=(0.5, 0.999))
    oD = torch.optim.Adam(D.parameters(), lr=config["gan_lr"], betas=(0.5, 0.999))

    loader = DataLoader(
        TensorDataset(torch.tensor(X_all, dtype=torch.float32),
                      torch.tensor(y_all, dtype=torch.float32)),
        batch_size=1024, shuffle=True, drop_last=True)

    print(f"    cGAN training ({config['gan_epochs']} epochs)...")
    t0 = time.time()

    for ep in range(config["gan_epochs"]):
        expansion.train(); G.train(); D.train()
        dl, gl, n = 0, 0, 0
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            ri = expansion(bx)
            bs = ri.shape[0]

            # Discriminator step
            z = torch.randn(bs, config["latent_dim"], device=device)
            fl = torch.randint(0, 2, (bs,), device=device).float()
            fi = G(z, fl).detach()
            dr, dc, _ = D(ri)
            df, _, _ = D(fi)
            ld = (F.binary_cross_entropy_with_logits(dr, 0.9 * torch.ones_like(dr)) +
                  F.binary_cross_entropy_with_logits(df, 0.1 * torch.ones_like(df)) +
                  0.5 * F.binary_cross_entropy_with_logits(dc.squeeze(), by))
            oD.zero_grad(); ld.backward(); oD.step()

            # Generator step
            z = torch.randn(bs, config["latent_dim"], device=device)
            fl = torch.randint(0, 2, (bs,), device=device).float()
            fi = G(z, fl)
            dg, dcg, _ = D(fi)
            lg = (F.binary_cross_entropy_with_logits(dg, torch.ones(bs, 1, device=device)) +
                  0.5 * F.binary_cross_entropy_with_logits(dcg.squeeze(), fl))
            oG.zero_grad(); lg.backward(); oG.step()

            dl += ld.item(); gl += lg.item(); n += 1
        if (ep + 1) % 20 == 0:
            print(f"      Epoch {ep+1}/{config['gan_epochs']}: D={dl/n:.3f} G={gl/n:.3f}")

    gan_state = {
        "expansion": {k: v.cpu().clone() for k, v in expansion.state_dict().items()},
        "generator": {k: v.cpu().clone() for k, v in G.state_dict().items()},
        "discriminator": {k: v.cpu().clone() for k, v in D.state_dict().items()},
    }

    # Build disc classifier: reuse disc conv features, new classification head
    from ..models.discriminator import Discriminator

    class DiscClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.feat = D.feat
            self.head = nn.Sequential(
                nn.Linear(64, 32), nn.GELU(), nn.Dropout(0.3), nn.Linear(32, 1))
            self.REP_DIM = 64
        def extract(self, x):
            return self.feat(x)
        def forward(self, x):
            return self.head(self.feat(x))

    print(f"    Finetuning disc classifier ({config['finetune_epochs']} epochs)...")
    dc = DiscClassifier()
    model = ExpansionModel(dc)
    model.expansion.load_state_dict(gan_state["expansion"])

    auc, preds, _ = train_loop(
        model, X_train, y_train, X_val, y_val, device,
        epochs=config["finetune_epochs"], lr=config["finetune_lr"])

    elapsed = time.time() - t0
    print(f"    Best AUC: {auc:.6f} ({elapsed/60:.1f} min total)")

    return model, gan_state, auc, preds, elapsed


def train_gan_aug(backbone, generator, expansion_state, X_all, y_all,
                  X_train, y_train, X_val, y_val, device, config):
    """GAN-augmented: use trained generator to augment training data, then train CNN.

    Requires a pre-trained generator and expansion state from train_cgan.
    Returns (model, comp_auc, comp_preds, elapsed).
    """
    expansion = ExpansionLayer().to(device)
    expansion.load_state_dict(expansion_state)
    G = generator.to(device)

    # Generate synthetic images (20% of training set)
    n_gen = int(len(X_train) * config["aug_ratio"])
    print(f"    Generating {n_gen} synthetic images...")
    with torch.no_grad():
        z = torch.randn(n_gen, config.get("latent_dim", 64), device=device)
        lab = torch.randint(0, 2, (n_gen,), device=device).float()
        fake = G(z, lab).cpu().numpy()

    # Convert real training data to images
    with torch.no_grad():
        real_imgs = []
        for i in range(0, len(X_train), BATCH_SIZE):
            chunk = torch.tensor(X_train[i:i+BATCH_SIZE], dtype=torch.float32).to(device)
            real_imgs.append(expansion(chunk).cpu().numpy())
        real_imgs = np.concatenate(real_imgs)

    # Combine real + synthetic
    X_aug = np.concatenate([real_imgs, fake])
    y_aug = np.concatenate([y_train, lab.cpu().numpy()])
    perm = np.random.RandomState(42).permutation(len(X_aug))
    X_aug, y_aug = X_aug[perm], y_aug[perm]

    # Convert val to images too
    with torch.no_grad():
        val_imgs = []
        for i in range(0, len(X_val), BATCH_SIZE):
            chunk = torch.tensor(X_val[i:i+BATCH_SIZE], dtype=torch.float32).to(device)
            val_imgs.append(expansion(chunk).cpu().numpy())
        val_imgs = np.concatenate(val_imgs)

    print(f"    Training on augmented data ({len(X_aug)} samples, {config['train_epochs']} epochs)...")
    t0 = time.time()

    # Train backbone directly on images (no expansion layer — images are pre-computed)
    backbone = backbone.to(device)
    auc, preds, state = train_loop(
        backbone, X_aug, y_aug, val_imgs, y_val, device,
        epochs=config["train_epochs"], lr=config["lr"])

    # Wrap in expansion model for consistent evaluation interface
    model = ExpansionModel(backbone)
    model.expansion.load_state_dict(expansion_state)
    model = model.to(device)

    elapsed = time.time() - t0
    print(f"    Best AUC: {auc:.6f} ({elapsed/60:.1f} min)")

    return model, auc, preds, elapsed
