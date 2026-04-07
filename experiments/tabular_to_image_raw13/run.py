#!/usr/bin/env python3
"""Run raw-13 feature tabular-to-image experiments.

Usage:
    python -m experiments.tabular_to_image_raw13.run --model deepercnn
    python -m experiments.tabular_to_image_raw13.run --model simclr --device cpu
"""

import argparse
import json
import os
import time

import torch

from .config import MODEL_CONFIGS, CHECKPOINTS_DIR, METRICS_DIR
from .data import load_competition, load_uci
from .training import get_device, train_supervised, train_simclr, train_moco
from .training.generative import train_vae, train_cgan, train_gan_aug
from .models import (DeeperCNN, VisionTransformer, HybridCNNTransformer,
                     VAEModel, Generator, Discriminator, ExpansionModel)
from .evaluation import evaluate_zeroshot, evaluate_rep_transfer, compute_metrics


def save_results(name, data):
    os.makedirs(METRICS_DIR, exist_ok=True)
    path = METRICS_DIR / f"{name}.json"
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  Saved: {path}")


def save_checkpoint(name, state, suffix="best"):
    d = CHECKPOINTS_DIR / name
    os.makedirs(d, exist_ok=True)
    torch.save(state, d / f"{suffix}.pt")
    print(f"  Checkpoint: {d}/{suffix}.pt")


def _eval_and_save(name, model, comp_preds, y_val, uci, device, elapsed,
                   extra_meta=None, pretrain_state=None):
    """Shared evaluation + save logic for all models."""
    print("\n  Competition:")
    comp = compute_metrics(y_val, comp_preds)
    print(f"    AUC={comp['auc']:.6f}")

    print("\n  UCI zero-shot:")
    zs = evaluate_zeroshot(model, uci, device)

    print("\n  UCI rep transfer:")
    rt = evaluate_rep_transfer(model, uci, device)

    results = {
        "model": name,
        "training_time_sec": elapsed,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "competition": comp,
        "uci_zeroshot": zs,
        "uci_rep_transfer": rt,
    }
    if extra_meta:
        results.update(extra_meta)

    save_checkpoint(name, model.state_dict())
    if pretrain_state:
        save_checkpoint(name, pretrain_state, suffix="pretrained")
    save_results(name, results)


# ===================== SUPERVISED =====================

def run_deepercnn(X_tr, X_va, y_tr, y_va, X_all, y_all, uci, device):
    print(f"\n{'='*60}\n  DeeperCNN (supervised)\n{'='*60}")
    cfg = MODEL_CONFIGS["deepercnn"]
    backbone = DeeperCNN(dropout=cfg["dropout"])
    model, auc, preds, elapsed = train_supervised(backbone, X_tr, y_tr, X_va, y_va, device, cfg)
    _eval_and_save("deepercnn", model, preds, y_va, uci, device, elapsed)


def run_vit(X_tr, X_va, y_tr, y_va, X_all, y_all, uci, device):
    print(f"\n{'='*60}\n  ViT (supervised)\n{'='*60}")
    cfg = MODEL_CONFIGS["vit"]
    backbone = VisionTransformer(
        embed_dim=cfg["embed_dim"], n_heads=cfg["n_heads"],
        n_layers=cfg["n_layers"], dropout=cfg["dropout"])
    model, auc, preds, elapsed = train_supervised(backbone, X_tr, y_tr, X_va, y_va, device, cfg)
    _eval_and_save("vit", model, preds, y_va, uci, device, elapsed)


def run_hybrid(X_tr, X_va, y_tr, y_va, X_all, y_all, uci, device):
    print(f"\n{'='*60}\n  Hybrid CNN-Transformer (supervised)\n{'='*60}")
    cfg = MODEL_CONFIGS["hybrid"]
    backbone = HybridCNNTransformer(
        cnn_ch=cfg["cnn_ch"], n_heads=cfg["n_heads"],
        n_layers=cfg["n_layers"], dropout=cfg["dropout"])
    model, auc, preds, elapsed = train_supervised(backbone, X_tr, y_tr, X_va, y_va, device, cfg)
    _eval_and_save("hybrid", model, preds, y_va, uci, device, elapsed)


# ===================== SSL =====================

def run_simclr(X_tr, X_va, y_tr, y_va, X_all, y_all, uci, device):
    print(f"\n{'='*60}\n  SimCLR (contrastive SSL)\n{'='*60}")
    cfg = MODEL_CONFIGS["simclr"]
    backbone = DeeperCNN()
    model, ps, auc, preds, elapsed = train_simclr(
        backbone, X_all, y_all, X_tr, y_tr, X_va, y_va, device, cfg)
    _eval_and_save("simclr", model, preds, y_va, uci, device, elapsed,
                   pretrain_state=ps)


def run_moco(X_tr, X_va, y_tr, y_va, X_all, y_all, uci, device):
    print(f"\n{'='*60}\n  MoCo (momentum contrastive SSL)\n{'='*60}")
    cfg = MODEL_CONFIGS["moco"]
    model, ps, auc, preds, elapsed = train_moco(
        lambda: DeeperCNN(), X_all, y_all, X_tr, y_tr, X_va, y_va, device, cfg)
    _eval_and_save("moco", model, preds, y_va, uci, device, elapsed,
                   pretrain_state=ps)


def run_vae(X_tr, X_va, y_tr, y_va, X_all, y_all, uci, device):
    print(f"\n{'='*60}\n  VAE (reconstruction SSL)\n{'='*60}")
    cfg = MODEL_CONFIGS["vae"]
    vae = VAEModel(latent_dim=cfg["latent_dim"])
    model, ps, auc, preds, elapsed = train_vae(
        vae, X_all, y_all, X_tr, y_tr, X_va, y_va, device, cfg)
    _eval_and_save("vae", model, preds, y_va, uci, device, elapsed,
                   pretrain_state=ps)


# ===================== GENERATIVE =====================

def run_cgan(X_tr, X_va, y_tr, y_va, X_all, y_all, uci, device):
    print(f"\n{'='*60}\n  cGAN Discriminator\n{'='*60}")
    cfg = MODEL_CONFIGS["cgan"]
    G = Generator(latent_dim=cfg["latent_dim"])
    D = Discriminator()
    model, gs, auc, preds, elapsed = train_cgan(
        G, D, X_all, y_all, X_tr, y_tr, X_va, y_va, device, cfg)
    _eval_and_save("cgan", model, preds, y_va, uci, device, elapsed,
                   pretrain_state=gs)


def run_gan_aug(X_tr, X_va, y_tr, y_va, X_all, y_all, uci, device):
    """GAN-augmented CNN. Requires cGAN to be trained first (uses its generator)."""
    print(f"\n{'='*60}\n  GAN-Augmented DeeperCNN\n{'='*60}")
    cfg = MODEL_CONFIGS["gan_aug"]

    # Load cGAN generator and expansion from saved checkpoint
    cgan_ckpt = CHECKPOINTS_DIR / "cgan" / "pretrained.pt"
    if not cgan_ckpt.exists():
        print("  ERROR: cGAN checkpoint not found. Run --model cgan first.")
        return

    gs = torch.load(cgan_ckpt, map_location="cpu", weights_only=True)
    G = Generator(latent_dim=cfg["latent_dim"])
    G.load_state_dict(gs["generator"])

    backbone = DeeperCNN()
    model, auc, preds, elapsed = train_gan_aug(
        backbone, G, gs["expansion"], X_all, y_all,
        X_tr, y_tr, X_va, y_va, device, cfg)
    _eval_and_save("gan_aug", model, preds, y_va, uci, device, elapsed)


# ===================== MAIN =====================

RUNNERS = {
    "deepercnn": run_deepercnn,
    "vit": run_vit,
    "hybrid": run_hybrid,
    "simclr": run_simclr,
    "moco": run_moco,
    "vae": run_vae,
    "cgan": run_cgan,
    "gan_aug": run_gan_aug,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=list(RUNNERS.keys()))
    parser.add_argument("--device", default=None, choices=["cpu", "mps"])
    args = parser.parse_args()

    device = get_device(args.device)
    print(f"Device: {device}")

    print("\nLoading data...")
    X_tr, X_va, y_tr, y_va, X_all, y_all, scaler = load_competition()
    uci = load_uci(scaler)

    RUNNERS[args.model](X_tr, X_va, y_tr, y_va, X_all, y_all, uci, device)
    print("\nDone.")


if __name__ == "__main__":
    main()
