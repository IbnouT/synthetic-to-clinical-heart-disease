#!/usr/bin/env python3
"""ViT architecture ablation study.

Tests how ViT performance on tabular-to-image heart disease prediction
varies with architectural choices: depth, width, attention heads, and
positional encoding. All variants share the same learned expansion
layer (13 features -> 11x11 image) and training protocol.

The baseline is the default ViT: 4 layers, 4 heads, 64-dim, dropout 0.1.

Ablation axes:
    1. Depth:     n_layers in {1, 2, 4, 6}
    2. Width:     embed_dim in {32, 64, 128}
    3. Heads:     n_heads in {1, 2, 4, 8}
    4. Position:  with vs without positional embeddings

Each variant trains end-to-end on 630K competition data and evaluates
on UCI Cleveland/Hungarian via zero-shot and representation transfer.

Usage:
    python -m experiments.tabular_to_image_raw13.vit_ablation
    python -m experiments.tabular_to_image_raw13.vit_ablation --subset depth
    python -m experiments.tabular_to_image_raw13.vit_ablation --subset position heads
    python -m experiments.tabular_to_image_raw13.vit_ablation --device cpu
"""

import argparse
import json
import os
import time

import torch
import torch.nn as nn

from .config import MODEL_CONFIGS, METRICS_DIR, CHECKPOINTS_DIR, IMG_SIZE
from .data import load_competition, load_uci
from .training import get_device, train_supervised
from .models.vit import PatchEmbed, N_PIXELS
from .models.expansion import ExpansionModel
from .evaluation import evaluate_zeroshot, evaluate_rep_transfer, compute_metrics


# Output directories for ablation results and model checkpoints
ABLATION_DIR = METRICS_DIR / "vit_ablation"
ABLATION_CKPT_DIR = CHECKPOINTS_DIR / "vit_ablation"


# ---------------------------------------------------------------------------
# ViT variant with optional positional encoding removal
# ---------------------------------------------------------------------------

class VisionTransformerAblation(nn.Module):
    """ViT variant that supports disabling positional embeddings.

    Identical to the standard VisionTransformer when use_pos_embed=True.
    When False, positional embeddings are zeroed out, testing whether
    spatial arrangement of the learned image pixels matters.
    """

    def __init__(self, embed_dim=64, n_heads=4, n_layers=4,
                 dropout=0.1, use_pos_embed=True):
        super().__init__()
        self.REP_DIM = embed_dim
        self.use_pos_embed = use_pos_embed

        # Token embedding: each pixel projected to embed_dim, plus CLS token
        self.patch_embed = PatchEmbed(embed_dim)

        # Zero out positional embeddings if disabled
        if not use_pos_embed:
            self.patch_embed.pos_embed.requires_grad = False
            nn.init.zeros_(self.patch_embed.pos_embed)

        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=n_heads, dim_feedforward=embed_dim * 4,
            dropout=dropout, batch_first=True, activation='gelu',
            norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(embed_dim)

        self.head = nn.Sequential(
            nn.Linear(embed_dim, 64), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(64, 1))

    def extract(self, x):
        """Backbone pass: returns CLS token as representation."""
        x = self.patch_embed(x)
        x = self.encoder(x)
        return self.norm(x[:, 0])

    def forward(self, x):
        return self.head(self.extract(x))


# ---------------------------------------------------------------------------
# Ablation configuration
# ---------------------------------------------------------------------------

# Each entry: (variant_name, ViT kwargs override from baseline)
# Baseline: embed_dim=64, n_heads=4, n_layers=4, dropout=0.1, use_pos_embed=True

ABLATION_VARIANTS = {
    "depth": [
        ("depth_1L", {"n_layers": 1}),
        ("depth_2L", {"n_layers": 2}),
        ("depth_6L", {"n_layers": 6}),
    ],
    "width": [
        ("width_32d", {"embed_dim": 32, "n_heads": 4}),
        ("width_128d", {"embed_dim": 128, "n_heads": 4}),
    ],
    "heads": [
        ("heads_1h", {"n_heads": 1}),
        ("heads_2h", {"n_heads": 2}),
        ("heads_8h", {"n_heads": 8}),
    ],
    "position": [
        ("pos_without", {"use_pos_embed": False}),
    ],
}

# Baseline ViT hyperparameters (matching config.py)
BASELINE_VIT = {
    "embed_dim": 64,
    "n_heads": 4,
    "n_layers": 4,
    "dropout": 0.1,
    "use_pos_embed": True,
}


def run_variant(name, vit_kwargs, X_tr, X_va, y_tr, y_va, uci, device):
    """Train one ViT variant and return all metrics."""
    params = {**BASELINE_VIT, **vit_kwargs}
    cfg = MODEL_CONFIGS["vit"]

    # n_heads must divide embed_dim
    if params["embed_dim"] % params["n_heads"] != 0:
        params["n_heads"] = min(params["n_heads"], params["embed_dim"])
        while params["embed_dim"] % params["n_heads"] != 0:
            params["n_heads"] -= 1

    backbone = VisionTransformerAblation(
        embed_dim=params["embed_dim"],
        n_heads=params["n_heads"],
        n_layers=params["n_layers"],
        dropout=params["dropout"],
        use_pos_embed=params["use_pos_embed"])

    n_params = sum(p.numel() for p in backbone.parameters())
    print(f"\n  {name}: {params} ({n_params:,} params)")

    t0 = time.time()
    model, auc, preds, elapsed = train_supervised(
        backbone, X_tr, y_tr, X_va, y_va, device, cfg)
    elapsed = time.time() - t0

    comp = compute_metrics(y_va, preds)
    print(f"    Competition AUC: {comp['auc']:.6f} ({elapsed/60:.1f} min)")

    # Save checkpoint (expansion + backbone weights)
    ckpt_dir = ABLATION_CKPT_DIR / name
    os.makedirs(ckpt_dir, exist_ok=True)
    torch.save(model.state_dict(), ckpt_dir / "best.pt")
    print(f"    Checkpoint: {ckpt_dir}/best.pt")

    zs = evaluate_zeroshot(model, uci, device)
    rt = evaluate_rep_transfer(model, uci, device)

    return {
        "variant": name,
        "params": params,
        "n_parameters": n_params,
        "training_time_sec": elapsed,
        "competition": comp,
        "uci_zeroshot": zs,
        "uci_rep_transfer": rt,
    }


def main():
    parser = argparse.ArgumentParser(
        description="ViT architecture ablation study.")
    parser.add_argument(
        "--subset", nargs="+",
        choices=list(ABLATION_VARIANTS.keys()),
        default=list(ABLATION_VARIANTS.keys()),
        help="Which ablation axes to run (default: all)")
    parser.add_argument(
        "--only", nargs="+", default=None,
        help="Run only these specific variant names (skip all others)")
    parser.add_argument(
        "--device", default=None, choices=["cpu", "mps"])
    args = parser.parse_args()

    device = get_device(args.device)
    print(f"Device: {device}")

    print("\nLoading data...")
    X_tr, X_va, y_tr, y_va, X_all, y_all, scaler = load_competition()
    uci = load_uci(scaler)

    os.makedirs(ABLATION_DIR, exist_ok=True)
    all_results = []

    for axis in args.subset:
        print(f"\n{'='*60}")
        print(f"  Ablation axis: {axis}")
        print(f"{'='*60}")

        axis_results = []
        for variant_name, overrides in ABLATION_VARIANTS[axis]:
            if args.only and variant_name not in args.only:
                print(f"  Skipping {variant_name} (not in --only)")
                continue
            result = run_variant(
                variant_name, overrides,
                X_tr, X_va, y_tr, y_va, uci, device)
            axis_results.append(result)
            all_results.append(result)

            # Save per-axis results after each variant (crash-safe)
            # Merge with any existing results to avoid overwriting prior runs
            axis_path = ABLATION_DIR / f"ablation_{axis}.json"
            existing = []
            if axis_path.exists():
                try:
                    existing = json.load(open(axis_path))
                except (json.JSONDecodeError, IOError):
                    existing = []
            # Replace entries with matching variant names, append new ones
            merged = {r["variant"]: r for r in existing}
            for r in axis_results:
                merged[r["variant"]] = r
            with open(axis_path, "w") as f:
                json.dump(list(merged.values()), f, indent=2)

        # Print axis summary
        print(f"\n  --- {axis} summary (Cleveland AUC) ---")
        for r in axis_results:
            clev_zs = r["uci_zeroshot"].get("Cleveland", {}).get("auc", 0)
            clev_rt = max(
                (c.get("AUC", {}).get("mean", 0)
                 for c in r["uci_rep_transfer"]
                 .get("Cleveland", {})
                 .get("classifiers", {}).values()),
                default=0)
            print(f"    {r['variant']:<25} comp={r['competition']['auc']:.4f}"
                  f"  zs={clev_zs:.4f}  rep={clev_rt:.4f}"
                  f"  ({r['n_parameters']:,} params)")

    # Save combined results (merge with existing)
    combined_path = ABLATION_DIR / "ablation_all.json"
    existing_all = []
    if combined_path.exists():
        try:
            existing_all = json.load(open(combined_path))
        except (json.JSONDecodeError, IOError):
            existing_all = []
    merged_all = {r["variant"]: r for r in existing_all}
    for r in all_results:
        merged_all[r["variant"]] = r
    with open(combined_path, "w") as f:
        json.dump(list(merged_all.values()), f, indent=2)
    print(f"\nSaved {len(all_results)} variants to {combined_path}")
    print("Done.")


if __name__ == "__main__":
    main()
