#!/usr/bin/env python3
"""ViT attention analysis mapped back to the 13 clinical features.

The v12 ViT operates on 11x11 images produced by a learned expansion layer
(Linear: 13 -> 121). Each pixel is a weighted combination of all 13 clinical
features, so raw pixel-level attention doesn't map 1:1 to clinical features.

To recover clinical-feature-level attention, we use the expansion layer's
weight matrix to attribute each pixel's attention contribution back to the
source features. Specifically, for each pixel p, the expansion weight
|W[p, f]| tells us how much clinical feature f contributes to that pixel.
Normalizing these weights row-wise gives a pixel-to-feature attribution
matrix A[p, f]. Feature-level attention is then the weighted sum of
pixel-level attention using these attributions.

This produces:
  - A 13x13 mutual feature attention heatmap
  - Feature importance ranking from attention (compared with SHAP)
  - Class-conditional attention patterns (disease vs healthy)
  - Saved numpy arrays and a figure for the paper

Usage:
    python -m experiments.tabular_to_image_raw13.evaluation.vit_attention
"""

import argparse
import json
import os

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from ..config import CHECKPOINTS_DIR, METRICS_DIR
from ..data import load_competition
from ..data.loader import FEATURE_NAMES
from ..models import ExpansionModel, VisionTransformer
from ..training.utils import get_device

# Short labels for heatmap axes, matching the 13 UCI Heart Disease features
CLINICAL_LABELS = [
    "Age", "Sex", "Chest Pain", "Rest BP", "Cholesterol", "Fasting BS",
    "Rest ECG", "Max HR", "Exercise Angina", "ST Depression", "Slope",
    "Num Vessels", "Thal",
]

BATCH_SIZE = 512


def _build_pixel_to_feature_attribution(expansion_layer):
    """Build a pixel-to-feature attribution matrix from the expansion weights.

    The expansion layer is Linear(13 -> 121) + BatchNorm + GELU. We use the
    absolute values of the linear weights to measure how much each clinical
    feature contributes to each pixel. Row-normalizing gives proportions.

    Returns: attribution matrix of shape (121, 13), each row sums to 1.
    """
    linear = expansion_layer.net[0]
    W = linear.weight.detach().cpu().numpy()  # (121, 13)
    W_abs = np.abs(W)

    row_sums = W_abs.sum(axis=1, keepdims=True)
    row_sums = np.maximum(row_sums, 1e-10)
    return W_abs / row_sums  # (121, 13)


def _extract_attention_and_compute(model, X, attribution, device,
                                   max_samples=5000):
    """Extract ViT attention and compute feature-level results incrementally.

    Rather than storing all pixel-to-pixel attention matrices (which would
    require ~12GB for 5000 samples), we compute the feature-to-feature
    product A^T @ pixel_attn @ A per batch and accumulate the result.

    Returns:
        feat_attn: (13, 13) mutual feature attention matrix
        feature_importance: (13,) CLS-to-feature importance
        predictions: (N,) sigmoid probabilities
        cls_feat_per_sample: (N, 13) per-sample CLS-to-feature attention
    """
    model.eval()
    vit = model.backbone
    n_features = attribution.shape[1]
    A = attribution.T  # (13, 121) for the matrix product

    # Subsample if dataset is too large
    if len(X) > max_samples:
        rng = np.random.RandomState(42)
        idx = rng.choice(len(X), max_samples, replace=False)
        X = X[idx]

    # Hook the self-attention modules to capture per-head attention weights.
    # PyTorch's TransformerEncoderLayer has a fused fast path that bypasses
    # self.self_attn() in eval mode. Registering a forward hook on any
    # submodule disables the fast path, so our patched forward gets called.
    attention_storage = {}
    saved_forwards = []
    hook_handles = []

    for layer_idx, layer in enumerate(vit.encoder.layers):
        storage = []
        attention_storage[layer_idx] = storage
        original_forward = layer.self_attn.forward

        def make_hooked_forward(orig, store):
            def hooked_forward(*args, **kwargs):
                kwargs["need_weights"] = True
                kwargs["average_attn_weights"] = False
                out = orig(*args, **kwargs)
                # out = (attn_output, attn_weights) with shape (B, heads, seq, seq)
                store.append(out[1].detach().cpu().numpy())
                return out
            return hooked_forward

        layer.self_attn.forward = make_hooked_forward(original_forward, storage)
        saved_forwards.append((layer.self_attn, original_forward))

        # Register a no-op hook to disable the fused fast path
        handle = layer.self_attn.register_forward_hook(lambda m, i, o: None)
        hook_handles.append(handle)

    # Accumulators for incremental computation
    feat_attn_sum = np.zeros((n_features, n_features))
    cls_feat_sum = np.zeros(n_features)
    all_predictions = []
    all_cls_feat = []  # per-sample CLS->feature attention for class-conditional
    total_samples = 0

    with torch.no_grad():
        for batch_start in range(0, len(X), BATCH_SIZE):
            batch_end = min(batch_start + BATCH_SIZE, len(X))
            batch_x = torch.tensor(
                X[batch_start:batch_end], dtype=torch.float32).to(device)

            # Forward pass fills attention_storage via hooks
            logits = model(batch_x).squeeze(-1)
            preds = torch.sigmoid(logits).cpu().numpy()
            all_predictions.append(preds)
            B = len(preds)

            # Collect attention from all layers for this batch
            # Each entry in storage: (B, n_heads, 122, 122)
            batch_cls_to_pixel = []
            batch_pix_to_pix = []

            for layer_idx in range(len(vit.encoder.layers)):
                attn = attention_storage[layer_idx].pop(0)
                # CLS (position 0) attending to pixels (positions 1:122)
                batch_cls_to_pixel.append(attn[:, :, 0, 1:])  # (B, heads, 121)
                # Pixel-to-pixel attention (skip CLS row and column)
                batch_pix_to_pix.append(attn[:, :, 1:, 1:])  # (B, heads, 121, 121)

            # Average CLS-to-pixel over layers and heads: (B, 121)
            cls_pixel = np.stack(batch_cls_to_pixel).mean(axis=(0, 2))
            # Map to features: (B, 121) @ (121, 13) = (B, 13)
            cls_feat_batch = cls_pixel @ attribution
            all_cls_feat.append(cls_feat_batch)
            cls_feat_sum += cls_feat_batch.sum(axis=0)

            # Average pixel-to-pixel over layers and heads: (B, 121, 121)
            pix_attn = np.stack(batch_pix_to_pix).mean(axis=(0, 2))

            # Accumulate feature-to-feature: A @ pix_attn[i] @ A^T for each sample
            for i in range(B):
                feat_attn_sum += A @ pix_attn[i] @ A.T

            total_samples += B

            if (batch_start // BATCH_SIZE + 1) % 20 == 0:
                print(f"    Processed {total_samples}/{len(X)} samples")

    # Restore original forwards and remove hooks
    for handle in hook_handles:
        handle.remove()
    for self_attn, orig_fwd in saved_forwards:
        self_attn.forward = orig_fwd

    feat_attn = feat_attn_sum / total_samples
    feature_importance = cls_feat_sum / total_samples
    predictions = np.concatenate(all_predictions)
    cls_feat_per_sample = np.concatenate(all_cls_feat)

    return feat_attn, feature_importance, predictions, cls_feat_per_sample


def _class_conditional_attention(cls_feat_per_sample, predictions, threshold=0.5):
    """Split per-sample feature attention by predicted class.

    Returns importance vectors for disease and healthy predictions.
    """
    disease_mask = predictions > threshold
    healthy_mask = ~disease_mask

    disease_feat = (cls_feat_per_sample[disease_mask].mean(axis=0)
                    if disease_mask.sum() > 0 else np.zeros(13))
    healthy_feat = (cls_feat_per_sample[healthy_mask].mean(axis=0)
                    if healthy_mask.sum() > 0 else np.zeros(13))

    return disease_feat, healthy_feat


def _plot_heatmap(feat_attn, feature_importance, disease_feat, healthy_feat,
                  output_dir):
    """Generate and save attention analysis figures."""
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # Panel 1: feature-to-feature mutual attention
    ax = axes[0]
    sns.heatmap(feat_attn, xticklabels=CLINICAL_LABELS, yticklabels=CLINICAL_LABELS,
                cmap="YlOrRd", annot=False, ax=ax,
                square=True, cbar_kws={"shrink": 0.8})
    ax.set_title("Mutual Feature Attention (ViT)")
    ax.tick_params(axis="x", rotation=45)
    ax.tick_params(axis="y", rotation=0)

    # Panel 2: feature importance ranking
    ax = axes[1]
    order = np.argsort(feature_importance)[::-1]
    ax.barh(range(13), feature_importance[order], color="steelblue")
    ax.set_yticks(range(13))
    ax.set_yticklabels([CLINICAL_LABELS[i] for i in order])
    ax.set_xlabel("Mean CLS Attention (attributed)")
    ax.set_title("Feature Importance from ViT Attention")
    ax.invert_yaxis()

    # Panel 3: class-conditional attention difference
    ax = axes[2]
    diff = disease_feat - healthy_feat
    order_diff = np.argsort(np.abs(diff))[::-1]
    colors = ["#d62728" if d > 0 else "#2ca02c" for d in diff[order_diff]]
    ax.barh(range(13), diff[order_diff], color=colors)
    ax.set_yticks(range(13))
    ax.set_yticklabels([CLINICAL_LABELS[i] for i in order_diff])
    ax.set_xlabel("Attention Difference (Disease − Healthy)")
    ax.set_title("Class-Conditional Attention Shift")
    ax.invert_yaxis()
    ax.axvline(0, color="black", linewidth=0.5)

    plt.tight_layout()
    path = os.path.join(output_dir, "vit_clinical_feature_attention_v12.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Figure saved: {path}")

    # Standalone heatmap for the paper (single panel, with numeric annotations)
    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(feat_attn, xticklabels=CLINICAL_LABELS, yticklabels=CLINICAL_LABELS,
                cmap="YlOrRd", annot=True, fmt=".3f", ax=ax,
                square=True, cbar_kws={"shrink": 0.8})
    ax.set_title("ViT Clinical Feature Attention (raw 13 features)")
    ax.tick_params(axis="x", rotation=45)
    ax.tick_params(axis="y", rotation=0)
    plt.tight_layout()
    path2 = os.path.join(output_dir, "vit_attention_heatmap_paper.png")
    plt.savefig(path2, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Paper figure saved: {path2}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default=None, choices=["cpu", "mps"])
    parser.add_argument("--max-samples", type=int, default=5000,
                        help="Max samples for attention extraction")
    args = parser.parse_args()

    device = get_device(args.device)
    print(f"Device: {device}")

    # Load trained ViT
    ckpt_path = CHECKPOINTS_DIR / "vit" / "best.pt"
    if not ckpt_path.exists():
        print(f"ERROR: ViT checkpoint not found at {ckpt_path}")
        print("Train first: python -m experiments.tabular_to_image_raw13.run --model vit")
        return

    model = ExpansionModel(VisionTransformer())
    state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model = model.to(device)
    model.eval()
    print("Loaded ViT checkpoint")

    # Build pixel-to-feature attribution from the expansion layer
    attribution = _build_pixel_to_feature_attribution(model.expansion)
    print(f"Attribution matrix: {attribution.shape}")
    print(f"  Top contributing feature per pixel (first 10): "
          f"{[CLINICAL_LABELS[i] for i in attribution[:10].argmax(axis=1)]}")

    # Load competition data
    print("\nLoading competition data...")
    X_train, X_val, y_train, y_val, X_all, y_all, scaler = load_competition()
    n_eval = min(len(X_val), args.max_samples)

    # Extract attention and compute feature-level results (incremental, low memory)
    print(f"\nExtracting attention and computing feature attribution ({n_eval} samples)...")
    feat_attn, feature_importance, predictions, cls_feat = (
        _extract_attention_and_compute(model, X_val, attribution, device,
                                       max_samples=args.max_samples))

    n_disease = int((predictions > 0.5).sum())
    print(f"  Samples: {len(predictions)} ({n_disease} disease, "
          f"{len(predictions) - n_disease} healthy)")

    # Feature importance ranking
    print("\n  Feature importance (CLS attention → clinical features):")
    order = np.argsort(feature_importance)[::-1]
    for rank, idx in enumerate(order):
        print(f"    {rank+1}. {CLINICAL_LABELS[idx]:20s} {feature_importance[idx]:.4f}")

    # Class-conditional analysis
    print("\nClass-conditional attention...")
    disease_feat, healthy_feat = _class_conditional_attention(cls_feat, predictions)

    diff = disease_feat - healthy_feat
    order_diff = np.argsort(np.abs(diff))[::-1]
    print("\n  Disease vs Healthy attention difference:")
    for idx in order_diff:
        direction = "Disease+" if diff[idx] > 0 else "Healthy+"
        print(f"    {CLINICAL_LABELS[idx]:20s} {diff[idx]:+.4f} ({direction})")

    # Save results
    output_dir = str(METRICS_DIR / "vit_attention")
    os.makedirs(output_dir, exist_ok=True)

    results = {
        "method": "expansion_weighted_attention",
        "description": (
            "ViT attention mapped to clinical features via the expansion layer's "
            "weight matrix. Each pixel's attention is attributed back to the 13 "
            "clinical features proportionally to the absolute expansion weights."
        ),
        "n_samples": int(len(predictions)),
        "n_disease": n_disease,
        "n_healthy": int(len(predictions) - n_disease),
        "feature_names": FEATURE_NAMES,
        "feature_labels": CLINICAL_LABELS,
        "feature_importance_ranking": [
            {"rank": r+1, "feature": CLINICAL_LABELS[i],
             "attention": float(feature_importance[i])}
            for r, i in enumerate(order)
        ],
        "class_conditional_difference": [
            {"feature": CLINICAL_LABELS[i],
             "disease_attention": float(disease_feat[i]),
             "healthy_attention": float(healthy_feat[i]),
             "difference": float(diff[i])}
            for i in range(13)
        ],
    }

    json_path = os.path.join(output_dir, "vit_attention_analysis.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved: {json_path}")

    np.save(os.path.join(output_dir, "feature_attention_matrix.npy"), feat_attn)
    np.save(os.path.join(output_dir, "feature_importance.npy"), feature_importance)
    np.save(os.path.join(output_dir, "attribution_matrix.npy"), attribution)
    np.save(os.path.join(output_dir, "disease_attention.npy"), disease_feat)
    np.save(os.path.join(output_dir, "healthy_attention.npy"), healthy_feat)
    print("  Numpy arrays saved")

    # Generate figures
    print("\nGenerating figures...")
    figures_dir = str(METRICS_DIR / "vit_attention" / "figures")
    _plot_heatmap(feat_attn, feature_importance, disease_feat, healthy_feat,
                  figures_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
