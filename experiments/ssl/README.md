# Self-Supervised Pretraining

SSL pretraining on the 630K synthetic competition dataset, with downstream evaluation on both competition and UCI clinical data.

## Methods

**SCARF** — Corrupts random features and trains the encoder to distinguish original from corrupted views via contrastive loss. Learns which feature patterns are clinically meaningful.

**MAE** — Masks 30% of features and trains a decoder to reconstruct them. Learns feature dependencies across the 13 clinical dimensions.

**SemiMAE** — Extends MAE by including the target label as a 14th input dimension. Two masking variants:
- `semi-random`: label masked at the same 30% rate as features
- `semi-always`: label always masked (100%), features at 30%

At inference the label is replaced with a learned mask token, so the decoder predicts diagnosis from feature patterns without ever seeing the true label.

## Structure

```
experiments/ssl/
├── models.py       # SCARF, MAE, SemiMAE architectures
├── data.py         # Data loading (shared from code/data/)
├── pretrain.py     # Training script
├── evaluate.py     # Downstream evaluation
└── results/
    ├── checkpoints/    # Pretrained model weights (.pt)
    ├── metrics/        # JSON results with per-fold scores
    └── figures/        # Comparison plots
```

## Usage

All commands run from the `code/` directory:

```bash
# Pretrain (optional — checkpoints already included)
python -m experiments.ssl.pretrain --method scarf
python -m experiments.ssl.pretrain --method mae
python -m experiments.ssl.pretrain --method semi-random
python -m experiments.ssl.pretrain --method semi-always

# Evaluate all pretrained models
python -m experiments.ssl.evaluate

# Evaluate specific methods
python -m experiments.ssl.evaluate --methods mae scarf

# UCI transfer only (skip competition evaluation)
python -m experiments.ssl.evaluate --skip-competition
```

## Results

On the competition data, SSL representations provide marginal gains over raw features — tree-based models already capture the same patterns.

On UCI transfer, SemiMAE-always decoder achieves 0.928 AUC on Cleveland (LogReg downstream), exceeding both the best within-dataset single model (0.916) and ensemble (0.920). An ablation shows the gain comes from the decoder mechanism rather than encoder quality.

Full per-fold results are in `results/metrics/`.
