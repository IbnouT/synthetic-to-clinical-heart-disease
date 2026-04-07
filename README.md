# Can Synthetic Data Train Clinical Models?

Self-supervised transfer, benchmark analysis, and representation robustness for heart disease prediction.

**Course:** SYS5185 — Winter 2026, University of Ottawa

**Authors:** Aladji Ibnou A M Alia Dit Papa Toure, Ayegbe Jean-Noel Djahoua, Chadi El Hamdouchi

## Overview

We investigate whether representations learned from a large synthetic dataset (~630K samples, Kaggle Playground Series S6E2) can improve heart disease prediction on small UCI clinical targets (Cleveland n=303, Hungarian n=294). Three modeling approaches are developed on the synthetic source domain and evaluated on both the benchmark and clinical data:

1. **Competition pipeline** — 14 model families with progressive feature engineering and a 6-step ensemble chain. Achieves 0.95535 AUC on the Kaggle private leaderboard.

2. **Self-supervised pretraining (SemiMAE)** — SCARF, MAE, and our proposed Semi-Supervised Masked Autoencoder, which treats diagnosis as a masked feature to be reconstructed. The SemiMAE decoder achieves 0.928 AUC on Cleveland without fine-tuning on target labels, exceeding both the best within-dataset single model (0.916) and ensemble (0.920).

3. **Tabular-to-image projection** — 8 image architectures (DeeperCNN, ViT, Hybrid, SimCLR, MoCo, VAE, AC-GAN, GAN-Aug) trained on learned image representations. ViT attention recovers the same top features as SHAP on tabular models.

The strong Cleveland results do not replicate on Hungarian, suggesting that synthetic-to-clinical transfer depends on distributional similarity between source and target data.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Requires Python 3.10+. GPU is optional but speeds up PyTorch-based models (MLP, TabPFN, RealMLP, SSL, tabular-to-image).

## Project Structure

```
├── src/                        # Core library
│   ├── config.py               # Paths, column names, model configurations
│   ├── data.py                 # Competition data loading and fold creation
│   ├── data_external.py        # UCI dataset loading and preprocessing
│   ├── metrics.py              # AUC, accuracy, precision, recall, F1
│   ├── training.py             # K-fold CV training loop
│   ├── training_stacked.py     # Per-fold CB+XGB+RF stacking
│   ├── utils.py                # Save/load predictions and metrics
│   ├── features/               # Feature pipeline modules
│   │   ├── basic.py            # Raw 13 features, one-hot, integer-coded
│   │   ├── enriched.py         # Frequency encoding + UCI target means
│   │   ├── origstats.py        # UCI per-value statistics (mean, median, std, count)
│   │   ├── top_pipe.py         # Target encoding + correlation interactions
│   │   ├── pairproto.py        # Pairwise target stats + prototype distances
│   │   ├── stacking.py         # Cross-stacking base features
│   │   └── helpers.py          # Shared encoding primitives
│   ├── models/                 # Model family implementations
│   │   ├── registry.py         # Unified interface for all model families
│   │   ├── boosting.py         # CatBoost, XGBoost, LightGBM
│   │   ├── trees.py            # Random Forest, Extra Trees
│   │   ├── linear.py           # Logistic Regression, Ridge, SVM
│   │   ├── neighbors.py        # KNN
│   │   ├── neural.py           # PyTorch MLP
│   │   └── tabular_dl.py       # RealMLP, TabPFN
│   └── ensemble/               # Ensemble blending
│       ├── blend.py            # Rank blend, hill-climbing, band-gating
│       └── definitions.py      # Ensemble configurations
│
├── experiments/
│   ├── train_models.py         # Train all competition single models
│   ├── build_ensembles.py      # Build 6-step ensemble chain
│   ├── zero_shot_transfer.py   # Train on competition data, predict UCI
│   ├── within_dataset_cv.py    # Within-dataset CV on UCI (with Optuna tuning)
│   ├── ensemble_transfer.py    # Multi-family ensemble on UCI
│   ├── external/               # Shared utilities for external validation
│   ├── ssl/                    # Self-supervised pretraining (SCARF, MAE, SemiMAE)
│   └── tabular_to_image_raw13/ # Image-based architectures
│
├── notebooks/                  # Analysis and visualization
│   ├── 01_eda.ipynb            # Exploratory data analysis
│   ├── 02_competition_results.ipynb
│   ├── 03_ablation.ipynb       # 6-step ensemble ablation
│   ├── 04_calibration.ipynb    # Reliability diagrams
│   ├── 05_external_validation.ipynb
│   ├── 06_shap.ipynb           # Feature importance analysis
│   └── 07_distribution_shift.ipynb
│
├── data/
│   ├── competition/            # train.csv, test.csv, original.csv
│   ├── external/               # UCI Heart Disease subsets (Cleveland, Hungarian, etc.)
│   └── folds/                  # Precomputed fold indices for reproducibility
│
├── results/
│   ├── oof/                    # Out-of-fold predictions (competition)
│   ├── test_preds/             # Test predictions (competition)
│   ├── metrics/                # Per-model metric JSONs
│   ├── external/               # External validation results
│   ├── figures/                # Generated plots
│   └── tabular_to_image_raw13/ # Image experiment results
│
├── requirements.txt
└── REFERENCES.md               # Attribution for adapted code
```

## Reproduction

### Competition Pipeline

Competition models were trained during the Kaggle competition. The precomputed OOF predictions are stored in `results/oof/` and can be used directly to build ensembles:

```bash
python -m experiments.build_ensembles
python -m experiments.build_ensembles --verify-only   # check AUCs without overwriting
```

To retrain individual models:

```bash
python -m experiments.train_models --model 02_cb_raw
```

### External Validation (UCI)

Three evaluation scenarios on the UCI clinical datasets:

```bash
# Zero-shot transfer: train on 630K competition data, predict UCI
python -m experiments.zero_shot_transfer

# Within-dataset CV: train and evaluate directly on each UCI subset
python -m experiments.within_dataset_cv
python -m experiments.within_dataset_cv --resume      # skip completed evaluations

# Multi-family ensemble on UCI
python -m experiments.ensemble_transfer
```

### Self-Supervised Pretraining

SCARF, MAE, and SemiMAE pretrained on the 630K competition dataset, then evaluated on UCI transfer. Pretrained checkpoints are included in `experiments/ssl/results/checkpoints/`.

```bash
# Pretrain from scratch (optional — checkpoints already included)
python -m experiments.ssl.pretrain --method scarf
python -m experiments.ssl.pretrain --method mae
python -m experiments.ssl.pretrain --method semi-random
python -m experiments.ssl.pretrain --method semi-always

# Evaluate all pretrained models on competition + UCI data
python -m experiments.ssl.evaluate
```

See [`experiments/ssl/README.md`](experiments/ssl/README.md) for method details.

### Tabular-to-Image

Eight image architectures trained on learned 11x11 image representations of the 13 clinical features:

```bash
# Train a specific architecture
python -m experiments.tabular_to_image_raw13.run --model deepercnn
python -m experiments.tabular_to_image_raw13.run --model vit

# Evaluate all trained models (competition + UCI transfer)
python -m experiments.tabular_to_image_raw13.evaluate_all

# ViT attention ablation
python -m experiments.tabular_to_image_raw13.vit_ablation
```

Architectures: DeeperCNN, ViT, Hybrid CNN-Transformer, SimCLR, MoCo, VAE, AC-GAN, GAN-Aug.

## Data Sources

- **Competition**: [Kaggle Playground Series S6E2](https://www.kaggle.com/competitions/playground-series-s6e2) — ~630K synthetic heart disease samples generated via CTGAN from UCI Cleveland
- **External**: [UCI Heart Disease](https://archive.ics.uci.edu/dataset/45/heart+disease) — Cleveland, Hungarian, Switzerland, VA Long Beach
