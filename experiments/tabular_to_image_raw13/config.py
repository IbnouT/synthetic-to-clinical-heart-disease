"""Configuration for raw-13 feature tabular-to-image experiments."""

from pathlib import Path

# Resolve paths relative to the code root (code/v12/code/)
CODE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = CODE_DIR / "data"
COMPETITION_CSV = DATA_DIR / "competition" / "train.csv"
UCI_DIR = DATA_DIR / "external"

# All outputs go under code/results/ per project convention
RESULTS_DIR = CODE_DIR / "results" / "tabular_to_image_raw13"
CHECKPOINTS_DIR = RESULTS_DIR / "checkpoints"
METRICS_DIR = RESULTS_DIR / "metrics"

# Image size: 13 raw features expanded to 121 = 11x11
IMG_SIZE = 11
EXPANSION_DIM = IMG_SIZE * IMG_SIZE
N_RAW_FEATURES = 13

# Training defaults
TRAIN_SPLIT = 0.2
RANDOM_SEED = 42
BATCH_SIZE = 2048

# Per-model hyperparameters
# Supervised models: epoch count is a ceiling, early stopping (patience=10) triggers first
# SSL/GAN pretrain: no early stopping, runs full epochs (convergence monitored via loss)
MODEL_CONFIGS = {
    "deepercnn": {
        "epochs": 100, "lr": 1e-3, "dropout": 0.3,
    },
    "vit": {
        "epochs": 100, "lr": 1e-3,
        "embed_dim": 64, "n_heads": 4, "n_layers": 4, "dropout": 0.1,
    },
    "hybrid": {
        "epochs": 100, "lr": 1e-3,
        "cnn_ch": 64, "n_heads": 4, "n_layers": 3, "dropout": 0.15,
    },
    "simclr": {
        "pretrain_epochs": 100, "finetune_epochs": 50,
        "pretrain_lr": 3e-4, "finetune_lr": 5e-4,
    },
    "moco": {
        "pretrain_epochs": 100, "finetune_epochs": 50,
        "pretrain_lr": 3e-4, "finetune_lr": 5e-4,
        "momentum": 0.999, "queue_size": 8192,
    },
    "vae": {
        "pretrain_epochs": 100, "finetune_epochs": 50,
        "pretrain_lr": 1e-3, "finetune_lr": 5e-4,
        "latent_dim": 64,
    },
    "cgan": {
        "gan_epochs": 100, "finetune_epochs": 50,
        "gan_lr": 2e-4, "finetune_lr": 5e-4, "latent_dim": 64,
    },
    "gan_aug": {
        "gan_epochs": 100, "train_epochs": 50, "lr": 1e-3,
        "aug_ratio": 0.2, "latent_dim": 64,
    },
}

# UCI cross-validation setup per dataset
UCI_CV = {
    "Cleveland": {"n_splits": 10, "n_repeats": 1},
    "Hungarian": {"n_splits": 10, "n_repeats": 1},
}

# Paths to UCI data files
UCI_FILES = {
    "Cleveland": UCI_DIR / "cleveland" / "processed.cleveland.data",
    "Hungarian": UCI_DIR / "hungarian" / "processed.hungarian.data",
}

# Downstream classifiers for representation transfer evaluation
DOWNSTREAM_CLASSIFIERS = ["LogReg", "LightGBM", "CatBoost", "XGBoost"]
