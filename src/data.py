"""Data loading and fold handling for the Kaggle and UCI datasets.

The Kaggle dataset has ~630K synthetic samples generated from the UCI
Heart Disease collection. Labels are stored as text ("Presence"/"Absence")
and converted to binary 0/1 here so downstream code works on numeric targets.

Folds are precomputed and saved to disk the first time they are needed.
This ensures that every run uses the exact same train/validation split,
which is critical for reproducibility of the OOF predictions.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from src.config import FOLDS_DIR
from src.config import ID_COLUMN
from src.config import N_FOLDS
from src.config import ORIGINAL_PATH
from src.config import PRIMARY_SEED
from src.config import TARGET_COLUMN
from src.config import TARGET_POSITIVE
from src.config import TEST_PATH
from src.config import TRAIN_PATH


def load_train_test():
    """Load Kaggle train/test dataframes, adding a binary target column."""
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    # Convert text label ("Presence"/"Absence") to 0/1.
    train_df["target"] = (train_df[TARGET_COLUMN] == TARGET_POSITIVE).astype(int)
    return train_df, test_df


def load_original_uci():
    """Load the original 270-row UCI Cleveland dataset for feature statistics."""
    original_df = pd.read_csv(ORIGINAL_PATH)
    original_df["target"] = (original_df[TARGET_COLUMN] == TARGET_POSITIVE).astype(int)
    return original_df


def load_folds(seed=PRIMARY_SEED, n_folds=N_FOLDS):
    """
    Load precomputed fold indices, or build and save them if missing.

    Returns a list of (train_indices, val_indices) tuples. The fold file
    is saved as a compressed numpy archive keyed by fold number.
    """
    fold_path = FOLDS_DIR / f"folds_s{seed}_k{n_folds}.npz"

    # Load existing fold file if available.
    if fold_path.exists():
        fold_file = np.load(fold_path)
        folds = []
        for i in range(n_folds):
            folds.append((fold_file[f"train_{i}"], fold_file[f"val_{i}"]))
        return folds

    # Build new fold split and save it for future runs.
    train_df, _ = load_train_test()
    y = train_df["target"].to_numpy()

    splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    arrays = {}
    folds = []

    # Generate stratified splits and store them as named arrays.
    for i, (train_idx, val_idx) in enumerate(splitter.split(train_df, y)):
        arrays[f"train_{i}"] = train_idx.astype(np.int32)
        arrays[f"val_{i}"] = val_idx.astype(np.int32)
        folds.append((train_idx, val_idx))

    # Save to disk so all future runs use the exact same folds.
    np.savez_compressed(fold_path, **arrays)
    return folds


def load_test_ids():
    """Return the test row IDs needed for building submission files."""
    _, test_df = load_train_test()
    return test_df[ID_COLUMN].copy()
