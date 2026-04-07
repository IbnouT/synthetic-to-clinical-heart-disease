"""Data loading for SSL experiments.

Loads the S6E2 competition dataset (630K synthetic heart disease samples)
and 4 UCI clinical heart disease subsets (Cleveland, Hungarian,
Switzerland, VA Long Beach).

The competition data is used for pretraining SSL encoders. The UCI
datasets are used for transfer evaluation: can representations learned
from synthetic data improve classification on real clinical data?

Data is shared with the main project:
  code/data/competition/train.csv              Competition training set (630K rows)
  code/data/external/{cleveland,...}/*.data     UCI clinical subsets
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler


# Data lives in code/data/ (shared with the competition pipeline).
# This file is at code/experiments/ssl/data.py, so code/ is 3 levels up.
_CODE_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = _CODE_ROOT / "data" / "competition"

# The 13 clinical features shared across competition and UCI data.
# Competition CSV uses these column names; UCI uses positional columns.
COMPETITION_FEATURE_COLS = [
    "Age", "Sex", "Chest pain type", "BP", "Cholesterol", "FBS over 120",
    "EKG results", "Max HR", "Exercise angina", "ST depression",
    "Slope of ST", "Number of vessels fluro", "Thallium",
]

# UCI column names (same 13 features, different naming convention)
UCI_FEATURE_NAMES = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal",
]

# Clinical fields where zero or negative values indicate missing data
_POSITIVE_ONLY = {"age", "trestbps", "chol", "thalach"}
_CATEGORICAL = {"sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"}
_NUMERICAL = {"age", "trestbps", "chol", "thalach", "oldpeak"}


def load_competition_data():
    """Load the 630K competition training set.

    Returns (X, y, scaler) where X is scaled features, y is binary
    labels, and scaler is the fitted StandardScaler (needed to scale
    UCI data the same way at transfer time).
    """
    train_path = DATA_DIR / "train.csv"
    if not train_path.exists():
        raise FileNotFoundError(
            f"Competition data not found at {train_path}. "
            "Ensure code/data/competition/train.csv exists."
        )
    train_df = pd.read_csv(train_path)
    train_df["target"] = (train_df["Heart Disease"] == "Presence").astype(int)

    X_raw = train_df[COMPETITION_FEATURE_COLS].values.astype(np.float32)
    y = train_df["target"].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw).astype(np.float32)

    return X_scaled, y, scaler


def load_uci_datasets():
    """Load 4 UCI Heart Disease clinical subsets.

    Missing values are imputed (median for numerical, mode for
    categorical) to preserve full sample counts (303/294/123/200).

    Returns a dict: {name: {"X": array, "y": array, "n": int}}.
    """
    uci_dir = _CODE_ROOT / "data" / "external"

    files = {
        "Cleveland":     uci_dir / "cleveland" / "processed.cleveland.data",
        "Hungarian":     uci_dir / "hungarian" / "processed.hungarian.data",
        "Switzerland":   uci_dir / "switzerland" / "processed.switzerland.data",
        "VA Long Beach": uci_dir / "va_longbeach" / "processed.va.data",
    }

    col_names = UCI_FEATURE_NAMES + ["target"]
    datasets = {}

    for name, filepath in files.items():
        if not filepath.exists():
            print(f"  Warning: {filepath} not found, skipping {name}")
            continue

        df = pd.read_csv(filepath, header=None, na_values="?")
        df.columns = col_names[:len(df.columns)]

        # Binarize: 0 = no disease, 1-4 = disease present
        df["target"] = (pd.to_numeric(df["target"], errors="coerce") > 0).astype(int)

        for col in UCI_FEATURE_NAMES:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # Replace non-positive sentinels with NaN in clinical fields
        for col in _POSITIVE_ONLY:
            if col in df.columns:
                df.loc[df[col] <= 0, col] = np.nan

        # Impute missing values
        for col in _NUMERICAL:
            if col in df.columns and df[col].isna().any():
                df[col] = df[col].fillna(df[col].median())
        for col in _CATEGORICAL:
            if col in df.columns and df[col].isna().any():
                mode_val = df[col].mode(dropna=True)
                df[col] = df[col].fillna(mode_val.iloc[0] if len(mode_val) > 0 else 0)

        df[UCI_FEATURE_NAMES] = df[UCI_FEATURE_NAMES].fillna(0)

        X = df[UCI_FEATURE_NAMES].values.astype(np.float32)
        y = df["target"].values

        datasets[name] = {"X": X, "y": y, "n": len(y)}
        print(f"  {name:15s}: n={len(y):3d}, positive={y.mean():.1%}")

    return datasets


# Cross-validation protocols matching the paper. Larger datasets use
# standard 10-fold; smaller ones use repeated 5-fold for stability.
CV_CONFIG = {
    "Cleveland":     {"n_splits": 10, "n_repeats": 1},
    "Hungarian":     {"n_splits": 10, "n_repeats": 1},
    "Switzerland":   {"n_splits": 5,  "n_repeats": 3},
    "VA Long Beach": {"n_splits": 5,  "n_repeats": 3},
}
