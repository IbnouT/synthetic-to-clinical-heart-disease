"""Load competition and UCI clinical datasets (raw 13 features, scaled).

Competition data: 630K samples from the S6E2 Kaggle competition.
UCI data: Cleveland (303) and Hungarian (294) clinical subsets.

Missing values in UCI are imputed rather than dropped to preserve
full sample counts. Numerical features use median, categorical use mode.
Non-positive values in fields like cholesterol and blood pressure are
treated as clinical sentinel values and replaced before imputation.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from ..config import COMPETITION_CSV, UCI_FILES, RANDOM_SEED, TRAIN_SPLIT

# Standard 13 UCI Heart Disease features
FEATURE_NAMES = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak", "slope",
    "ca", "thal",
]

# Used for imputation strategy: median vs mode
_NUMERICAL = {"age", "trestbps", "chol", "thalach", "oldpeak"}
_CATEGORICAL = {"sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"}
# These fields should be positive; zero or negative means missing
_POSITIVE_ONLY = {"chol", "trestbps", "thalach"}


def load_competition():
    """Load competition data, scale, and split 80/20.

    Returns (X_train, X_val, y_train, y_val, X_all, y_all, scaler).
    The scaler is fitted on all data and used later to scale UCI samples.
    """
    df = pd.read_csv(COMPETITION_CSV)
    target_col = "Heart Disease"
    feature_cols = [c for c in df.columns if c not in ("id", target_col)]

    X = df[feature_cols].values.astype(np.float32)
    # Target is 'Presence'/'Absence' string in this dataset
    y = (df[target_col].str.strip().str.lower() == "presence").astype(np.float32)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_val, y_train, y_val = train_test_split(
        X_scaled, y, test_size=TRAIN_SPLIT,
        random_state=RANDOM_SEED, stratify=y)

    return X_train, X_val, y_train, y_val, X_scaled, y, scaler


def load_uci(scaler):
    """Load UCI datasets, impute missing values, scale with competition scaler.

    Returns dict: {name: {"X": scaled_array, "y": binary_labels, "n": count}}.
    """
    col_names = FEATURE_NAMES + ["target"]
    datasets = {}

    for name, filepath in UCI_FILES.items():
        df = pd.read_csv(filepath, header=None, na_values="?")
        df.columns = col_names[:len(df.columns)]

        # Target: 0 = healthy, 1-4 = disease stages -> binary
        df["target"] = (pd.to_numeric(df["target"], errors="coerce") > 0).astype(int)

        for col in FEATURE_NAMES:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # Replace sentinel values (e.g. cholesterol=0 means not recorded)
        for col in _POSITIVE_ONLY:
            if col in df.columns:
                df.loc[df[col] <= 0, col] = np.nan

        # Impute: median for continuous, mode for discrete
        for col in _NUMERICAL:
            if col in df.columns and df[col].isna().any():
                df[col] = df[col].fillna(df[col].median())
        for col in _CATEGORICAL:
            if col in df.columns and df[col].isna().any():
                mode = df[col].mode(dropna=True)
                df[col] = df[col].fillna(mode.iloc[0] if len(mode) > 0 else 0)

        # Safety: anything still NaN gets zeroed
        df[FEATURE_NAMES] = df[FEATURE_NAMES].fillna(0)

        X = scaler.transform(df[FEATURE_NAMES].values.astype(np.float32))
        y = df["target"].values.astype(np.float32)
        datasets[name] = {"X": X, "y": y, "n": len(y)}

    return datasets
