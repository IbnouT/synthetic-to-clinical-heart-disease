"""
Load and preprocess UCI Heart Disease clinical datasets for external validation.

The UCI Heart Disease collection includes four hospital subsets:
  - Cleveland (303 patients, 45.9% disease prevalence)
  - Hungarian (294 patients, 36.1%)
  - Switzerland (123 patients, 93.5%)
  - VA Long Beach (200 patients, 74.5%)

Each file uses the UCI processed format: 14 columns (13 features + target),
no header row, "?" for missing values. The target column encodes disease
severity 0-4, which we binarize to absence (0) vs presence (1-4).

Column names are mapped to the Kaggle naming convention so the same
feature pipelines can process both Kaggle and clinical data.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold


logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------
# Column definitions (UCI Heart Disease standard 14-attribute format)
# -----------------------------------------------------------------------

# Raw UCI column names as they appear in the processed.*.data files
UCI_COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak", "slope",
    "ca", "thal", "num",
]

# Map from UCI names to Kaggle column names for feature pipeline reuse
UCI_TO_KAGGLE = {
    "age": "Age",
    "sex": "Sex",
    "cp": "Chest pain type",
    "trestbps": "BP",
    "chol": "Cholesterol",
    "fbs": "FBS over 120",
    "restecg": "EKG results",
    "thalach": "Max HR",
    "exang": "Exercise angina",
    "oldpeak": "ST depression",
    "slope": "Slope of ST",
    "ca": "Number of vessels fluro",
    "thal": "Thallium",
    "num": "Heart Disease",
}

# The 13 clinical features (Kaggle naming convention)
FEATURE_COLUMNS = [
    "Age", "Sex", "Chest pain type", "BP", "Cholesterol",
    "FBS over 120", "EKG results", "Max HR", "Exercise angina",
    "ST depression", "Slope of ST", "Number of vessels fluro", "Thallium",
]

# Clinical fields where zero or negative values are physiologically impossible
# and should be treated as missing data sentinels
POSITIVE_ONLY = ["Age", "BP", "Cholesterol", "Max HR"]

# Feature type classification for imputation strategy
CATEGORICAL = [
    "Sex", "Chest pain type", "FBS over 120", "EKG results",
    "Exercise angina", "Slope of ST", "Number of vessels fluro", "Thallium",
]
NUMERICAL = ["Age", "BP", "Cholesterol", "Max HR", "ST depression"]

# Dataset metadata for cross-validation strategy selection
DATASET_INFO = {
    "cleveland":    {"path": "external/cleveland/processed.cleveland.data",  "cv_folds": 10, "cv_repeats": 1},
    "hungarian":    {"path": "external/hungarian/processed.hungarian.data",  "cv_folds": 10, "cv_repeats": 1},
    "switzerland":  {"path": "external/switzerland/processed.switzerland.data", "cv_folds": 5, "cv_repeats": 3},
    "va_longbeach": {"path": "external/va_longbeach/processed.va.data",     "cv_folds": 5, "cv_repeats": 3},
}


def _project_root() -> Path:
    """Resolve the code/ root (one level up from src/)."""
    return Path(__file__).resolve().parents[1]


def _impute(X: pd.DataFrame) -> None:
    """
    Fill missing values in place: median for continuous features,
    mode for categorical. Falls back to 0 if no valid values exist.
    """
    for col in NUMERICAL:
        if col not in X.columns:
            continue
        series = pd.to_numeric(X[col], errors="coerce")
        if series.isna().any():
            fill = series.median()
            if pd.isna(fill):
                fill = 0.0
            series = series.fillna(fill)
        X[col] = series

    for col in CATEGORICAL:
        if col not in X.columns:
            continue
        series = pd.to_numeric(X[col], errors="coerce")
        if series.isna().any():
            modes = series.mode(dropna=True)
            fill = modes.iloc[0] if len(modes) > 0 else 0.0
            series = series.fillna(fill)
        X[col] = series


def _clean_sentinels(X: pd.DataFrame, dataset_name: str) -> None:
    """
    Convert physiologically impossible values to NaN. Zero or negative
    values in age, blood pressure, cholesterol, or heart rate indicate
    missing data, not actual measurements.
    """
    for col in POSITIVE_ONLY:
        if col not in X.columns:
            continue
        series = pd.to_numeric(X[col], errors="coerce")
        bad = series <= 0
        if bad.any():
            X[col] = series.mask(bad, np.nan)
            logger.info(
                "%s: %d non-positive sentinel(s) in %s replaced with NaN",
                dataset_name, int(bad.sum()), col,
            )
        else:
            X[col] = series


def load_uci_dataset(
    name: str,
    data_root: Path | None = None,
) -> tuple[pd.DataFrame, np.ndarray, dict]:
    """
    Load one UCI Heart Disease subset and return cleaned features + binary target.

    Preprocessing steps:
    1. Read processed.*.data (no header, "?" as NaN)
    2. Map columns to Kaggle naming convention
    3. Binarize target: 0 stays 0, severity 1-4 becomes 1
    4. Replace physiologically impossible sentinel values with NaN
    5. Impute missing values (median for continuous, mode for categorical)
    6. Guarantee no NaN in output

    Returns (X, y, metadata) where X has the 13 standard feature columns.
    """
    if name not in DATASET_INFO:
        raise ValueError(
            f"Unknown dataset '{name}'. Available: {list(DATASET_INFO.keys())}"
        )

    info = DATASET_INFO[name]
    if data_root is None:
        data_root = _project_root() / "data"
    path = data_root / info["path"]

    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    # Read the raw UCI format: no header, "?" marks missing values
    df = pd.read_csv(path, header=None, na_values=["?"])

    if len(df.columns) != len(UCI_COLUMNS):
        logger.warning(
            "%s: got %d columns, expected %d",
            name, len(df.columns), len(UCI_COLUMNS),
        )
    df.columns = UCI_COLUMNS[:len(df.columns)]

    # Rename to Kaggle conventions so feature pipelines work unchanged
    df = df.rename(columns=UCI_TO_KAGGLE)

    # Binarize the target: original UCI encoding uses 0 (no disease)
    # and 1-4 (increasing severity). We collapse all severity levels to 1.
    target = pd.to_numeric(df["Heart Disease"], errors="coerce")
    if target.isna().any():
        raise ValueError(f"{name}: target column has non-numeric values")
    y = (target > 0).astype(int).values

    # Extract the 13 clinical features
    feature_cols = [c for c in FEATURE_COLUMNS if c in df.columns]
    X = df[feature_cols].copy()

    # Coerce everything to numeric
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    # Clean sentinel values and impute missing data
    _clean_sentinels(X, name)
    X = X.replace([np.inf, -np.inf], np.nan)
    _impute(X)

    # Final safety check: no NaN should survive
    residual = int(X.isna().sum().sum())
    if residual > 0:
        logger.warning("%s: %d residual NaN after imputation, filling with 0", name, residual)
        X = X.fillna(0.0)

    metadata = {
        "dataset": name,
        "n_samples": len(y),
        "n_features": len(feature_cols),
        "n_positive": int(y.sum()),
        "n_negative": int(len(y) - y.sum()),
        "prevalence": round(y.mean() * 100, 1),
        "cv_folds": info["cv_folds"],
        "cv_repeats": info["cv_repeats"],
    }

    logger.info(
        "Loaded %s: n=%d, prevalence=%.1f%%, pos=%d, neg=%d",
        name, len(y), metadata["prevalence"],
        metadata["n_positive"], metadata["n_negative"],
    )

    return X, y, metadata


def load_all_datasets(
    data_root: Path | None = None,
) -> dict[str, tuple[pd.DataFrame, np.ndarray, dict]]:
    """Load all four UCI datasets and return a dict keyed by name."""
    return {
        name: load_uci_dataset(name, data_root)
        for name in DATASET_INFO
    }


def get_cv_strategy(name: str):
    """
    Return the appropriate sklearn CV splitter for a dataset.

    Cleveland and Hungarian (n~300) use 10-fold stratified CV.
    Switzerland and VA Long Beach (n<200) use 5-fold x 3 repeats
    to stabilize estimates on small samples.
    """
    info = DATASET_INFO[name]
    if info["cv_repeats"] == 1:
        return StratifiedKFold(
            n_splits=info["cv_folds"],
            shuffle=True,
            random_state=42,
        )
    else:
        return RepeatedStratifiedKFold(
            n_splits=info["cv_folds"],
            n_repeats=info["cv_repeats"],
            random_state=42,
        )
