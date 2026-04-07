"""Basic feature pipelines with minimal transformation.

These pipelines return the 13 clinical features with simple encoding
(raw, string-typed, integer-coded, one-hot) and no engineered columns.
They serve as baselines and as inputs for models that handle feature
encoding internally (CatBoost native categoricals, TabPFN).

Pipelines:
    raw              13 original columns as-is
    raw_categorical  13 columns cast to string for CatBoost categoricals
    raw_integer      13 columns as integer codes for TabPFN
    onehot           one-hot encoded (file column order)
    te_alpha10       raw 13 columns (TE added per fold in training loop)
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

from src.config import ALL_FEATURES, FILE_ORDER_FEATURES


def build_raw(train_df, test_df):
    """Return the 13 original clinical input columns."""
    x_train = train_df[ALL_FEATURES].copy()
    x_test = test_df[ALL_FEATURES].copy()
    return x_train, x_test


def build_raw_categorical(train_df, test_df):
    """Return 13 features cast to strings for CatBoost categorical encoding."""
    x_train = train_df[ALL_FEATURES].astype(str).copy()
    x_test = test_df[ALL_FEATURES].astype(str).copy()
    return x_train, x_test


def build_raw_integer(train_df, test_df):
    """Integer-encode all 13 features for TabPFN.

    Category codes are derived from the combined train+test set so that
    both splits share the same integer mapping.
    """
    x_train = train_df[ALL_FEATURES].copy()
    x_test = test_df[ALL_FEATURES].copy()

    for col in ALL_FEATURES:
        combined = pd.concat([x_train[col], x_test[col]], ignore_index=True).astype("category")
        codes = combined.cat.codes.values
        n = len(x_train)
        x_train[col] = codes[:n]
        x_test[col] = codes[n:]

    return x_train.values.astype(np.float32), x_test.values.astype(np.float32)


def build_onehot(train_df, test_df):
    """One-hot encode all 13 features in csv file column order."""
    enc = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    x_train = enc.fit_transform(train_df[FILE_ORDER_FEATURES].astype(str))
    x_test = enc.transform(test_df[FILE_ORDER_FEATURES].astype(str))
    return x_train, x_test


def build_te_alpha10(train_df, test_df):
    """
    Raw features (file order) for CatBoost with per-fold target encoding.

    Returns only the 13 raw columns. Target encoding, orig stats, and
    frequency columns depend on the fold split and are added per fold
    in the training loop.
    """
    x_train = train_df[FILE_ORDER_FEATURES].copy()
    x_test = test_df[FILE_ORDER_FEATURES].copy()
    return x_train, x_test


# All basic pipeline builders.
BASIC_BUILDERS = {
    "raw": build_raw,
    "raw_categorical": build_raw_categorical,
    "raw_integer": build_raw_integer,
    "onehot": build_onehot,
    "te_alpha10": build_te_alpha10,
}
