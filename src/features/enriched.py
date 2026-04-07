"""Encoding-focused feature pipelines.

These pipelines combine frequency encoding, UCI target means, and
one-hot encoding into feature matrices for models that benefit from
explicit distributional information. None of them use per-value
multi-statistic enrichment (that's in origstats.py).

Pipelines:
    cb_baseline          39 cols: raw (strings) + frequency + UCI means
    tree39               39 cols: integer-coded raw + frequency + UCI means
    enriched_tree        34 cols: raw + UCI means + cat frequency
    onehot_scaled        one-hot + freq + UCI means, StandardScaler
    onehot_freq_origmean 466 cols: one-hot + freq + UCI means (float32)
    freq_origmean        26 cols: frequency + UCI means only
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import ALL_FEATURES, CAT_FEATURES, FILE_ORDER_FEATURES
from src.data import load_original_uci
from src.features.helpers import (
    frequency_encode_column,
    uci_mean_column,
)


def build_cb_baseline(train_df, test_df):
    """
    39 columns: 13 raw (as strings for CatBoost native categoricals)
    + 13 frequency (train+test combined) + 13 UCI target averages.
    """
    x_train = train_df[ALL_FEATURES].copy().astype(str)
    x_test = test_df[ALL_FEATURES].copy().astype(str)
    original_df = load_original_uci()

    for col in ALL_FEATURES:
        freq_tr, freq_te = frequency_encode_column(
            train_df[col], test_df[col], source="combined"
        )
        x_train[f"{col}_freq"] = freq_tr
        x_test[f"{col}_freq"] = freq_te

        mean_tr, mean_te = uci_mean_column(train_df[col], test_df[col], original_df)
        x_train[f"{col}_orig"] = mean_tr
        x_test[f"{col}_orig"] = mean_te

    return x_train, x_test


def build_tree39(train_df, test_df):
    """
    39 columns: 13 integer-coded raw + 13 frequency + 13 UCI target means.

    Integer coding maps each unique value to a sequential integer based on
    sorted order from the combined train+test set. This gives tree-based
    models a numeric representation without one-hot expansion.
    """
    original_df = load_original_uci()
    x_train = pd.DataFrame(index=train_df.index)
    x_test = pd.DataFrame(index=test_df.index)

    # Integer-encode each feature using sorted unique values from both sets.
    for col in ALL_FEATURES:
        combined = pd.concat([train_df[col], test_df[col]], ignore_index=True)
        sorted_vals = sorted(combined.unique())
        val_to_int = {v: i for i, v in enumerate(sorted_vals)}
        x_train[f"{col}_raw"] = train_df[col].map(val_to_int).fillna(-1).astype(float)
        x_test[f"{col}_raw"] = test_df[col].map(val_to_int).fillna(-1).astype(float)

    for col in ALL_FEATURES:
        freq_tr, freq_te = frequency_encode_column(
            train_df[col], test_df[col], source="combined"
        )
        x_train[f"{col}_freq"] = freq_tr
        x_test[f"{col}_freq"] = freq_te

    for col in ALL_FEATURES:
        mean_tr, mean_te = uci_mean_column(train_df[col], test_df[col], original_df)
        x_train[f"{col}_orig"] = mean_tr
        x_test[f"{col}_orig"] = mean_te

    return x_train, x_test


def build_enriched_tree(train_df, test_df):
    """
    34 columns: 13 raw (file order) + 13 UCI target means + 8 cat frequency.

    All numeric output. Frequency is only for the 8 categorical features,
    using the training set distribution.
    """
    original_df = load_original_uci()

    x_train = train_df[FILE_ORDER_FEATURES].copy()
    x_test = test_df[FILE_ORDER_FEATURES].copy()

    # UCI target mean per feature value from the 270-row dataset.
    for col in ALL_FEATURES:
        if col not in original_df.columns:
            continue
        mean_tr, mean_te = uci_mean_column(train_df[col], test_df[col], original_df)
        x_train[f"{col}_orig"] = mean_tr.fillna(0.5)
        x_test[f"{col}_orig"] = mean_te.fillna(0.5)

    # Training-set frequency for categorical features.
    for col in CAT_FEATURES:
        freq_tr, freq_te = frequency_encode_column(
            train_df[col], test_df[col], source="train"
        )
        x_train[f"{col}_freq"] = freq_tr
        x_test[f"{col}_freq"] = freq_te

    return x_train, x_test


def build_onehot_scaled(train_df, test_df):
    """One-hot + frequency + UCI target means, then StandardScaler."""
    original_df = load_original_uci()

    # One-hot encode using ALL_FEATURES order.
    enc = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    x_oh_train = enc.fit_transform(train_df[ALL_FEATURES])
    x_oh_test = enc.transform(test_df[ALL_FEATURES])

    # Combined train+test frequency per feature.
    freq_train, freq_test = [], []
    for col in ALL_FEATURES:
        fr_tr, fr_te = frequency_encode_column(
            train_df[col], test_df[col], source="combined"
        )
        freq_train.append(fr_tr.values.reshape(-1, 1))
        freq_test.append(fr_te.values.reshape(-1, 1))

    # UCI target mean per feature value.
    orig_train, orig_test = [], []
    for col in ALL_FEATURES:
        if col not in original_df.columns:
            continue
        m_tr, m_te = uci_mean_column(train_df[col], test_df[col], original_df)
        orig_train.append(m_tr.values.reshape(-1, 1))
        orig_test.append(m_te.values.reshape(-1, 1))

    x_train = np.hstack([x_oh_train] + freq_train + orig_train)
    x_test = np.hstack([x_oh_test] + freq_test + orig_test)

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    return x_train, x_test


def build_onehot_freq_origmean(train_df, test_df):
    """
    One-hot + frequency + UCI target means, returned as float32 arrays.

    Wide matrix for models that benefit from explicit encoding of all
    feature interactions (MLP, KNN). Column order: onehot, freq, orig mean.
    """
    original_df = load_original_uci()

    enc = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    x_oh_train = enc.fit_transform(train_df[FILE_ORDER_FEATURES].astype(str))
    x_oh_test = enc.transform(test_df[FILE_ORDER_FEATURES].astype(str))

    freq_train, freq_test = [], []
    for col in FILE_ORDER_FEATURES:
        fr_tr, fr_te = frequency_encode_column(
            train_df[col], test_df[col], source="combined"
        )
        freq_train.append(fr_tr.values.reshape(-1, 1))
        freq_test.append(fr_te.values.reshape(-1, 1))

    orig_train, orig_test = [], []
    for col in FILE_ORDER_FEATURES:
        if col not in original_df.columns:
            continue
        m_tr, m_te = uci_mean_column(train_df[col], test_df[col], original_df)
        orig_train.append(m_tr.values.reshape(-1, 1))
        orig_test.append(m_te.values.reshape(-1, 1))

    x_train = np.hstack([x_oh_train] + freq_train + orig_train).astype(np.float32)
    x_test = np.hstack([x_oh_test] + freq_test + orig_test).astype(np.float32)

    return x_train, x_test


def build_freq_origmean(train_df, test_df):
    """
    26 columns: 13 combined frequency + 13 UCI target means.

    No raw features. The model operates entirely on encoded
    representations: distributional frequency and clinical outcome
    information per feature level.
    """
    original_df = load_original_uci()

    freq_parts_tr, freq_parts_te = [], []
    for col in ALL_FEATURES:
        freq_tr, freq_te = frequency_encode_column(
            train_df[col], test_df[col], source="combined"
        )
        freq_parts_tr.append(freq_tr.values.reshape(-1, 1))
        freq_parts_te.append(freq_te.values.reshape(-1, 1))

    orig_parts_tr, orig_parts_te = [], []
    global_mean = float(original_df["target"].mean())
    for col in ALL_FEATURES:
        if col not in original_df.columns:
            continue
        mean_map = original_df.groupby(col)["target"].mean().to_dict()
        orig_parts_tr.append(
            train_df[col].map(mean_map).fillna(global_mean).values.reshape(-1, 1)
        )
        orig_parts_te.append(
            test_df[col].map(mean_map).fillna(global_mean).values.reshape(-1, 1)
        )

    x_train = np.hstack(freq_parts_tr + orig_parts_tr)
    x_test = np.hstack(freq_parts_te + orig_parts_te)

    return x_train, x_test


ENRICHED_BUILDERS = {
    "cb_baseline": build_cb_baseline,
    "tree39": build_tree39,
    "enriched_tree": build_enriched_tree,
    "onehot_scaled": build_onehot_scaled,
    "onehot_freq_origmean": build_onehot_freq_origmean,
    "freq_origmean": build_freq_origmean,
}
