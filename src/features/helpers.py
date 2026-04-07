"""Shared encoding and statistics helpers for feature pipelines.

These primitives are used across multiple feature pipelines to avoid
duplicating frequency encoding, target encoding, and UCI statistics
computation. Each function handles one encoding operation and returns
arrays ready to be added to a feature matrix.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from scipy.stats import skew


# ---------------------------------------------------------------------------
# Frequency encoding
# ---------------------------------------------------------------------------

def frequency_encode_column(train_col, test_col, source="combined"):
    """
    Map each feature value to its normalized frequency.

    When source="combined", frequency is computed from the union of
    train and test values. When source="train", only training data
    is used. Test values not seen in the reference set get frequency 0.

    Parameters
    ----------
    train_col : pd.Series
        Training column.
    test_col : pd.Series
        Test column.
    source : str
        "combined" for train+test frequency, "train" for train-only.

    Returns
    -------
    train_freq : pd.Series
    test_freq : pd.Series
    """
    if source == "combined":
        reference = pd.concat([train_col, test_col], ignore_index=True)
    else:
        reference = train_col

    freq_map = reference.value_counts(normalize=True)
    return train_col.map(freq_map), test_col.map(freq_map).fillna(0.0)


def frequency_encode_columns(train_df, test_df, columns, source="train"):
    """
    Frequency-encode multiple columns, returning new DataFrames.

    Each column gets a ``{col}_freq`` output column. Uses only the
    training distribution by default (safer for evaluation).

    Parameters
    ----------
    train_df : pd.DataFrame
    test_df : pd.DataFrame
    columns : list of str
    source : str
        "combined" or "train".

    Returns
    -------
    train_freq : pd.DataFrame
        Columns named ``{col}_freq``.
    test_freq : pd.DataFrame
        Same columns, same order.
    """
    train_freq = pd.DataFrame(index=train_df.index)
    test_freq = pd.DataFrame(index=test_df.index)

    for col in columns:
        tr, te = frequency_encode_column(train_df[col], test_df[col], source=source)
        train_freq[f"{col}_freq"] = tr
        test_freq[f"{col}_freq"] = te

    return train_freq, test_freq


# ---------------------------------------------------------------------------
# Out-of-fold target encoding
# ---------------------------------------------------------------------------

def target_encode_oof(train_df, test_df, columns, y, n_splits=5, seed=42):
    """
    Out-of-fold target encoding.

    Each training row gets the mean target value from the other folds.
    Test rows use the full training set mean per feature value. Unseen
    values fall back to the global target mean.

    Parameters
    ----------
    train_df : pd.DataFrame
        Training data with the feature columns.
    test_df : pd.DataFrame
        Test data with the same feature columns.
    columns : list of str
        Columns to target-encode.
    y : np.ndarray
        Binary target vector aligned with train_df.
    n_splits : int
        Number of stratified folds for the OOF computation.
    seed : int
        Random seed for fold splitting.

    Returns
    -------
    train_te : pd.DataFrame
        Columns named ``{col}_te``.
    test_te : pd.DataFrame
        Same columns, same order.
    """
    _target = "__te_target__"
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    train_te = pd.DataFrame(index=train_df.index)
    test_te = pd.DataFrame(index=test_df.index)

    # Temporary frame with target for groupby operations.
    train_with_target = train_df.copy()
    train_with_target[_target] = y

    for col in columns:
        te_col = f"{col}_te"
        train_te[te_col] = 0.0

        # Each validation fold gets means from the other folds.
        for tr_idx, val_idx in skf.split(train_df, y):
            fold_means = train_with_target.iloc[tr_idx].groupby(col)[_target].mean()
            train_te.iloc[val_idx, train_te.columns.get_loc(te_col)] = (
                train_df[col].iloc[val_idx].map(fold_means)
            )

        # Test rows use the full training set mean per value.
        full_means = train_with_target.groupby(col)[_target].mean()
        test_te[te_col] = test_df[col].map(full_means).fillna(y.mean())

    return train_te, test_te


# ---------------------------------------------------------------------------
# UCI original dataset statistics
# ---------------------------------------------------------------------------

def uci_mean_column(train_col, test_col, original_df):
    """
    Map each value to its target average in the original UCI dataset.

    Values not present in the 270-row Cleveland dataset get the global
    target mean as fallback.
    """
    global_mean = float(original_df["target"].mean())
    mean_map = original_df.groupby(train_col.name)["target"].mean().to_dict()
    return (
        train_col.map(mean_map).fillna(global_mean),
        test_col.map(mean_map).fillna(global_mean),
    )


def uci_stats_columns(train_df, test_df, original_df, columns,
                       stats=("mean", "median", "std", "count"),
                       prefix="orig"):
    """
    Add per-value target statistics from a reference dataset.

    For each column, computes the requested aggregation statistics on the
    target variable grouped by feature value in the reference dataset.
    Creates ``{prefix}_{col}_{stat}`` columns in copies of the input frames.

    Parameters
    ----------
    train_df : pd.DataFrame
    test_df : pd.DataFrame
    original_df : pd.DataFrame
        Reference dataset with a "target" column (e.g. 270-row UCI original
        or 630K competition data).
    columns : list of str
        Columns to compute stats for.
    stats : tuple of str
        Aggregation functions to compute (from pandas GroupBy.agg).
    prefix : str
        Column name prefix ("orig" for UCI stats, "comp" for competition).

    Returns
    -------
    x_train : pd.DataFrame
        Copy of train_df[columns] with stat columns appended.
    x_test : pd.DataFrame
        Copy of test_df[columns] with stat columns appended.
    """
    x_train = train_df[columns].copy()
    x_test = test_df[columns].copy()

    # Global fallback values for each stat type.
    global_fills = {
        "mean": float(original_df["target"].mean()),
        "median": float(original_df["target"].median()),
        "std": 0.0,
        "skew": 0.0,
        "count": 0.0,
    }

    for col in columns:
        if col not in original_df.columns:
            continue

        # Replace string "skew" with scipy.stats.skew (biased estimator)
        # to match the competition codebase. Pandas .skew() uses the
        # unbiased estimator which gives slightly different values.
        agg_funcs = [skew if s == "skew" else s for s in stats]
        grouped = original_df.groupby(col)["target"].agg(agg_funcs).reset_index()
        stat_names = {s: f"{prefix}_{col}_{s}" for s in stats}
        grouped.columns = [col] + [stat_names[s] for s in stats]

        fill = {stat_names[s]: global_fills.get(s, 0.0) for s in stats}

        train_merged = train_df[[col]].merge(grouped, on=col, how="left").fillna(fill)
        test_merged = test_df[[col]].merge(grouped, on=col, how="left").fillna(fill)

        for stat_col in fill:
            x_train[stat_col] = train_merged[stat_col].values
            x_test[stat_col] = test_merged[stat_col].values

    return x_train, x_test
