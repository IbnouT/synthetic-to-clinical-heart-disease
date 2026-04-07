"""Top pipeline feature engineering.

Based on Mahajan's (2026) CB+XGB residual RF notebook for the
Playground Series S6E2 dataset. Combines frequency encoding,
OOF target encoding, and correlation-based interaction features
into ~47 columns suitable for any model family.

See REFERENCES.md for full attribution.
"""

import numpy as np
import pandas as pd
from itertools import combinations

from src.config import ALL_FEATURES
from src.features.helpers import frequency_encode_columns, target_encode_oof


def _add_correlation_interactions(train_te, test_te, n_pairs=8):
    """
    Multiply the top correlated pairs of target-encoded features.

    Identifies the n_pairs most correlated TE feature pairs (by absolute
    Pearson correlation) and creates their product as new features. This
    captures multiplicative interaction effects between features that
    individually predict the target well.

    The number of pairs (8) was chosen empirically in the original
    notebook to balance expressiveness against overfitting risk.
    """
    filled = train_te.fillna(0)
    corr_scores = {}
    for col_a, col_b in combinations(filled.columns, 2):
        r = abs(np.corrcoef(filled[col_a], filled[col_b])[0, 1])
        corr_scores[(col_a, col_b)] = r

    top_pairs = sorted(corr_scores, key=corr_scores.get, reverse=True)[:n_pairs]

    for col_a, col_b in top_pairs:
        interaction_name = f"{col_a}_x_{col_b}"
        train_te[interaction_name] = train_te[col_a] * train_te[col_b]
        test_te[interaction_name] = test_te[col_a] * test_te[col_b]

    return train_te, test_te


def build_top_pipe(train_df, test_df):
    """
    Build the full top-pipeline feature set.

    Combines frequency encoding, OOF target encoding, and correlation
    interaction features. All 13 clinical features get both freq and TE
    columns, plus the top 8 correlated TE pairs are multiplied for
    interaction features.

    Returns feature DataFrames ready for model training (all numeric).
    """
    all_cols = ALL_FEATURES
    y = train_df["target"].values

    # Frequency encoding for all features (train-only distribution).
    train_freq, test_freq = frequency_encode_columns(
        train_df, test_df, all_cols, source="train"
    )

    # OOF target encoding for all features.
    train_te, test_te = target_encode_oof(train_df, test_df, all_cols, y)

    # Correlation-based interaction features from the TE columns.
    train_te, test_te = _add_correlation_interactions(train_te, test_te, n_pairs=8)

    # Combine frequency and target-encoded features.
    x_train = pd.concat([train_freq, train_te], axis=1).fillna(0)
    x_test = pd.concat([test_freq, test_te], axis=1).fillna(0)

    return x_train, x_test
