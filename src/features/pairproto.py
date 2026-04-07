"""Pairwise and prototype distance feature pipeline.

Builds features for models needing diversity from the standard origstats
family. Two feature types are combined with standard origstats:

1. Bayesian-smoothed pairwise target stats from the original UCI dataset.
   For each clinical feature pair, computes smoothed target prevalence
   and log-odds from the 270-row Cleveland study, with prior smoothing
   (alpha=15) to stabilize estimates for rare combinations.

2. Gower-style prototype distances. Each row is compared to every
   positive and negative case in the original dataset using a mixed
   distance metric (categorical mismatch + range-normalized numeric
   difference). The min/mean-k distances plus derived margin features
   give the model a kNN-like signal from the clinical reference data.

Together these features capture pairwise interaction patterns and
instance-level similarity that individual feature statistics miss.
"""

import numpy as np
import pandas as pd

from src.config import ALL_FEATURES, CAT_FEATURES, NUM_FEATURES
from src.data import load_original_uci
from src.features.helpers import uci_stats_columns


# Feature pairs chosen based on clinical relevance and error analysis.
CLINICAL_PAIRS = [
    ("Thallium", "Number of vessels fluro"),
    ("Thallium", "ST depression_bin"),
    ("Chest pain type", "Slope of ST"),
    ("Exercise angina", "ST depression_bin"),
    ("Sex", "Age_bin"),
    ("Max HR_bin", "Exercise angina"),
    ("EKG results", "Max HR_bin"),
    ("Number of vessels fluro", "Chest pain type"),
    ("Thallium", "Chest pain type"),
    ("Slope of ST", "ST depression_bin"),
]

# Bayesian smoothing strength. Higher values pull rare-combination
# estimates toward the global target rate more aggressively.
SMOOTHING_ALPHA = 15


def _bin_numeric_column(series, reference_series, n_bins=5):
    """
    Bin a numeric column into quantile-based integer bins.

    Bin edges are computed from the reference series (training data)
    to ensure consistent binning across train and test sets.
    """
    _, edges = pd.qcut(reference_series, q=n_bins, duplicates="drop", retbins=True)
    binned = pd.cut(series, bins=edges, labels=False, include_lowest=True)
    return binned.fillna(-1).astype(int)


def _build_pairwise_features(train_df, test_df, original_df, target_col="target"):
    """
    Compute Bayesian-smoothed target statistics for each feature pair.

    For each of 10 clinical feature pairs, computes from the original
    270-row UCI dataset:
      - Smoothed target prevalence (shrunk toward global rate)
      - Log-odds of the smoothed prevalence
      - Sample count for the pair combination
      - Binary indicator for missing (unseen) combinations
      - Binary indicator for rare combinations (count < 3)

    Returns 50 new columns (5 per pair) as dicts of arrays.
    """
    global_rate = float(original_df[target_col].mean())

    # Add quantile bins for numeric features that appear in pairs.
    for df in [train_df, test_df, original_df]:
        for col in NUM_FEATURES:
            bin_col = f"{col}_bin"
            if bin_col not in df.columns:
                df[bin_col] = _bin_numeric_column(df[col], train_df[col])

    pairwise_train_parts = {}
    pairwise_test_parts = {}

    for col_a, col_b in CLINICAL_PAIRS:
        pair_tag = f"pair_{col_a}_{col_b}"

        # Aggregate target stats from the original dataset for this pair.
        grouped = original_df.groupby([col_a, col_b])[target_col].agg(
            ["sum", "count"]
        ).reset_index()
        grouped.columns = [col_a, col_b, "sum_pos", "n"]

        # Bayesian smoothed mean: pulls sparse estimates toward the prior.
        grouped["smoothed_mean"] = (
            (grouped["sum_pos"] + SMOOTHING_ALPHA * global_rate)
            / (grouped["n"] + SMOOTHING_ALPHA)
        )

        # Log-odds transformation for linear separability.
        eps = 1e-4
        grouped["log_odds"] = np.log(
            (grouped["smoothed_mean"] + eps) / (1 - grouped["smoothed_mean"] + eps)
        )

        # Map pair stats onto train and test rows via left join.
        for df, parts_dict in [(train_df, pairwise_train_parts),
                                (test_df, pairwise_test_parts)]:
            merged = df[[col_a, col_b]].merge(
                grouped[[col_a, col_b, "smoothed_mean", "log_odds", "n"]],
                on=[col_a, col_b],
                how="left",
            )
            parts_dict[f"{pair_tag}_smean"] = merged["smoothed_mean"].fillna(global_rate).values
            parts_dict[f"{pair_tag}_logodds"] = merged["log_odds"].fillna(0.0).values
            parts_dict[f"{pair_tag}_count"] = merged["n"].fillna(0).values
            parts_dict[f"{pair_tag}_missing"] = merged["smoothed_mean"].isna().astype(int).values
            parts_dict[f"{pair_tag}_rare"] = (merged["n"].fillna(0) < 3).astype(int).values

    return pairwise_train_parts, pairwise_test_parts


def _gower_distance_batch(df, prototypes, num_ranges, chunk_size=10000):
    """
    Compute Gower-style distances from each row to a set of prototypes.

    Categorical features contribute mismatch fraction (0 or 1 per
    feature, averaged). Numeric features contribute range-normalized
    absolute differences. The two components are averaged.

    Returns min distance, mean of 3 nearest, and mean of 5 nearest
    for memory-efficient chunked computation.
    """
    proto_cat = prototypes[CAT_FEATURES].values
    proto_num = prototypes[NUM_FEATURES].values / (num_ranges.values + 1e-8)

    n = len(df)
    min_dist = np.full(n, np.inf)
    mean_k3 = np.zeros(n)
    mean_k5 = np.zeros(n)

    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        chunk_cat = df[CAT_FEATURES].values[start:end]
        chunk_num = df[NUM_FEATURES].values[start:end] / (num_ranges.values + 1e-8)

        cat_dist = np.zeros((end - start, len(proto_cat)))
        for j in range(len(proto_cat)):
            cat_dist[:, j] = (chunk_cat != proto_cat[j]).sum(axis=1) / len(CAT_FEATURES)

        num_dist = np.zeros((end - start, len(proto_num)))
        for j in range(len(proto_num)):
            num_dist[:, j] = np.abs(chunk_num - proto_num[j]).mean(axis=1)

        total_dist = (cat_dist + num_dist) / 2

        min_dist[start:end] = total_dist.min(axis=1)

        if total_dist.shape[1] >= 3:
            mean_k3[start:end] = np.partition(total_dist, 3, axis=1)[:, :3].mean(axis=1)
        else:
            mean_k3[start:end] = total_dist.mean(axis=1)

        if total_dist.shape[1] >= 5:
            mean_k5[start:end] = np.partition(total_dist, 5, axis=1)[:, :5].mean(axis=1)
        else:
            mean_k5[start:end] = total_dist.mean(axis=1)

    return min_dist, mean_k3, mean_k5


def _build_prototype_features(train_df, test_df, original_df, target_col="target"):
    """
    Compute Gower-style distances to positive and negative prototypes.

    Each row gets 10 features: min/mean-3/mean-5 distance to positive
    and negative prototypes, plus margin, ratio, nearest label, and
    smooth margin columns.
    """
    orig_pos = original_df[original_df[target_col] == 1]
    orig_neg = original_df[original_df[target_col] == 0]

    # Robust range for normalization: IQR between 5th and 95th percentiles.
    q05 = train_df[NUM_FEATURES].quantile(0.05)
    q95 = train_df[NUM_FEATURES].quantile(0.95)
    num_ranges = (q95 - q05).replace(0, 1)

    proto_train = {}
    proto_test = {}

    for df, parts in [(train_df, proto_train), (test_df, proto_test)]:
        min_pos, mean3_pos, mean5_pos = _gower_distance_batch(df, orig_pos, num_ranges)
        min_neg, mean3_neg, mean5_neg = _gower_distance_batch(df, orig_neg, num_ranges)

        parts["proto_min_dist_pos"] = min_pos
        parts["proto_mean3_dist_pos"] = mean3_pos
        parts["proto_mean5_dist_pos"] = mean5_pos
        parts["proto_min_dist_neg"] = min_neg
        parts["proto_mean3_dist_neg"] = mean3_neg
        parts["proto_mean5_dist_neg"] = mean5_neg
        parts["proto_margin"] = min_neg - min_pos
        parts["proto_ratio"] = min_pos / (min_neg + 1e-8)
        parts["proto_nearest_label"] = (min_pos < min_neg).astype(int)
        parts["proto_margin_k5"] = mean5_neg - mean5_pos

    return proto_train, proto_test


def build_pairwise_proto(train_df, test_df):
    """
    Build the full pairwise + prototype feature set.

    Combines three feature groups:
      1. Raw features + 5 origstats per feature (78 cols)
      2. Bayesian pairwise target stats (50 cols from 10 pairs)
      3. Gower prototype distances (10 cols)

    Total: ~138 features.
    """
    original_df = load_original_uci()

    # Raw features + 5 UCI stats per feature.
    x_train, x_test = uci_stats_columns(
        train_df, test_df, original_df, ALL_FEATURES,
        stats=("mean", "median", "std", "skew", "count"),
    )

    # Pairwise Bayesian features (requires binned numeric columns).
    train_copy = train_df.copy()
    test_copy = test_df.copy()
    orig_copy = original_df.copy()

    pair_train, pair_test = _build_pairwise_features(
        train_copy, test_copy, orig_copy
    )
    for col_name, values in pair_train.items():
        x_train[col_name] = values
    for col_name, values in pair_test.items():
        x_test[col_name] = values

    # Prototype distance features.
    proto_train, proto_test = _build_prototype_features(
        train_df, test_df, original_df
    )
    for col_name, values in proto_train.items():
        x_train[col_name] = values
    for col_name, values in proto_test.items():
        x_test[col_name] = values

    return x_train, x_test
