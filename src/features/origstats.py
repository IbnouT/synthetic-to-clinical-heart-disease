"""Target statistics feature pipelines.

These pipelines enrich the raw clinical features with per-value target
statistics from a reference dataset. The UCI variants use the original
270-row UCI Cleveland dataset; the competition variant uses the 630K
competition dataset for scenarios where UCI data would cause leakage
(external validation on UCI clinical datasets).

Pipelines:
    origstats               65 cols: raw + 4 UCI stats per feature
    origstats_tuned        104 cols: raw + 5 UCI stats + freq + target encoding
    origstats_categorical   65 cols: origstats with categorical dtype for RealMLP
    origstats_mini          52 cols: raw + 3 UCI stats per feature
    origstats_full         ~104 cols: raw + 5 UCI stats + freq + OOF target encoding
    competition_stats       78 cols: raw + 5 competition stats per feature
"""

from src.config import ALL_FEATURES, FILE_ORDER_FEATURES
from src.data import load_original_uci, load_train_test
from src.features.helpers import (
    frequency_encode_column,
    target_encode_oof,
    uci_stats_columns,
)


def build_origstats(train_df, test_df):
    """
    65 columns: 13 raw + 4 UCI stats per feature (mean, median, std, count).

    Stats come from the 270-row original UCI dataset grouped by each
    feature value, giving the model access to real clinical distributions.
    """
    return uci_stats_columns(
        train_df, test_df, load_original_uci(), ALL_FEATURES,
        stats=("mean", "median", "std", "count"),
    )


def build_origstats_tuned(train_df, test_df):
    """
    104 columns: 13 raw (file order) + 5 UCI stats per feature (including
    skew) + 13 train-only frequency + 13 out-of-fold target encoding.
    """
    original_df = load_original_uci()

    # Start with raw features + 5 UCI stats (78 columns).
    x_train, x_test = uci_stats_columns(
        train_df, test_df, original_df, FILE_ORDER_FEATURES,
        stats=("mean", "median", "std", "skew", "count"),
    )

    # Train-only frequency encoding (13 columns).
    for col in FILE_ORDER_FEATURES:
        freq_tr, freq_te = frequency_encode_column(
            train_df[col], test_df[col], source="train"
        )
        x_train[f"{col}_freq"] = freq_tr
        x_test[f"{col}_freq"] = freq_te

    # Out-of-fold target encoding (13 columns).
    y_train = train_df["target"].to_numpy()
    train_te, test_te = target_encode_oof(
        train_df, test_df, FILE_ORDER_FEATURES, y_train,
    )
    for col in FILE_ORDER_FEATURES:
        te_col = f"{col}_te"
        x_train[te_col] = train_te[te_col].values
        x_test[te_col] = test_te[te_col].values

    return x_train, x_test


def build_origstats_categorical(train_df, test_df):
    """Origstats features with all columns typed as categorical for RealMLP.

    RealMLP reads the dtype to choose between learned embeddings (category)
    and piecewise-linear encoding (numeric).
    """
    x_train, x_test = build_origstats(train_df, test_df)

    for col in x_train.columns:
        x_train[col] = x_train[col].astype(str).astype("category")
        x_test[col] = x_test[col].astype(str).astype("category")

    return x_train, x_test


def build_origstats_mini(train_df, test_df):
    """
    52 columns: 13 raw + 3 UCI stats per feature (mean, std, count).

    Lighter version of origstats used for multi-seed ensemble diversity.
    """
    return uci_stats_columns(
        train_df, test_df, load_original_uci(), ALL_FEATURES,
        stats=("mean", "std", "count"),
    )


def build_origstats_full(train_df, test_df):
    """
    Build the full origstats feature set for stacked models.

    Combines raw features, 5 UCI statistics per feature, frequency
    encoding, and OOF target encoding into ~104 columns. Used by the
    cb_origstats_f10 stacked model family.
    """
    original_df = load_original_uci()
    y = train_df["target"].values

    # Raw features + 5 UCI statistics (78 columns).
    x_train, x_test = uci_stats_columns(
        train_df, test_df, original_df, ALL_FEATURES,
        stats=("mean", "median", "std", "skew", "count"),
    )

    # Frequency encoding from training data (13 columns).
    for col in ALL_FEATURES:
        freq_tr, freq_te = frequency_encode_column(
            train_df[col], test_df[col], source="train"
        )
        x_train[f"{col}_freq"] = freq_tr.values
        x_test[f"{col}_freq"] = freq_te.values

    # OOF target encoding (13 columns).
    train_te, test_te = target_encode_oof(train_df, test_df, ALL_FEATURES, y)
    for col in ALL_FEATURES:
        te_col = f"{col}_te"
        x_train[te_col] = train_te[te_col].values
        x_test[te_col] = test_te[te_col].values

    return x_train, x_test


def build_competition_stats(train_df, test_df):
    """
    78 columns: 13 raw + 5 competition stats per feature.

    Same approach as origstats but uses the 630K competition dataset
    as the reference instead of the 270-row UCI original. This avoids
    leakage when evaluating on UCI clinical datasets, since the UCI
    datasets are subsets of the original data used to generate the
    competition dataset.

    Used for external validation scenarios S2 and S4 where competition-
    derived statistics are applied to UCI clinical samples.

    Uses FILE_ORDER_FEATURES for column ordering to ensure consistent
    tree model results across runs.
    """
    comp_train, _ = load_train_test()
    comp_ref = comp_train[FILE_ORDER_FEATURES].copy()
    comp_ref["target"] = comp_train["target"].values

    return uci_stats_columns(
        train_df, test_df, comp_ref, FILE_ORDER_FEATURES,
        stats=("mean", "median", "std", "skew", "count"),
        prefix="comp",
    )


ORIGSTATS_BUILDERS = {
    "origstats": build_origstats,
    "origstats_tuned": build_origstats_tuned,
    "origstats_categorical": build_origstats_categorical,
    "origstats_mini": build_origstats_mini,
    "origstats_full": build_origstats_full,
    "competition_stats": build_competition_stats,
}
