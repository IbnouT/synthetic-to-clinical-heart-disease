"""Cross-stacking feature pipeline.

Builds the base feature set for models that incorporate OOF predictions
from other models as an additional input. The OOF column itself is
added per fold in the training loop (see training.py).
"""

from src.config import ALL_FEATURES, CAT_FEATURES
from src.data import load_original_uci
from src.features.helpers import frequency_encode_column, uci_mean_column


def build_cross_stack_lr(train_df, test_df):
    """
    Raw features + UCI target means + categorical frequency for cross-stacking.

    Returns the base feature set. The LR OOF column is injected
    per fold in the training loop.
    """
    original_df = load_original_uci()

    x_train = train_df[ALL_FEATURES].copy()
    x_test = test_df[ALL_FEATURES].copy()

    # UCI target mean per feature value.
    for col in ALL_FEATURES:
        if col not in original_df.columns:
            continue
        stats = original_df.groupby(col)["target"].mean().to_dict()
        x_train[f"{col}_orig"] = train_df[col].map(stats).fillna(0.5)
        x_test[f"{col}_orig"] = test_df[col].map(stats).fillna(0.5)

    # Frequency encoding for categorical features.
    for col in CAT_FEATURES:
        freq_tr, freq_te = frequency_encode_column(
            train_df[col], test_df[col], source="combined"
        )
        x_train[f"{col}_freq"] = freq_tr
        x_test[f"{col}_freq"] = freq_te

    return x_train, x_test


STACKING_BUILDERS = {
    "cross_stack_lr": build_cross_stack_lr,
}
