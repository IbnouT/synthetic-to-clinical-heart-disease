"""Tree ensemble model training (Random Forest, Extra Trees)."""

from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier


def _build_tree_model(family, seed, params):
    """Create the right tree ensemble based on family name."""
    if family == "extra_trees":
        return ExtraTreesClassifier(random_state=seed, **params)
    return RandomForestClassifier(random_state=seed, **params)


def train_fold_trees(x_tr, y_tr, x_va=None, y_va=None, x_te=None, config=None, seed=42, fold_idx=0):
    """Train a tree ensemble on one fold and return val/test predictions."""
    model = _build_tree_model(config["family"], seed, config["params"])
    model.fit(x_tr, y_tr)

    val_scores = model.predict_proba(x_va)[:, 1] if x_va is not None else None
    test_scores = model.predict_proba(x_te)[:, 1] if x_te is not None else None
    return val_scores, test_scores, model


# Maps both tree family names to the same trainer function.
FOLD_TRAINERS = {
    "random_forest": train_fold_trees,
    "extra_trees": train_fold_trees,
}
