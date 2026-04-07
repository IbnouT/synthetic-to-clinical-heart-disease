"""Gradient-boosted tree model training (CatBoost, XGBoost, LightGBM).

Each function trains one CV fold: builds the model, fits with eval_set
and early stopping, and returns validation and test predictions.
When x_va is None (e.g. zero-shot transfer on full training set),
the model trains without early stopping and no validation scores
are returned. When x_te is None (e.g. UCI within-dataset CV),
test predictions are skipped.
"""

import pandas as pd
from catboost import CatBoostClassifier, Pool
import lightgbm as lgb
from lightgbm import early_stopping as lgb_early_stopping
from xgboost import XGBClassifier


# ---------------------------------------------------------------------------
# CatBoost
# ---------------------------------------------------------------------------

def train_fold_catboost(x_tr, y_tr, x_va=None, y_va=None, x_te=None, config=None, seed=42, fold_idx=0):
    """
    Train CatBoost on one fold with optional categorical feature handling.

    When cat_features is set in config, data is wrapped in CatBoost Pool
    objects and the model uses use_best_model=True for early stopping.
    When x_va is None, trains on full data without early stopping.
    """
    params = dict(config["params"])
    cat_features = config.get("cat_features", None)

    if cat_features is not None:
        params["cat_features"] = cat_features

    model = CatBoostClassifier(random_seed=seed, **params)

    if x_va is not None and y_va is not None:
        if cat_features is not None:
            train_pool = Pool(x_tr, y_tr, cat_features=cat_features)
            val_pool = Pool(x_va, y_va, cat_features=cat_features)
            model.fit(train_pool, eval_set=val_pool, use_best_model=True)
        else:
            model.fit(x_tr, y_tr, eval_set=(x_va, y_va), verbose=False)
    else:
        if cat_features is not None:
            train_pool = Pool(x_tr, y_tr, cat_features=cat_features)
            model.fit(train_pool, verbose=False)
        else:
            model.fit(x_tr, y_tr, verbose=False)

    val_scores = model.predict_proba(x_va)[:, 1] if x_va is not None else None
    test_scores = model.predict_proba(x_te)[:, 1] if x_te is not None else None
    return val_scores, test_scores, model


# ---------------------------------------------------------------------------
# XGBoost
# ---------------------------------------------------------------------------

def train_fold_xgboost(x_tr, y_tr, x_va=None, y_va=None, x_te=None, config=None, seed=42, fold_idx=0):
    """Train XGBoost on one fold with optional eval_set early stopping."""
    model = XGBClassifier(random_state=seed, **config["params"])

    if x_va is not None and y_va is not None:
        model.fit(x_tr, y_tr, eval_set=[(x_va, y_va)], verbose=False)
    else:
        model.set_params(early_stopping_rounds=None)
        model.fit(x_tr, y_tr, verbose=False)

    val_scores = model.predict_proba(x_va)[:, 1] if x_va is not None else None
    test_scores = model.predict_proba(x_te)[:, 1] if x_te is not None else None
    return val_scores, test_scores, model


# ---------------------------------------------------------------------------
# LightGBM (sklearn API)
# ---------------------------------------------------------------------------

def train_fold_lgb(x_tr, y_tr, x_va=None, y_va=None, x_te=None, config=None, seed=42, fold_idx=0):
    """Train LightGBM (sklearn API) on one fold with optional early stopping.

    Early stopping monitors the metric specified by eval_metric in config.
    If not set, LightGBM defaults to the objective's native metric (logloss
    for binary classification). When x_va is None, trains without early
    stopping.
    """
    stop_rounds = config["params"].get("early_stopping_rounds", 100)
    eval_metric = config.get("eval_metric", None)
    model = lgb.LGBMClassifier(random_state=seed, **config["params"])

    if x_va is not None and y_va is not None:
        fit_kwargs = {
            "eval_set": [(x_va, y_va)],
            "callbacks": [lgb_early_stopping(stop_rounds)],
        }
        if eval_metric is not None:
            fit_kwargs["eval_metric"] = eval_metric
        model.fit(x_tr, y_tr, **fit_kwargs)
    else:
        model.fit(x_tr, y_tr)

    val_scores = model.predict_proba(x_va)[:, 1] if x_va is not None else None
    test_scores = model.predict_proba(x_te)[:, 1] if x_te is not None else None
    return val_scores, test_scores, model


# ---------------------------------------------------------------------------
# LightGBM (native API)
# ---------------------------------------------------------------------------

def train_fold_lgb_native(x_tr, y_tr, x_va=None, y_va=None, x_te=None, config=None, seed=42, fold_idx=0):
    """Train LightGBM using the native lgb.train() API with early stopping."""
    if isinstance(x_tr, pd.DataFrame):
        feat_names = list(x_tr.columns)
        x_tr_np = x_tr.to_numpy()
        x_va_np = x_va.to_numpy() if x_va is not None else None
        x_te_np = x_te.to_numpy() if x_te is not None else None
    else:
        feat_names = None
        x_tr_np = x_tr
        x_va_np = x_va
        x_te_np = x_te

    dtrain = lgb.Dataset(x_tr_np, label=y_tr, feature_name=feat_names,
                         free_raw_data=False)

    num_rounds = config.get("num_boost_round", 10000)
    stop_rounds = config.get("early_stopping_rounds", 200)

    callbacks = [lgb.log_evaluation(period=0)]
    valid_sets = []
    if x_va_np is not None and y_va is not None:
        dval = lgb.Dataset(x_va_np, label=y_va, feature_name=feat_names,
                           free_raw_data=False)
        valid_sets = [dval]
        callbacks.append(lgb.early_stopping(stopping_rounds=stop_rounds))

    model = lgb.train(
        config["params"],
        dtrain,
        num_boost_round=num_rounds,
        valid_sets=valid_sets,
        callbacks=callbacks,
    )

    val_scores = model.predict(x_va_np) if x_va_np is not None else None
    test_scores = model.predict(x_te_np) if x_te_np is not None else None
    return val_scores, test_scores, model


# ---------------------------------------------------------------------------
# Dispatcher - maps family names to fold training functions
# ---------------------------------------------------------------------------

FOLD_TRAINERS = {
    "catboost": train_fold_catboost,
    "xgboost": train_fold_xgboost,
    "lightgbm": train_fold_lgb,
    "lightgbm_native": train_fold_lgb_native,
}
