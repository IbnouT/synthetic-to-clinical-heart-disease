"""Model registry for external validation experiments.

Provides a uniform interface for all 14 model families used in the
within-dataset and zero-shot transfer evaluations. Each family exposes:
  - get_default_params(): tuned defaults from the original experiments
  - get_optuna_space(trial): Optuna search space for dataset-tuning
  - build_model(params): construct an unfitted model or pipeline
  - train(model, X, y, X_val, y_val): fit with optional early stopping
  - predict(model, X): return positive-class probability scores

Models that need feature scaling (LR, Ridge, Lasso, ElasticNet, SVM,
KNN, MLP) wrap the estimator in a StandardScaler pipeline. Gradient
boosters (CatBoost, XGBoost, LightGBM) handle their own eval_set
early stopping. TabPFN uses subsample ensembling for datasets larger
than its context window.
"""
from __future__ import annotations

import gc
import logging
import numpy as np

from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import torch  # must be imported before CatBoost on Python 3.14
import lightgbm as lgb
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

# Pretrained tabular transformer and tuned feed-forward network
from tabpfn import TabPFNClassifier
from pytabkit.models.sklearn.sklearn_interfaces import RealMLP_TD_Classifier

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------
# Shared utilities
# -----------------------------------------------------------------------

def _positive_class_scores(model: Any, X: Any) -> np.ndarray:
    """Extract positive-class probabilities from a fitted model or pipeline.

    Handles 1D output (some wrappers), single-column output, and the
    standard 2-column predict_proba format.
    """
    proba = np.asarray(model.predict_proba(X), dtype=np.float64)
    if proba.ndim == 1:
        return np.clip(proba.reshape(-1), 0.0, 1.0)
    if proba.shape[1] == 1:
        return np.clip(proba[:, 0], 0.0, 1.0)
    return np.clip(proba[:, -1], 0.0, 1.0)


def _build_scaled_pipeline(estimator_class, params: Dict[str, Any]):
    """Wrap an sklearn estimator in a StandardScaler pipeline.

    Pops scaler_with_mean and scaler_with_std from params before
    constructing the estimator (these are pipeline-level settings,
    not estimator hyperparameters).
    """
    params = dict(params)
    with_mean = bool(params.pop("scaler_with_mean", True))
    with_std = bool(params.pop("scaler_with_std", True))
    return Pipeline([
        ("scaler", StandardScaler(with_mean=with_mean, with_std=with_std)),
        ("model", estimator_class(**params)),
    ])


# -----------------------------------------------------------------------
# CatBoost
# -----------------------------------------------------------------------

class _CatBoost:
    @staticmethod
    def get_default_params() -> Dict[str, Any]:
        return {
            "iterations": 1500,
            "learning_rate": 0.03,
            "depth": 6,
            "l2_leaf_reg": 3.0,
            "loss_function": "Logloss",
            "eval_metric": "AUC",
            "random_seed": 42,
            "verbose": False,
            "allow_writing_files": False,
            "thread_count": -1,
        }

    @staticmethod
    def get_optuna_space(trial) -> Dict[str, Any]:
        bootstrap = trial.suggest_categorical(
            "bootstrap_type", ["Bayesian", "Bernoulli", "MVS"])
        space = {
            "iterations": trial.suggest_int("iterations", 400, 4000),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.2, log=True),
            "depth": trial.suggest_int("depth", 3, 10),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-3, 100.0, log=True),
            "bootstrap_type": bootstrap,
            "random_strength": trial.suggest_float("random_strength", 1e-3, 10.0, log=True),
            "early_stopping_rounds": trial.suggest_int("early_stopping_rounds", 50, 300),
        }
        # bagging_temperature only valid with Bayesian bootstrap
        if bootstrap == "Bayesian":
            space["bagging_temperature"] = trial.suggest_float(
                "bagging_temperature", 0.0, 10.0)
        # subsample only valid with Bernoulli and MVS
        if bootstrap in ("Bernoulli", "MVS"):
            space["subsample"] = trial.suggest_float("subsample", 0.5, 1.0)
        return space

    @staticmethod
    def build_model(params: Dict[str, Any]) -> CatBoostClassifier:
        merged = _CatBoost.get_default_params()
        merged.update(params)
        return CatBoostClassifier(**merged)

    @staticmethod
    def train(model, X_train, y_train, X_val=None, y_val=None):
        fit_kwargs = {}
        if X_val is not None and y_val is not None:
            fit_kwargs["eval_set"] = (X_val, y_val)
            explicit_use_best = model.get_params().get("use_best_model", None)
            fit_kwargs["use_best_model"] = (
                True if explicit_use_best is None else bool(explicit_use_best)
            )
        model.fit(X_train, y_train, **fit_kwargs)
        return model

    @staticmethod
    def predict(model, X) -> np.ndarray:
        return _positive_class_scores(model, X)


# -----------------------------------------------------------------------
# XGBoost
# -----------------------------------------------------------------------

class _XGBoost:
    @staticmethod
    def get_default_params() -> Dict[str, Any]:
        return {
            "n_estimators": 1200,
            "learning_rate": 0.03,
            "max_depth": 6,
            "min_child_weight": 1.0,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.0,
            "reg_lambda": 1.0,
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "random_state": 42,
            "n_jobs": -1,
            "tree_method": "hist",
            "verbosity": 0,
        }

    @staticmethod
    def get_optuna_space(trial) -> Dict[str, Any]:
        return {
            "n_estimators": trial.suggest_int("n_estimators", 300, 4000),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.2, log=True),
            "max_depth": trial.suggest_int("max_depth", 2, 12),
            "min_child_weight": trial.suggest_float("min_child_weight", 1e-2, 50.0, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "gamma": trial.suggest_float("gamma", 1e-8, 10.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 100.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 100.0, log=True),
            "early_stopping_rounds": trial.suggest_int("early_stopping_rounds", 50, 300),
        }

    @staticmethod
    def build_model(params: Dict[str, Any]) -> XGBClassifier:
        merged = _XGBoost.get_default_params()
        merged.update(params)
        return XGBClassifier(**merged)

    @staticmethod
    def train(model, X_train, y_train, X_val=None, y_val=None):
        fit_kwargs = {}
        if X_val is not None and y_val is not None:
            fit_kwargs["eval_set"] = [(X_val, y_val)]
            fit_kwargs["verbose"] = False
        else:
            # Disable early stopping when there is no validation set
            if getattr(model, "early_stopping_rounds", None):
                model.set_params(early_stopping_rounds=None)
        model.fit(X_train, y_train, **fit_kwargs)
        return model

    @staticmethod
    def predict(model, X) -> np.ndarray:
        return _positive_class_scores(model, X)


# -----------------------------------------------------------------------
# LightGBM
# -----------------------------------------------------------------------

class _LightGBM:
    @staticmethod
    def get_default_params() -> Dict[str, Any]:
        return {
            "n_estimators": 1200,
            "learning_rate": 0.03,
            "num_leaves": 63,
            "max_depth": -1,
            "min_child_samples": 20,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.0,
            "reg_lambda": 1.0,
            "objective": "binary",
            "random_state": 42,
            "n_jobs": -1,
            "verbosity": -1,
        }

    @staticmethod
    def get_optuna_space(trial) -> Dict[str, Any]:
        return {
            "n_estimators": trial.suggest_int("n_estimators", 300, 4000),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.2, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 16, 512),
            "max_depth": trial.suggest_int("max_depth", -1, 16),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 100.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 100.0, log=True),
        }

    @staticmethod
    def build_model(params: Dict[str, Any]) -> lgb.LGBMClassifier:
        merged = _LightGBM.get_default_params()
        merged.update(params)
        # early_stopping_rounds is handled in the fit call, not the constructor
        merged.pop("early_stopping_rounds", None)
        return lgb.LGBMClassifier(**merged)

    @staticmethod
    def train(model, X_train, y_train, X_val=None, y_val=None):
        fit_kwargs = {}
        if X_val is not None and y_val is not None:
            fit_kwargs["eval_set"] = [(X_val, y_val)]
            fit_kwargs["callbacks"] = [
                lgb.early_stopping(stopping_rounds=100, verbose=False),
            ]
        model.fit(X_train, y_train, **fit_kwargs)
        return model

    @staticmethod
    def predict(model, X) -> np.ndarray:
        return _positive_class_scores(model, X)


# -----------------------------------------------------------------------
# Random Forest
# -----------------------------------------------------------------------

class _RandomForest:
    @staticmethod
    def get_default_params() -> Dict[str, Any]:
        return {
            "n_estimators": 600,
            "max_depth": None,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "max_features": "sqrt",
            "bootstrap": True,
            "class_weight": None,
            "random_state": 42,
            "n_jobs": -1,
        }

    @staticmethod
    def get_optuna_space(trial) -> Dict[str, Any]:
        return {
            "n_estimators": trial.suggest_int("n_estimators", 200, 2000),
            "max_depth": trial.suggest_int("max_depth", 3, 40),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
            "bootstrap": trial.suggest_categorical("bootstrap", [True, False]),
            "class_weight": trial.suggest_categorical("class_weight", [None, "balanced"]),
        }

    @staticmethod
    def build_model(params: Dict[str, Any]) -> RandomForestClassifier:
        merged = _RandomForest.get_default_params()
        merged.update(params)
        return RandomForestClassifier(**merged)

    @staticmethod
    def train(model, X_train, y_train, X_val=None, y_val=None):
        model.fit(X_train, y_train)
        return model

    @staticmethod
    def predict(model, X) -> np.ndarray:
        return _positive_class_scores(model, X)


# -----------------------------------------------------------------------
# Extra Trees
# -----------------------------------------------------------------------

class _ExtraTrees:
    @staticmethod
    def get_default_params() -> Dict[str, Any]:
        return {
            "n_estimators": 800,
            "max_depth": None,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "max_features": "sqrt",
            "bootstrap": False,
            "class_weight": None,
            "random_state": 42,
            "n_jobs": -1,
        }

    @staticmethod
    def get_optuna_space(trial) -> Dict[str, Any]:
        return {
            "n_estimators": trial.suggest_int("n_estimators", 200, 2500),
            "max_depth": trial.suggest_int("max_depth", 3, 40),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
            "bootstrap": trial.suggest_categorical("bootstrap", [True, False]),
            "class_weight": trial.suggest_categorical("class_weight", [None, "balanced"]),
        }

    @staticmethod
    def build_model(params: Dict[str, Any]) -> ExtraTreesClassifier:
        merged = _ExtraTrees.get_default_params()
        merged.update(params)
        return ExtraTreesClassifier(**merged)

    @staticmethod
    def train(model, X_train, y_train, X_val=None, y_val=None):
        model.fit(X_train, y_train)
        return model

    @staticmethod
    def predict(model, X) -> np.ndarray:
        return _positive_class_scores(model, X)


# -----------------------------------------------------------------------
# Logistic Regression
# -----------------------------------------------------------------------

class _LogisticRegression:
    @staticmethod
    def get_default_params() -> Dict[str, Any]:
        return {
            "penalty": "l2",
            "C": 1.0,
            "solver": "lbfgs",
            "max_iter": 2000,
            "class_weight": None,
            "random_state": 42,
            "n_jobs": -1,
            "scaler_with_mean": True,
            "scaler_with_std": True,
        }

    @staticmethod
    def get_optuna_space(trial) -> Dict[str, Any]:
        penalty = trial.suggest_categorical("penalty", ["l1", "l2", "elasticnet"])
        if penalty == "l1":
            solver = trial.suggest_categorical("solver", ["liblinear", "saga"])
            l1_ratio = None
        elif penalty == "elasticnet":
            solver = "saga"
            l1_ratio = trial.suggest_float("l1_ratio", 0.01, 0.99)
        else:
            solver = trial.suggest_categorical("solver", ["lbfgs", "liblinear", "saga"])
            l1_ratio = None
        space = {
            "penalty": penalty,
            "solver": solver,
            "C": trial.suggest_float("C", 1e-4, 100.0, log=True),
            "max_iter": trial.suggest_int("max_iter", 1000, 5000),
            "class_weight": trial.suggest_categorical("class_weight", [None, "balanced"]),
        }
        if l1_ratio is not None:
            space["l1_ratio"] = l1_ratio
        return space

    @staticmethod
    def build_model(params: Dict[str, Any]) -> Pipeline:
        merged = _LogisticRegression.get_default_params()
        merged.update(params)
        return _build_scaled_pipeline(LogisticRegression, merged)

    @staticmethod
    def train(model, X_train, y_train, X_val=None, y_val=None):
        model.fit(X_train, y_train)
        return model

    @staticmethod
    def predict(model, X) -> np.ndarray:
        return _positive_class_scores(model, X)


# -----------------------------------------------------------------------
# Ridge Classifier
# -----------------------------------------------------------------------

class _Ridge:
    @staticmethod
    def get_default_params() -> Dict[str, Any]:
        return {
            "alpha": 1.0,
            "solver": "auto",
            "fit_intercept": True,
            "tol": 1e-4,
            "class_weight": None,
            "random_state": 42,
            "scaler_with_mean": True,
            "scaler_with_std": True,
        }

    @staticmethod
    def get_optuna_space(trial) -> Dict[str, Any]:
        return {
            "alpha": trial.suggest_float("alpha", 1e-4, 100.0, log=True),
            "solver": trial.suggest_categorical(
                "solver", ["auto", "svd", "cholesky", "lsqr", "sag", "saga"]),
            "tol": trial.suggest_float("tol", 1e-6, 1e-2, log=True),
            "class_weight": trial.suggest_categorical("class_weight", [None, "balanced"]),
        }

    @staticmethod
    def build_model(params: Dict[str, Any]) -> Pipeline:
        merged = _Ridge.get_default_params()
        merged.update(params)
        return _build_scaled_pipeline(RidgeClassifier, merged)

    @staticmethod
    def train(model, X_train, y_train, X_val=None, y_val=None):
        model.fit(X_train, y_train)
        return model

    @staticmethod
    def predict(model, X) -> np.ndarray:
        """Ridge has no predict_proba, so we sigmoid the decision function."""
        raw = np.asarray(model.decision_function(X), dtype=np.float64).reshape(-1)
        return 1.0 / (1.0 + np.exp(-raw))


# -----------------------------------------------------------------------
# Lasso (L1-regularized logistic regression)
# -----------------------------------------------------------------------

class _Lasso:
    @staticmethod
    def get_default_params() -> Dict[str, Any]:
        return {
            "penalty": "l1",
            "C": 1.0,
            "solver": "saga",
            "max_iter": 4000,
            "class_weight": None,
            "random_state": 42,
            "n_jobs": -1,
            "scaler_with_mean": True,
            "scaler_with_std": True,
        }

    @staticmethod
    def get_optuna_space(trial) -> Dict[str, Any]:
        return {
            "penalty": "l1",
            "solver": trial.suggest_categorical("solver", ["liblinear", "saga"]),
            "C": trial.suggest_float("C", 1e-4, 100.0, log=True),
            "max_iter": trial.suggest_int("max_iter", 1500, 6000),
            "class_weight": trial.suggest_categorical("class_weight", [None, "balanced"]),
        }

    @staticmethod
    def build_model(params: Dict[str, Any]) -> Pipeline:
        merged = _Lasso.get_default_params()
        merged.update(params)
        return _build_scaled_pipeline(LogisticRegression, merged)

    @staticmethod
    def train(model, X_train, y_train, X_val=None, y_val=None):
        model.fit(X_train, y_train)
        return model

    @staticmethod
    def predict(model, X) -> np.ndarray:
        return _positive_class_scores(model, X)


# -----------------------------------------------------------------------
# Elastic Net (elasticnet-regularized logistic regression)
# -----------------------------------------------------------------------

class _ElasticNet:
    @staticmethod
    def get_default_params() -> Dict[str, Any]:
        return {
            "penalty": "elasticnet",
            "C": 1.0,
            "l1_ratio": 0.5,
            "solver": "saga",
            "max_iter": 4000,
            "class_weight": None,
            "random_state": 42,
            "n_jobs": -1,
            "scaler_with_mean": True,
            "scaler_with_std": True,
        }

    @staticmethod
    def get_optuna_space(trial) -> Dict[str, Any]:
        return {
            "penalty": "elasticnet",
            "solver": "saga",
            "C": trial.suggest_float("C", 1e-4, 100.0, log=True),
            "l1_ratio": trial.suggest_float("l1_ratio", 0.01, 0.99),
            "max_iter": trial.suggest_int("max_iter", 1500, 6000),
            "class_weight": trial.suggest_categorical("class_weight", [None, "balanced"]),
        }

    @staticmethod
    def build_model(params: Dict[str, Any]) -> Pipeline:
        merged = _ElasticNet.get_default_params()
        merged.update(params)
        return _build_scaled_pipeline(LogisticRegression, merged)

    @staticmethod
    def train(model, X_train, y_train, X_val=None, y_val=None):
        model.fit(X_train, y_train)
        return model

    @staticmethod
    def predict(model, X) -> np.ndarray:
        return _positive_class_scores(model, X)


# -----------------------------------------------------------------------
# SVM (SVC with probability calibration via Platt scaling)
# -----------------------------------------------------------------------

class _SVM:
    @staticmethod
    def get_default_params() -> Dict[str, Any]:
        return {
            "C": 1.0,
            "kernel": "rbf",
            "gamma": "scale",
            "degree": 3,
            "probability": True,
            "class_weight": None,
            "random_state": 42,
            "max_train_samples": 50000,
            "scaler_with_mean": True,
            "scaler_with_std": True,
        }

    @staticmethod
    def get_optuna_space(trial) -> Dict[str, Any]:
        kernel = trial.suggest_categorical("kernel", ["linear", "rbf", "poly", "sigmoid"])
        space = {
            "C": trial.suggest_float("C", 1e-3, 100.0, log=True),
            "kernel": kernel,
            "probability": True,
            "class_weight": trial.suggest_categorical("class_weight", [None, "balanced"]),
        }
        if kernel in {"rbf", "poly", "sigmoid"}:
            space["gamma"] = trial.suggest_float("gamma", 1e-4, 10.0, log=True)
        if kernel == "poly":
            space["degree"] = trial.suggest_int("degree", 2, 5)
        return space

    @staticmethod
    def build_model(params: Dict[str, Any]) -> Pipeline:
        merged = _SVM.get_default_params()
        merged.update(params)
        merged["probability"] = True
        # max_train_samples is our custom guard, not an SVC param
        max_train_samples = int(merged.pop("max_train_samples", 50000))
        pipeline = _build_scaled_pipeline(SVC, merged)
        pipeline._max_train_samples = max_train_samples
        return pipeline

    @staticmethod
    def train(model, X_train, y_train, X_val=None, y_val=None):
        max_n = getattr(model, "_max_train_samples", 50000)
        n = X_train.shape[0] if hasattr(X_train, "shape") else len(X_train)
        if n > max_n:
            rng = np.random.RandomState(42)
            idx = rng.choice(n, size=max_n, replace=False)
            idx.sort()
            X_train = X_train.iloc[idx] if hasattr(X_train, "iloc") else X_train[idx]
            y_train = y_train[idx]
            logger.warning(
                "SVM: subsampled training set %d -> %d rows", n, max_n)
        model.fit(X_train, y_train)
        return model

    @staticmethod
    def predict(model, X) -> np.ndarray:
        return _positive_class_scores(model, X)


# -----------------------------------------------------------------------
# K-Nearest Neighbors
# -----------------------------------------------------------------------

class _KNN:
    @staticmethod
    def get_default_params() -> Dict[str, Any]:
        return {
            "n_neighbors": 15,
            "weights": "distance",
            "metric": "minkowski",
            "p": 2,
            "n_jobs": -1,
            "scaler_with_mean": True,
            "scaler_with_std": True,
        }

    @staticmethod
    def get_optuna_space(trial) -> Dict[str, Any]:
        metric = trial.suggest_categorical("metric", ["minkowski", "manhattan", "euclidean"])
        return {
            "n_neighbors": trial.suggest_int("n_neighbors", 3, 151),
            "weights": trial.suggest_categorical("weights", ["uniform", "distance"]),
            "metric": metric,
            "p": trial.suggest_int("p", 1, 3) if metric == "minkowski" else 2,
        }

    @staticmethod
    def build_model(params: Dict[str, Any]) -> Pipeline:
        merged = _KNN.get_default_params()
        merged.update(params)
        return _build_scaled_pipeline(KNeighborsClassifier, merged)

    @staticmethod
    def train(model, X_train, y_train, X_val=None, y_val=None):
        model.fit(X_train, y_train)
        return model

    @staticmethod
    def predict(model, X) -> np.ndarray:
        return _positive_class_scores(model, X)


# -----------------------------------------------------------------------
# MLP (sklearn MLPClassifier, not the PyTorch version)
# -----------------------------------------------------------------------

class _MLP:
    @staticmethod
    def get_default_params() -> Dict[str, Any]:
        return {
            "hidden_layer_sizes": (128, 64),
            "activation": "relu",
            "solver": "adam",
            "alpha": 1e-4,
            "learning_rate": "adaptive",
            "learning_rate_init": 1e-3,
            "batch_size": 256,
            "max_iter": 500,
            "early_stopping": True,
            "n_iter_no_change": 20,
            "random_state": 42,
            "scaler_with_mean": True,
            "scaler_with_std": True,
        }

    @staticmethod
    def get_optuna_space(trial) -> Dict[str, Any]:
        return {
            "hidden_layer_sizes": trial.suggest_categorical(
                "hidden_layer_sizes",
                [(64,), (128,), (256,), (128, 64), (256, 128), (256, 128, 64)]),
            "activation": trial.suggest_categorical("activation", ["relu", "tanh"]),
            "alpha": trial.suggest_float("alpha", 1e-6, 1e-1, log=True),
            "learning_rate_init": trial.suggest_float("learning_rate_init", 1e-4, 1e-1, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [64, 128, 256, 512]),
            "max_iter": trial.suggest_int("max_iter", 200, 1000),
        }

    @staticmethod
    def build_model(params: Dict[str, Any]) -> Pipeline:
        merged = _MLP.get_default_params()
        merged.update(params)
        return _build_scaled_pipeline(MLPClassifier, merged)

    @staticmethod
    def train(model, X_train, y_train, X_val=None, y_val=None):
        model.fit(X_train, y_train)
        return model

    @staticmethod
    def predict(model, X) -> np.ndarray:
        return _positive_class_scores(model, X)


# -----------------------------------------------------------------------
# TabPFN (pretrained transformer with subsample ensembling)
# -----------------------------------------------------------------------

class _TabPFN:
    """TabPFN wrapper with subsample ensembling for large training sets.

    TabPFN has a context window limit (~10K samples). For datasets
    larger than the subsample threshold, we draw multiple stratified
    subsamples from the training set, fit a separate TabPFN on each,
    and average the predictions.
    """
    _SUBSAMPLE_THRESHOLD = 10000
    _DEFAULT_SUBSAMPLE_SIZE = 3000
    _DEFAULT_N_SUBSAMPLES = 8
    _BATCH_SIZE = 5000

    @staticmethod
    def get_default_params() -> Dict[str, Any]:
        return {
            "n_estimators": 4,
            "random_state": 42,
            "ignore_pretraining_limits": True,
            "device": "cpu",
            "subsample_size": 3000,
            "n_subsamples": 8,
        }

    @staticmethod
    def get_optuna_space(trial) -> Dict[str, Any]:
        return {
            "n_estimators": trial.suggest_categorical(
                "n_estimators", [4, 8, 16, 32]),
            "subsample_size": trial.suggest_categorical(
                "subsample_size", [1000, 3000, 5000, 10000]),
            "n_subsamples": trial.suggest_categorical(
                "n_subsamples", [4, 8, 16]),
        }

    @staticmethod
    def build_model(params: Dict[str, Any]):
        """Return a dict of TabPFN config (actual model built at train time)."""
        merged = _TabPFN.get_default_params()
        merged.update(params)
        return {"_tabpfn_config": merged}

    @staticmethod
    def train(model_dict, X_train, y_train, X_val=None, y_val=None):
        """Store training data for prediction-time subsample ensembling."""
        cfg = model_dict["_tabpfn_config"]
        X_arr = X_train.values if hasattr(X_train, "values") else np.asarray(X_train)
        y_arr = np.asarray(y_train)
        model_dict["_X_train"] = X_arr
        model_dict["_y_train"] = y_arr
        model_dict["_use_sub"] = len(y_arr) > _TabPFN._SUBSAMPLE_THRESHOLD
        return model_dict

    @staticmethod
    def predict(model_dict, X) -> np.ndarray:
        cfg = model_dict["_tabpfn_config"]
        X_arr = X.values if hasattr(X, "values") else np.asarray(X, dtype=np.float32)
        X_train = model_dict["_X_train"]
        y_train = model_dict["_y_train"]

        sub_size = int(cfg.get("subsample_size", 3000))
        n_sub = int(cfg.get("n_subsamples", 8))
        use_sub = model_dict.get("_use_sub", False)

        tabpfn_params = {
            k: v for k, v in cfg.items()
            if k not in ("subsample_size", "n_subsamples")
        }

        if not use_sub:
            # Small dataset: fit on full training set
            clf = TabPFNClassifier(**tabpfn_params)
            clf.fit(X_train, y_train)
            proba = clf.predict_proba(X_arr)
            return proba[:, 1] if proba.ndim == 2 else proba

        # Large dataset: subsample ensembling
        rng = np.random.RandomState(cfg.get("random_state", 42))
        all_preds = []

        for i in range(n_sub):
            idx = _stratified_subsample(y_train, sub_size, rng)
            clf = TabPFNClassifier(**tabpfn_params)
            clf.fit(X_train[idx], y_train[idx])

            # Batched prediction to control memory
            preds = []
            for start in range(0, len(X_arr), _TabPFN._BATCH_SIZE):
                chunk = X_arr[start:start + _TabPFN._BATCH_SIZE]
                p = clf.predict_proba(chunk)
                preds.append(p[:, 1] if p.ndim == 2 else p)
            all_preds.append(np.concatenate(preds))

            del clf
            gc.collect()
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()

        return np.mean(all_preds, axis=0)


def _stratified_subsample(y, n, rng):
    """Draw a stratified subsample of size n preserving class proportions."""
    classes, counts = np.unique(y, return_counts=True)
    fracs = counts / counts.sum()
    indices = []
    for cls, frac in zip(classes, fracs):
        cls_idx = np.where(y == cls)[0]
        n_cls = max(1, int(round(n * frac)))
        chosen = rng.choice(cls_idx, size=min(n_cls, len(cls_idx)), replace=False)
        indices.append(chosen)
    return np.concatenate(indices)


# -----------------------------------------------------------------------
# RealMLP (pytabkit)
# -----------------------------------------------------------------------

class _RealMLP:
    @staticmethod
    def get_default_params() -> Dict[str, Any]:
        return {
            "n_epochs": 100,
            "batch_size": 128,
            "lr": 0.04,
            "p_drop": 0.073,
            "hidden_sizes": "rectangular",
            "hidden_width": 384,
            "n_ens": 2,
            "use_early_stopping": True,
            "early_stopping_additive_patience": 20,
            "early_stopping_multiplicative_patience": 1,
            "device": "cpu",
            "random_state": 42,
        }

    @staticmethod
    def get_optuna_space(trial) -> Dict[str, Any]:
        return {
            "hidden_width": trial.suggest_categorical(
                "hidden_width", [128, 256, 384, 512]),
            "hidden_sizes": trial.suggest_categorical(
                "hidden_sizes", ["rectangular", "funnel"]),
            "p_drop": trial.suggest_float("p_drop", 0.0, 0.5),
            "lr": trial.suggest_float("lr", 1e-5, 1e-1, log=True),
            "batch_size": trial.suggest_categorical(
                "batch_size", [128, 256, 512, 1024]),
            "n_epochs": trial.suggest_int("n_epochs", 20, 300),
            "n_ens": trial.suggest_categorical("n_ens", [1, 2, 4]),
        }

    @staticmethod
    def build_model(params: Dict[str, Any]):
        merged = _RealMLP.get_default_params()
        merged.update(params)
        # Convert list hidden_sizes to string format (pytabkit convention)
        hs = merged.get("hidden_sizes")
        if isinstance(hs, list):
            merged["hidden_width"] = max(hs)
            merged["hidden_sizes"] = "rectangular"
        return RealMLP_TD_Classifier(**merged)

    @staticmethod
    def train(model, X_train, y_train, X_val=None, y_val=None):
        if X_val is not None and y_val is not None:
            model.fit(X_train, y_train, X_val, y_val)
        else:
            model.fit(X_train, y_train)
        return model

    @staticmethod
    def predict(model, X) -> np.ndarray:
        return _positive_class_scores(model, X)


# -----------------------------------------------------------------------
# Registry lookup
# -----------------------------------------------------------------------

_REGISTRY = {
    "catboost": _CatBoost,
    "xgboost": _XGBoost,
    "lightgbm": _LightGBM,
    "random_forest": _RandomForest,
    "extra_trees": _ExtraTrees,
    "logistic_regression": _LogisticRegression,
    "ridge": _Ridge,
    "lasso": _Lasso,
    "elastic_net": _ElasticNet,
    "svm": _SVM,
    "knn": _KNN,
    "mlp": _MLP,
    "tabpfn": _TabPFN,
    "realmlp": _RealMLP,
}

# The 12 families that are stable for external validation sweeps.
# TabPFN and RealMLP are available but have heavier dependencies.
STANDARD_FAMILIES = [
    "catboost", "xgboost", "lightgbm",
    "random_forest", "extra_trees",
    "logistic_regression", "ridge", "lasso", "elastic_net",
    "svm", "knn", "mlp",
]

ALL_FAMILIES = sorted(_REGISTRY.keys())


def get_model(name: str):
    """Look up a model family by name.

    Returns the module-like class with get_default_params, build_model,
    train, predict, and get_optuna_space methods.

    Raises KeyError if the name is not registered.
    """
    if name not in _REGISTRY:
        raise KeyError(
            f"Unknown model '{name}'. Available: {sorted(_REGISTRY.keys())}"
        )
    return _REGISTRY[name]


def list_models():
    """Return all registered model family names."""
    return sorted(_REGISTRY.keys())
