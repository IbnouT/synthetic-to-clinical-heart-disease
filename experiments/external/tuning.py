"""Tuning configuration for UCI external validation experiments.

Defines three sets of hyperparameters per model family:
  - DEFAULT_PARAMS:  reasonable practitioner defaults for the "default" tuning mode
  - get_competition_params(): loads competition-verified params for the "competition" mode
  - OPTUNA_SPACES:  search space functions for the "dataset" tuning mode (Optuna nested CV)

Default params match what was used in the original external validation study.
Competition params come from src/config.py MODEL_CONFIGS (the same params that
produced the verified competition OOFs).
Optuna spaces define the per-family hyperparameter ranges for dataset-specific tuning.
"""

import json
from pathlib import Path

OPTUNA_TRIALS = {
    "cleveland": 100,
    "hungarian": 100,
    "va_longbeach": 50,
    "switzerland": 30,
}
OPTUNA_TRIALS_DEFAULT = 100


# ---------------------------------------------------------------------------
# Default hyperparameters (one dict per family)
# ---------------------------------------------------------------------------
# These are the "default" tuning condition: reasonable values a practitioner
# would use without competition-specific tuning or dataset-specific search.

DEFAULT_PARAMS = {
    "catboost": {
        "iterations": 1500,
        "learning_rate": 0.03,
        "depth": 6,
        "l2_leaf_reg": 3.0,
        "loss_function": "Logloss",
        "eval_metric": "AUC",
        "verbose": False,
        "allow_writing_files": False,
        "thread_count": -1,
    },
    "xgboost": {
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
        "n_jobs": -1,
        "tree_method": "hist",
        "verbosity": 0,
    },
    "lightgbm": {
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
        "n_jobs": -1,
        "verbosity": -1,
    },
    "random_forest": {
        "n_estimators": 600,
        "max_depth": None,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "max_features": "sqrt",
        "bootstrap": True,
        "class_weight": None,
        "n_jobs": -1,
    },
    "extra_trees": {
        "n_estimators": 800,
        "max_depth": None,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "max_features": "sqrt",
        "bootstrap": False,
        "class_weight": None,
        "n_jobs": -1,
    },
    "logistic_regression": {
        "penalty": "l2",
        "C": 1.0,
        "solver": "lbfgs",
        "max_iter": 2000,
        "class_weight": None,
        "n_jobs": -1,
    },
    "ridge": {
        "alpha": 1.0,
        "solver": "auto",
        "fit_intercept": True,
        "tol": 1e-4,
        "class_weight": None,
    },
    "lasso": {
        "penalty": "l1",
        "C": 1.0,
        "solver": "saga",
        "max_iter": 4000,
        "class_weight": None,
        "n_jobs": -1,
    },
    "elastic_net": {
        "penalty": "elasticnet",
        "C": 1.0,
        "l1_ratio": 0.5,
        "solver": "saga",
        "max_iter": 4000,
        "class_weight": None,
        "n_jobs": -1,
    },
    "svc": {
        "C": 1.0,
        "kernel": "rbf",
        "gamma": "scale",
        "degree": 3,
        "class_weight": None,
    },
    "knn": {
        "n_neighbors": 15,
        "weights": "distance",
        "metric": "minkowski",
        "p": 2,
        "n_jobs": -1,
    },
    "pytorch_mlp": {
        "hidden_dims": [128, 64],
        "dropout": 0.3,
        "lr": 0.001,
        "epochs": 200,
    },
    "tabpfn": {
        "n_estimators": 4,
        "device": "cpu",
    },
    "realmlp": {
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
    },
}


# ---------------------------------------------------------------------------
# Competition params loader
# ---------------------------------------------------------------------------

# Maps UCI family names to verified_params JSON filenames.
# These files contain the best competition-verified hyperparameters
# per model family, stored in data/verified_params/.
_VERIFIED_PARAMS_FILES = {
    "catboost": "catboost_origstats.json",
    "xgboost": "xgboost_default.json",
    "lightgbm": "lightgbm.json",
    "random_forest": "random_forest.json",
    "extra_trees": "extra_trees.json",
    "logistic_regression": "logistic_regression_onehot.json",
    "ridge": "ridge.json",
    "lasso": "lasso.json",
    "elastic_net": "elastic_net.json",
    "svc": "svm.json",
    "svm": "svm.json",
    "knn": "knn.json",
    # pytorch_mlp uses PyTorch SimpleMLP, not sklearn MLPClassifier.
    # Competition params (sklearn format) are incompatible with PyTorch.
    "tabpfn": "tabpfn.json",
    "realmlp": "realmlp.json",
}


def get_competition_params(family):
    """Load competition-verified params for a model family.

    Reads from data/verified_params/<family>.json. These files contain
    the best hyperparameters found during the Kaggle competition,
    verified against competition OOF predictions.
    Returns the params dict, or None if no file exists for the family.
    """
    filename = _VERIFIED_PARAMS_FILES.get(family)
    if filename is None:
        return None

    params_dir = Path(__file__).resolve().parent.parent.parent / "data" / "verified_params"
    params_path = params_dir / filename

    if not params_path.exists():
        return None

    with open(params_path) as f:
        data = json.load(f)

    params = data["params"] if isinstance(data, dict) and "params" in data else data

    # Remove params that are set by the fold trainer (avoids duplicates).
    # Seeds are set via the seed argument, and SVC probability is always
    # enabled by train_fold_svc.
    for managed_key in ("random_seed", "random_state", "probability"):
        params.pop(managed_key, None)

    return params


# ---------------------------------------------------------------------------
# Optuna search spaces
# ---------------------------------------------------------------------------
# Each function takes an Optuna trial and returns a dict of suggested params.
# These are the ranges Optuna explores during the "dataset" tuning condition.

def _optuna_catboost(trial):
    """CatBoost search space with proper bootstrap_type conditioning.

    Bayesian bootstrap uses bagging_temperature (not subsample).
    Bernoulli and MVS use subsample (not bagging_temperature).

    Both parameters are always suggested to keep Optuna's parameter
    space consistent across trials (Optuna requires the same parameter
    names every trial). Only the relevant parameter is included in the
    returned dict.
    """
    bootstrap = trial.suggest_categorical(
        "bootstrap_type", ["Bayesian", "Bernoulli", "MVS"])

    # Always suggest both so Optuna sees a fixed parameter space
    bagging_temp = trial.suggest_float("bagging_temperature", 0.0, 10.0)
    subsample = trial.suggest_float("subsample", 0.5, 1.0)

    space = {
        "iterations": trial.suggest_int("iterations", 400, 4000),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.2, log=True),
        "depth": trial.suggest_int("depth", 3, 10),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-3, 100.0, log=True),
        "bootstrap_type": bootstrap,
        "random_strength": trial.suggest_float("random_strength", 1e-3, 10.0, log=True),
        "early_stopping_rounds": trial.suggest_int("early_stopping_rounds", 50, 300),
    }

    # Only include the parameter that matches the chosen bootstrap type
    if bootstrap == "Bayesian":
        space["bagging_temperature"] = bagging_temp
    else:
        space["subsample"] = subsample

    return space


def _optuna_xgboost(trial):
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


def _optuna_lightgbm(trial):
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


def _optuna_random_forest(trial):
    return {
        "n_estimators": trial.suggest_int("n_estimators", 200, 2000),
        "max_depth": trial.suggest_int("max_depth", 3, 40),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
        "bootstrap": trial.suggest_categorical("bootstrap", [True, False]),
        "class_weight": trial.suggest_categorical("class_weight", [None, "balanced"]),
    }


def _optuna_extra_trees(trial):
    return {
        "n_estimators": trial.suggest_int("n_estimators", 200, 2500),
        "max_depth": trial.suggest_int("max_depth", 3, 40),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
        "bootstrap": trial.suggest_categorical("bootstrap", [True, False]),
        "class_weight": trial.suggest_categorical("class_weight", [None, "balanced"]),
    }


def _optuna_logistic_regression(trial):
    """Logistic regression search space.

    All conditional parameters (solver, l1_ratio) are suggested every
    trial to keep Optuna's parameter space fixed. Only the compatible
    values are used in the returned dict.
    """
    penalty = trial.suggest_categorical("penalty", ["l1", "l2", "elasticnet"])

    # Always suggest all params so Optuna sees a fixed space
    solver_full = trial.suggest_categorical("solver", ["lbfgs", "liblinear", "saga"])
    l1_ratio = trial.suggest_float("l1_ratio", 0.01, 0.99)

    # Pick the compatible solver for the chosen penalty
    if penalty == "l1":
        solver = solver_full if solver_full in ("liblinear", "saga") else "saga"
    elif penalty == "elasticnet":
        solver = "saga"
    else:
        solver = solver_full

    space = {
        "penalty": penalty,
        "solver": solver,
        "C": trial.suggest_float("C", 1e-4, 100.0, log=True),
        "max_iter": trial.suggest_int("max_iter", 1000, 5000),
        "class_weight": trial.suggest_categorical("class_weight", [None, "balanced"]),
    }

    # Only include l1_ratio when elasticnet needs it
    if penalty == "elasticnet":
        space["l1_ratio"] = l1_ratio

    return space


def _optuna_ridge(trial):
    return {
        "alpha": trial.suggest_float("alpha", 1e-4, 100.0, log=True),
        "solver": trial.suggest_categorical(
            "solver", ["auto", "svd", "cholesky", "lsqr", "sag", "saga"]),
        "tol": trial.suggest_float("tol", 1e-6, 1e-2, log=True),
        "class_weight": trial.suggest_categorical("class_weight", [None, "balanced"]),
    }


def _optuna_lasso(trial):
    return {
        "penalty": "l1",
        "solver": trial.suggest_categorical("solver", ["liblinear", "saga"]),
        "C": trial.suggest_float("C", 1e-4, 100.0, log=True),
        "max_iter": trial.suggest_int("max_iter", 1500, 6000),
        "class_weight": trial.suggest_categorical("class_weight", [None, "balanced"]),
    }


def _optuna_elastic_net(trial):
    return {
        "penalty": "elasticnet",
        "solver": "saga",
        "C": trial.suggest_float("C", 1e-4, 100.0, log=True),
        "l1_ratio": trial.suggest_float("l1_ratio", 0.01, 0.99),
        "max_iter": trial.suggest_int("max_iter", 1500, 6000),
        "class_weight": trial.suggest_categorical("class_weight", [None, "balanced"]),
    }


def _optuna_svc(trial):
    """SVC search space with fixed parameter names across all trials.

    gamma and degree are always suggested but only included when
    the chosen kernel uses them.
    """
    kernel = trial.suggest_categorical("kernel", ["linear", "rbf", "poly", "sigmoid"])

    # Always suggest so Optuna sees a fixed space
    gamma = trial.suggest_float("gamma", 1e-4, 10.0, log=True)
    degree = trial.suggest_int("degree", 2, 5)

    space = {
        "C": trial.suggest_float("C", 1e-3, 100.0, log=True),
        "kernel": kernel,
        "probability": True,
        "class_weight": trial.suggest_categorical("class_weight", [None, "balanced"]),
    }
    if kernel in ("rbf", "poly", "sigmoid"):
        space["gamma"] = gamma
    if kernel == "poly":
        space["degree"] = degree
    return space


def _optuna_knn(trial):
    """KNN search space. The Minkowski p parameter is always suggested
    but only included when the metric is minkowski.
    """
    metric = trial.suggest_categorical("metric", ["minkowski", "manhattan", "euclidean"])
    p = trial.suggest_int("p", 1, 3)
    return {
        "n_neighbors": trial.suggest_int("n_neighbors", 3, 151),
        "weights": trial.suggest_categorical("weights", ["uniform", "distance"]),
        "metric": metric,
        "p": p if metric == "minkowski" else 2,
    }


def _optuna_pytorch_mlp(trial):
    return {
        "hidden_dims": trial.suggest_categorical(
            "hidden_dims",
            [(64,), (128,), (256,), (128, 64), (256, 128), (256, 128, 64)]),
        "dropout": trial.suggest_float("dropout", 0.0, 0.5),
        "lr": trial.suggest_float("lr", 1e-4, 1e-1, log=True),
        "epochs": trial.suggest_int("epochs", 100, 500),
    }


def _optuna_tabpfn(trial):
    return {
        "n_estimators": trial.suggest_categorical(
            "n_estimators", [4, 8, 16, 32]),
    }


def _optuna_realmlp(trial):
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


# Lookup table: family name -> Optuna space function
OPTUNA_SPACES = {
    "catboost": _optuna_catboost,
    "xgboost": _optuna_xgboost,
    "lightgbm": _optuna_lightgbm,
    "random_forest": _optuna_random_forest,
    "extra_trees": _optuna_extra_trees,
    "logistic_regression": _optuna_logistic_regression,
    "ridge": _optuna_ridge,
    "lasso": _optuna_lasso,
    "elastic_net": _optuna_elastic_net,
    "svm": _optuna_svc,
    "svc": _optuna_svc,  # alternative name used by some scripts
    "knn": _optuna_knn,
    "mlp": _optuna_pytorch_mlp,
    "pytorch_mlp": _optuna_pytorch_mlp,  # alternative name used by some scripts
    "tabpfn": _optuna_tabpfn,
    "realmlp": _optuna_realmlp,
}
