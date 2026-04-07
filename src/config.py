"""Project paths, column definitions, and model configurations."""

from pathlib import Path


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT_DIR / "data"
COMPETITION_DIR = DATA_DIR / "competition"
EXTERNAL_DIR = DATA_DIR / "external"
FOLDS_DIR = DATA_DIR / "folds"

RESULTS_DIR = ROOT_DIR / "results"
OOF_DIR = RESULTS_DIR / "oof"
TEST_PREDS_DIR = RESULTS_DIR / "test_preds"
METRICS_DIR = RESULTS_DIR / "metrics"
SAVED_MODELS_DIR = RESULTS_DIR / "saved_models"
FIGURES_DIR = ROOT_DIR / "figures"

TRAIN_PATH = COMPETITION_DIR / "train.csv"
TEST_PATH = COMPETITION_DIR / "test.csv"

# Original 270-row UCI Cleveland dataset, used for per-value target statistics.
ORIGINAL_PATH = COMPETITION_DIR / "original.csv"


# ---------------------------------------------------------------------------
# Column names
# ---------------------------------------------------------------------------

TARGET_COLUMN = "Heart Disease"
TARGET_POSITIVE = "Presence"
ID_COLUMN = "id"

# 8 categorical clinical features (small number of discrete levels each).
CAT_FEATURES = [
    "Sex",
    "Chest pain type",
    "FBS over 120",
    "EKG results",
    "Exercise angina",
    "Slope of ST",
    "Number of vessels fluro",
    "Thallium",
]

# 5 continuous clinical measurements.
NUM_FEATURES = [
    "Age",
    "BP",
    "Cholesterol",
    "Max HR",
    "ST depression",
]

# All 13 features: categoricals first, then numericals.
ALL_FEATURES = CAT_FEATURES + NUM_FEATURES

# Column order as it appears in the csv file (used by some pipelines).
FILE_ORDER_FEATURES = [
    "Age",
    "Sex",
    "Chest pain type",
    "BP",
    "Cholesterol",
    "FBS over 120",
    "EKG results",
    "Max HR",
    "Exercise angina",
    "ST depression",
    "Slope of ST",
    "Number of vessels fluro",
    "Thallium",
]


# ---------------------------------------------------------------------------
# Cross-validation defaults
# ---------------------------------------------------------------------------

N_FOLDS = 10
PRIMARY_SEED = 42

# Threshold for converting probabilities to binary predictions.
DEFAULT_THRESHOLD = 0.5


# ---------------------------------------------------------------------------
# Model configurations
# ---------------------------------------------------------------------------
# Each entry: label, features (pipeline name from features.py), family
# (model builder name), params (hyperparameters from Optuna tuning).

MODEL_CONFIGS = {

    # -- CatBoost variants ------------------------------------------------

    "02_cb_raw": {
        "label": "CatBoost (raw, 13 features)",
        "features": "raw",
        "family": "catboost",
        "params": {
            "iterations": 2000,
            "learning_rate": 0.03,
            "depth": 6,
            "eval_metric": "AUC",
            "early_stopping_rounds": 100,
            "verbose": 0,
        },
    },

    "cb_baseline": {
        "label": "CatBoost (freq + UCI means, 39 features)",
        "features": "cb_baseline",
        "family": "catboost",
        "cat_features": list(range(13)),
        "params": {
            "iterations": 3000,
            "learning_rate": 0.03,
            "depth": 6,
            "early_stopping_rounds": 200,
            "l2_leaf_reg": 3,
            "task_type": "CPU",
            "verbose": 0,
        },
    },

    "07_cb_origstats": {
        "label": "CatBoost (origstats, 65 features)",
        "features": "origstats",
        "family": "catboost",
        "params": {
            "iterations": 2000,
            "learning_rate": 0.03,
            "depth": 6,
            "eval_metric": "AUC",
            "early_stopping_rounds": 100,
            "verbose": 0,
        },
    },

    "cb_rs_5": {
        "label": "CatBoost (random_strength=5, 39 features)",
        "features": "cb_baseline",
        "family": "catboost",
        "cat_features": list(range(13)),
        "params": {
            "iterations": 3000,
            "learning_rate": 0.03,
            "depth": 6,
            "early_stopping_rounds": 200,
            "l2_leaf_reg": 3,
            "random_strength": 5,
            "task_type": "CPU",
            "verbose": 0,
        },
    },

    "te_cb_a10": {
        "label": "CatBoost + target encoding (alpha=10)",
        "features": "te_alpha10",
        "family": "catboost",
        "te_alpha": 10,
        "cat_features": [FILE_ORDER_FEATURES.index(f) for f in CAT_FEATURES],
        "params": {
            "iterations": 1000,
            "learning_rate": 0.05,
            "depth": 6,
            "eval_metric": "AUC",
            "early_stopping_rounds": 50,
            "task_type": "CPU",
            "verbose": 0,
        },
    },

    # CatBoost with model shrink regularization. All 13 features treated as
    # categorical (passed as strings to Pool). Shrink_rate=0.01 adds a
    # constant decay to the model, slightly regularizing the ensemble.
    "cb_shrink_0.01": {
        "label": "CatBoost shrink 0.01 (raw categorical, 13 features)",
        "features": "raw_categorical",
        "family": "catboost",
        "cat_features": list(range(13)),
        "params": {
            "iterations": 1000,
            "learning_rate": 0.1,
            "depth": 4,
            "model_shrink_rate": 0.01,
            "model_shrink_mode": "Constant",
            "early_stopping_rounds": 100,
            "verbose": 0,
        },
    },

    # Logistic regression on one-hot encoded features with StandardScaler.
    "06_lr_onehot": {
        "label": "Logistic Regression (onehot, ~466 features)",
        "features": "onehot",
        "family": "logistic_regression",
        "params": {
            "C": 1.0,
            "penalty": "l2",
            "solver": "lbfgs",
            "max_iter": 1000,
        },
    },

    # -- XGBoost variants -------------------------------------------------

    "xgb_origstats_tuned_top1": {
        "label": "XGBoost tuned (origstats_tuned, 104 features)",
        "features": "origstats_tuned",
        "family": "xgboost",
        "params": {
            "n_estimators": 30000,
            "learning_rate": 0.00718870321273411,
            "max_depth": 2,
            "reg_lambda": 0.35832858687915015,
            "min_child_weight": 13,
            "subsample": 0.7903500546551919,
            "colsample_bytree": 0.9610260191699795,
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "tree_method": "hist",
            "verbosity": 0,
            "early_stopping_rounds": 100,
            "n_jobs": -1,
        },
    },

    "xgb39_3seed": {
        "label": "XGBoost (39 features, 3-seed blend)",
        "features": "tree39",
        "family": "xgboost",
        "params": {
            "n_estimators": 3000,
            "learning_rate": 0.019158709669984418,
            "max_depth": 3,
            "subsample": 0.7844422271043349,
            "colsample_bytree": 0.5903265699263532,
            "reg_alpha": 3.7717545543707574,
            "reg_lambda": 0.11772809416829469,
            "min_child_weight": 8,
            "gamma": 1.0445933369727358,
            "tree_method": "hist",
            "early_stopping_rounds": 100,
            "eval_metric": "auc",
            "n_jobs": -1,
        },
        "multi_seed": [42, 123, 456],
    },

    # -- LightGBM variants ------------------------------------------------

    "lgb_deeptune_origstats_top1": {
        "label": "LightGBM deeptune (origstats, 65 features)",
        "features": "origstats",
        "family": "lightgbm_native",
        "num_boost_round": 10000,
        "early_stopping_rounds": 200,
        "params": {
            "objective": "binary",
            "metric": "auc",
            "boosting_type": "gbdt",
            "verbose": -1,
            "n_jobs": -1,
            "learning_rate": 0.026576055473175722,
            "num_leaves": 51,
            "max_depth": 3,
            "min_child_samples": 73,
            "subsample": 0.63594852797544,
            "colsample_bytree": 0.580054568952578,
            "reg_alpha": 4.921666090030823,
            "reg_lambda": 5.527125570834733e-08,
            "min_split_gain": 0.0583303774859317,
            "min_child_weight": 5.8997807277890875,
            "path_smooth": 9.641445889323663,
            "feature_fraction_bynode": 0.34065507958659097,
        },
    },

    "lgb_tuned_origstats_top2": {
        "label": "LightGBM tuned (origstats, 65 features)",
        "features": "origstats",
        "family": "lightgbm",
        "params": {
            "n_estimators": 10000,
            "learning_rate": 0.02,
            "objective": "binary",
            "metric": "auc",
            "verbose": -1,
            "n_jobs": -1,
            "subsample_freq": 1,
            "num_leaves": 15,
            "reg_lambda": 2.2747237411923233,
            "min_child_samples": 60,
            "subsample": 0.8377833306543102,
            "colsample_bytree": 0.8557600358994679,
        },
    },

    # -- Linear models ----------------------------------------------------

    "lr_tuned": {
        "label": "Logistic Regression (L1, tuned C)",
        "features": "onehot_scaled",
        "family": "logistic_regression",
        "params": {
            "C": 0.021303793957563048,
            "penalty": "l1",
            "solver": "saga",
            "max_iter": 1000,
            "n_jobs": -1,
        },
    },

    "lr_elasticnet_r0.9": {
        "label": "Elastic Net (l1_ratio=0.9)",
        "features": "onehot",
        "family": "logistic_regression",
        "params": {
            "penalty": "elasticnet",
            "solver": "saga",
            "l1_ratio": 0.9,
            "C": 1.0,
            "max_iter": 5000,
        },
    },

    "ridge_onehot": {
        "label": "Ridge (onehot features)",
        "features": "onehot",
        "family": "ridge",
        "params": {
            "alpha": 1.0,
        },
    },

    "svm_linear_origstats": {
        "label": "Linear SVM (origstats features)",
        "features": "origstats",
        "family": "svm",
        "per_fold_scaling": True,
        "params": {
            "C": 1.0,
            "max_iter": 5000,
        },
    },

    # -- Tree ensembles ---------------------------------------------------

    "05_rf_raw": {
        "label": "Random Forest (raw, 13 features)",
        "features": "raw",
        "family": "random_forest",
        "params": {
            "n_estimators": 500,
            "max_depth": None,
            "min_samples_leaf": 5,
            "max_features": "sqrt",
            "n_jobs": -1,
        },
    },

    "sweep_ExtraTrees_n2000_d20": {
        "label": "Extra Trees (enriched, n=2000, depth=20)",
        "features": "enriched_tree",
        "family": "extra_trees",
        "params": {
            "n_estimators": 2000,
            "max_depth": 20,
            "n_jobs": -1,
        },
    },

    # -- Neural networks --------------------------------------------------

    "mlp_256_128_64_32": {
        "label": "MLP (4 hidden layers, PyTorch)",
        "features": "onehot_freq_origmean",
        "family": "pytorch_mlp",
        "params": {
            "hidden_dims": [256, 128, 64, 32],
            "dropout": 0.3,
            "lr": 0.001,
            "epochs": 200,
        },
    },

    # -- Neighbors --------------------------------------------------------

    "knn_k21_manhattan": {
        "label": "KNN (k=21, manhattan)",
        "features": "onehot_freq_origmean",
        "family": "knn",
        "full_train_scaling": True,
        "params": {
            "n_neighbors": 21,
            "metric": "manhattan",
            "n_jobs": -1,
        },
    },

    # -- Tabular deep learning --------------------------------------------

    "realmlp_lean_f3_s42": {
        "label": "RealMLP (lean, 3-fold, seed 42)",
        "features": "origstats_categorical",
        "family": "realmlp",
        "n_folds": 3,
        "params": {
            "device": "cpu",
            "verbosity": 2,
            "n_epochs": 100,
            "batch_size": 128,
            "n_ens": 2,
            "use_early_stopping": True,
            "early_stopping_additive_patience": 20,
            "early_stopping_multiplicative_patience": 1,
            "act": "mish",
            "embedding_size": 8,
            "first_layer_lr_factor": 0.5962121993798933,
            "hidden_sizes": "rectangular",
            "hidden_width": 384,
            "lr": 0.04,
            "ls_eps": 0.011498317194338772,
            "ls_eps_sched": "coslog4",
            "max_one_hot_cat_size": 18,
            "n_hidden_layers": 4,
            "p_drop": 0.07301419697186451,
            "p_drop_sched": "flat_cos",
            "plr_hidden_1": 16,
            "plr_hidden_2": 8,
            "plr_lr_factor": 0.1151437622270563,
            "plr_sigma": 2.3316811282666916,
            "scale_lr_factor": 2.244801835541429,
            "sq_mom": 0.9881659450444177,
            "wd": 0.02369230879235962,
        },
    },

    "tabpfn_mps_10k_e4": {
        "label": "TabPFN (10K subsample, 4 estimators)",
        "features": "raw_integer",
        "family": "tabpfn",
        "cat_indices": [],  # TabPFN detected no object-typed columns
        "params": {
            "sub_size": 10000,
            "n_estimators": 4,
            "n_sub": 2,
            "device": "mps",
        },
    },

    # =====================================================================
    # Ensemble dependency models
    #
    # These are the base models that feed into the ensemble chain
    # (hillclimb, multi-seed blends, band-gated final prediction).
    # Each uses multi_seed with 5 random seeds and different fold files.
    # =====================================================================

    # -- Multi-seed CatBoost + freq encoding (Series 18) ------------------
    # CatBoost on 26 numeric features: 13 combined frequency encodings
    # + 13 UCI target means. No raw columns, no cat_features (all numeric).
    # Each seed uses a DIFFERENT fold split for fold-level diversity.
    "18_cb_freq_multiseed": {
        "label": "CatBoost freq+origmean (5-seed)",
        "features": "freq_origmean",
        "family": "catboost",
        "multi_seed": [42, 123, 456, 789, 2024],
        "per_seed_folds": True,
        "params": {
            "iterations": 1000,
            "learning_rate": 0.1,
            "depth": 4,
            "l2_leaf_reg": 1.0,
            "random_strength": 1.0,
            "bagging_temperature": 1.0,
            "border_count": 254,
            "task_type": "CPU",
            "early_stopping_rounds": 100,
            "verbose": 0,
        },
    },

    # -- Multi-seed LR + onehot (Series 19) -------------------------------
    # Logistic regression on one-hot encoded features. LR is nearly
    # deterministic so seed variation is minimal, but fold split changes
    # produce slightly different OOFs for diversity in the blend.
    "19_lr_onehot_multiseed": {
        "label": "Logistic Regression onehot (5-seed)",
        "features": "onehot",
        "family": "logistic_regression",
        "multi_seed": [42, 123, 456, 789, 2024],
        "per_seed_folds": True,
        "params": {
            "C": 1.0,
            "penalty": "l2",
            "solver": "lbfgs",
            "max_iter": 1000,
        },
    },

    # -- Multi-seed CatBoost + origstats (Series 20) ----------------------
    # CatBoost with 52 features: 13 raw + 3 UCI stats per feature (mean,
    # std, count). Uses SAME folds (seed 42) for all seeds -- diversity
    # comes from CatBoost random_seed only. The 8 categorical columns
    # use CatBoost native encoding.
    "20_cb_origstats_multiseed": {
        "label": "CatBoost origstats mini (5-seed)",
        "features": "origstats_mini",
        "family": "catboost",
        "cat_features": CAT_FEATURES,
        "multi_seed": [42, 123, 456, 789, 2024],
        "per_seed_folds": False,
        "params": {
            "iterations": 3000,
            "learning_rate": 0.05,
            "depth": 6,
            "l2_leaf_reg": 3,
            "eval_metric": "AUC",
            "early_stopping_rounds": 100,
            "task_type": "CPU",
            "verbose": 0,
        },
    },

    # -- Cross-stacked CatBoost + LR OOF (Series 40) ---------------------
    # CatBoost trained on the 13 raw features plus orig stats, frequency
    # encoding, AND the 06_lr_onehot OOF predictions as an extra numeric
    # feature. This cross-stacking leverages the LR's complementary signal.
    # 5 seeds, each with its own fold file.
    "40_cb_lr_stack": {
        "label": "CatBoost cross-stacked with LR OOF (5-seed)",
        "features": "cross_stack_lr",
        "family": "catboost",
        "cat_features": list(range(8)),
        "multi_seed": [42, 123, 456, 789, 2024],
        "per_seed_folds": True,
        "cross_stack_oof": "06_lr_onehot",
        "params": {
            "iterations": 1000,
            "learning_rate": 0.05,
            "depth": 6,
            "eval_metric": "AUC",
            "early_stopping_rounds": 50,
            "verbose": 0,
        },
    },

    # -- Pairwise + prototype CatBoost (cb_pairproto_s42) -----------------
    # CatBoost with single-feature orig stats, pairwise Bayesian-smoothed
    # target stats (10 feature pairs, alpha=15), and Gower-style prototype
    # distance features designed for ensemble diversity.
    "cb_pairproto_s42": {
        "label": "CatBoost pairwise + prototype features",
        "features": "pairwise_proto",
        "family": "catboost",
        "per_fold_seed": True,
        "params": {
            "iterations": 5000,
            "learning_rate": 0.01,
            "depth": 6,
            "loss_function": "Logloss",
            "eval_metric": "AUC",
            "bootstrap_type": "Bernoulli",
            "subsample": 0.9,
            "l2_leaf_reg": 12,
            "random_strength": 1.2,
            "task_type": "CPU",
            "early_stopping_rounds": 100,
            "verbose": 0,
        },
    },

    # -- Per-fold CB+XGB+RF stacking (cb_origstats_f10_s42) ---------------
    # Three models (CatBoost, XGBoost, RandomForest) trained per fold,
    # their predictions rank-blended. Uses origstats features (mean, median,
    # std, skew, count) plus frequency and target encoding. Per-fold seed
    # is 42 + fold_index.
    "cb_origstats_f10_s42": {
        "label": "Per-fold CB+XGB+RF stacking (origstats)",
        "features": "origstats_full",
        "family": "per_fold_stack",
        "n_folds": 10,
        "seed": 42,
        "per_fold_seed": True,
        "params": {},
    },

    # -- Per-fold CB+XGB+RF stacking (top_pipe_f10_s42) -------------------
    # Same stacking pattern as cb_origstats_f10_s42 but with the top
    # pipeline features: frequency encoding, target encoding, and
    # correlation-based feature growth (top 8 correlated TE pairs).
    "top_pipe_f10_s42": {
        "label": "Per-fold CB+XGB+RF stacking (top pipeline)",
        "features": "top_pipe",
        "family": "per_fold_stack",
        "n_folds": 10,
        "seed": 42,
        "per_fold_seed": True,
        "params": {},
    },


}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def ensure_output_dirs():
    """Create all output directories used by training and analysis code."""
    for directory in [OOF_DIR, TEST_PREDS_DIR, METRICS_DIR, FIGURES_DIR, FOLDS_DIR, SAVED_MODELS_DIR]:
        directory.mkdir(parents=True, exist_ok=True)
