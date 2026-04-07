"""Train competition models and save OOF predictions, test predictions, and metrics.

This is the main entry point for running model training. It reads model
configurations from config.py and uses the shared CV loop in training.py
to produce reproducible results.

Each run saves three files under results/:
  - oof/<experiment_id>_oof.npy         (out-of-fold predictions)
  - test_preds/<experiment_id>_test.npy (averaged test predictions)
  - metrics/<experiment_id>_metrics.json (AUC, accuracy, F1, etc.)

Usage examples:
    # Train one specific model
    python -m experiments.01_train_models --model 02_cb_raw

    # Train all configured models
    python -m experiments.01_train_models --all

    # Show the list of available model IDs
    python -m experiments.01_train_models --list
"""

import argparse

from src.config import MODEL_CONFIGS
from src.config import ensure_output_dirs
from src.training import run_cv
from src.training_stacked import run_stacked_cv
from src.data import load_train_test
from src.features import FEATURE_BUILDERS


def _train_model(exp_id, cfg):
    """
    Dispatch a model to the right training function.

    Most models use the standard run_cv loop. Per-fold stacked models
    (family='per_fold_stack') use run_stacked_cv which trains CB+XGB+RF
    within each fold and combines their predictions.
    """
    if cfg.get("family") == "per_fold_stack":
        train_df, test_df = load_train_test()
        x_train, x_test = FEATURE_BUILDERS[cfg["features"]](train_df, test_df)
        y = train_df["target"].values
        run_stacked_cv(
            exp_id, x_train, x_test, y,
            n_folds=cfg.get("n_folds", 10),
            seed=cfg.get("seed", 42),
        )
    else:
        run_cv(exp_id, cfg)


def main():
    """Parse command-line arguments and dispatch training."""
    parser = argparse.ArgumentParser(
        description="Train competition models with K-fold CV"
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--model", help="Experiment ID to train (e.g. 02_cb_raw)")
    group.add_argument("--all", action="store_true", help="Train all models")
    group.add_argument("--list", action="store_true", help="List available model IDs")
    args = parser.parse_args()

    if args.list:
        print("Available models:")
        for exp_id, cfg in MODEL_CONFIGS.items():
            print(f"  {exp_id:40s} {cfg['label']}")
        return

    ensure_output_dirs()

    if args.all:
        for exp_id, cfg in MODEL_CONFIGS.items():
            print(f"\n{'='*60}")
            print(f"Training: {exp_id} ({cfg['label']})")
            print(f"{'='*60}")
            _train_model(exp_id, cfg)
    else:
        if args.model not in MODEL_CONFIGS:
            print(f"Unknown model: {args.model}")
            print(f"Run with --list to see available model IDs.")
            return

        cfg = MODEL_CONFIGS[args.model]
        print(f"Training: {args.model} ({cfg['label']})")
        _train_model(args.model, cfg)


if __name__ == "__main__":
    main()
