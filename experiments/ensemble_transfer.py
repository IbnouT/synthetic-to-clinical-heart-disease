"""Diverse model family ensemble on UCI clinical datasets.

For each external dataset, loads the S3 within-dataset OOF predictions,
identifies the best configuration per model family, and tests multiple
ensemble strategies. This measures whether combining 14 diverse model
families produces better predictions than any single family.

Ensemble strategies:
  1. Equal-weight rank blend (all families)
  2. Greedy hill-climbing (forward selection maximizing AUC)
  3. Top-3 rank blend (3 best individual families)
  4. Top-5 rank blend (5 best individual families)
  5. Simple probability average

Depends on S3 within-dataset CV results being available in
results/external/evals/s3_*.json and s3_*.npz files.

Usage:
    python -m experiments.ensemble_transfer
    python -m experiments.ensemble_transfer --datasets cleveland hungarian
    python -m experiments.ensemble_transfer --dry-run
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import average_precision_score, roc_auc_score

from scipy.stats import rankdata

from src.ensemble.blend import hillclimb

from src.data_external import DATASET_INFO

logger = logging.getLogger(__name__)

EXTERNAL_DATASETS = list(DATASET_INFO.keys())


class _NumpyEncoder(json.JSONEncoder):
    """Serialize numpy types for JSON output."""
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def parse_args():
    """Parse command-line arguments for ensemble transfer evaluation."""
    parser = argparse.ArgumentParser(
        description="05b: Diverse family ensemble on UCI datasets.")
    parser.add_argument(
        "--datasets", nargs="+", default=EXTERNAL_DATASETS,
        choices=EXTERNAL_DATASETS)
    parser.add_argument(
        "--output-dir", type=str, default=None)
    parser.add_argument(
        "--dry-run", action="store_true")
    return parser.parse_args()


def _find_best_per_family(eval_dir, dataset_name):
    """Find the best S3 configuration per model family for one dataset.

    Scans all s3_*.json files matching this dataset and picks the
    configuration with the highest OOF AUC for each model family.
    Returns a dict mapping family name to its best eval metadata.
    """
    best = {}
    for json_path in sorted(eval_dir.glob(f"s3_*_{dataset_name}_*.json")):
        try:
            with open(json_path) as f:
                result = json.load(f)
        except (json.JSONDecodeError, IOError):
            continue

        model = result.get("model", "")
        if result.get("dataset", "") != dataset_name:
            continue
        auc = result.get("oof_auc", 0.0)
        if not model or np.isnan(auc):
            continue

        if model not in best or auc > best[model]["oof_auc"]:
            best[model] = {
                "tuning": result.get("tuning", "default"),
                "imbalance": result.get("imbalance", "native"),
                "oof_auc": auc,
                "eval_key": json_path.stem,
            }
    return best


def _load_oofs(eval_dir, best_configs):
    """Load OOF prediction arrays for each family's best configuration.

    Each s3_*.npz file contains 'oof' (predictions) and 'y' (labels).
    Returns (oofs_dict, y_true) where oofs_dict maps family name to
    its OOF prediction array.
    """
    oofs = {}
    y_true = None
    for model_name, cfg in best_configs.items():
        npz_path = eval_dir / f"{cfg['eval_key']}.npz"
        if not npz_path.exists():
            logger.warning("  Missing NPZ for %s: %s", model_name, npz_path.name)
            continue
        data = np.load(npz_path)
        oofs[model_name] = data["oof"]
        if y_true is None:
            y_true = data["y"]
    return oofs, y_true


def _rank_normalize(arr):
    """Convert predictions to rank-normalized [0, 1] scale."""
    return rankdata(arr) / len(arr)


def _rank_blend(oofs, y, names=None, weights=None):
    """Equal or weighted rank blend of multiple OOF arrays."""
    if names is None:
        names = sorted(oofs.keys())
    if weights is None:
        weights = {n: 1.0 / len(names) for n in names}
    blend = np.zeros(len(y), dtype=np.float64)
    for name in names:
        blend += weights[name] * _rank_normalize(oofs[name])
    auc = roc_auc_score(y, blend) if len(np.unique(y)) > 1 else float("nan")
    return blend, auc


def _top_k_blend(oofs, y, k):
    """Rank blend of the top-k families by individual AUC."""
    aucs = {n: roc_auc_score(y, oofs[n]) for n in oofs}
    top_names = sorted(aucs, key=aucs.get, reverse=True)[:k]
    top_oofs = {n: oofs[n] for n in top_names}
    blend, auc = _rank_blend(top_oofs, y)
    return blend, auc, top_names


def _simple_average(oofs, y):
    """Simple probability average without rank normalization."""
    names = sorted(oofs.keys())
    avg = np.mean([oofs[n] for n in names], axis=0)
    auc = roc_auc_score(y, avg) if len(np.unique(y)) > 1 else float("nan")
    return avg, auc


def main():
    """Build diverse-family ensembles from S3 OOF predictions."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s")

    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    output_dir = (Path(args.output_dir) if args.output_dir
                  else project_root / "results" / "external")
    eval_dir = project_root / "results" / "external" / "evals"

    if not eval_dir.exists():
        logger.error("No S3 eval directory found at %s", eval_dir)
        logger.error("Run within_dataset_cv.py first to generate S3 OOFs.")
        return

    all_results = []

    for ds_name in args.datasets:
        print(f"\n{'='*70}")
        print(f"Dataset: {ds_name}")
        print(f"{'='*70}")

        # Find best config per model family from S3 results
        best_configs = _find_best_per_family(eval_dir, ds_name)
        if not best_configs:
            logger.warning("  No S3 results found for %s, skipping", ds_name)
            continue

        print(f"\n  Best S3 config per family ({len(best_configs)} models):")
        for m, c in sorted(best_configs.items(),
                           key=lambda x: x[1]["oof_auc"], reverse=True):
            print(f"    {m:25s} {c['tuning']:12s} {c['imbalance']:6s} "
                  f"AUC={c['oof_auc']:.4f}")

        if args.dry_run:
            continue

        # Load OOF predictions for each family
        oofs, y = _load_oofs(eval_dir, best_configs)
        if len(oofs) < 3:
            logger.warning("  Only %d model OOFs loaded, need at least 3", len(oofs))
            continue

        print(f"\n  Loaded {len(oofs)} model OOFs")

        # Individual model results
        for name in sorted(oofs.keys()):
            auc = roc_auc_score(y, oofs[name])
            pr_auc = average_precision_score(y, oofs[name])
            print(f"    {name:25s}  AUC={auc:.6f}  PR-AUC={pr_auc:.4f}")
            all_results.append({
                "dataset": ds_name,
                "method": f"single_{name}",
                "ensemble_type": "single",
                "auc": round(float(auc), 6),
                "pr_auc": round(float(pr_auc), 4),
                "n_models": 1,
                "source_config": best_configs[name]["eval_key"],
            })

        # Ensemble methods
        print(f"\n  Ensemble results:")

        # 1. Equal-weight rank blend
        eq_blend, eq_auc = _rank_blend(oofs, y)
        eq_pr = float(average_precision_score(y, eq_blend))
        print(f"    {'rank_blend_equal':25s}  AUC={eq_auc:.6f}  PR-AUC={eq_pr:.4f}")
        all_results.append({
            "dataset": ds_name, "method": "rank_blend_equal",
            "ensemble_type": "rank_blend", "auc": round(eq_auc, 6),
            "pr_auc": round(eq_pr, 4), "n_models": len(oofs),
        })

        # 2. Hill-climbing greedy selection
        selected, hc_blend, hc_auc = hillclimb(oofs, y)
        hc_pr = float(average_precision_score(y, hc_blend))
        print(f"    {'hillclimb':25s}  AUC={hc_auc:.6f}  PR-AUC={hc_pr:.4f}")
        print(f"      Selected: {selected}")
        all_results.append({
            "dataset": ds_name, "method": "hillclimb",
            "ensemble_type": "hillclimb", "auc": round(hc_auc, 6),
            "pr_auc": round(hc_pr, 4), "n_models": len(selected),
            "selected_models": selected,
        })

        # 3. Simple probability average
        avg_blend, avg_auc = _simple_average(oofs, y)
        avg_pr = float(average_precision_score(y, avg_blend))
        print(f"    {'simple_average':25s}  AUC={avg_auc:.6f}  PR-AUC={avg_pr:.4f}")
        all_results.append({
            "dataset": ds_name, "method": "simple_average",
            "ensemble_type": "simple_avg", "auc": round(avg_auc, 6),
            "pr_auc": round(avg_pr, 4), "n_models": len(oofs),
        })

        # 4. Top-3 rank blend
        if len(oofs) >= 3:
            t3_blend, t3_auc, t3_names = _top_k_blend(oofs, y, 3)
            t3_pr = float(average_precision_score(y, t3_blend))
            print(f"    {'top3_rank_blend':25s}  AUC={t3_auc:.6f}  PR-AUC={t3_pr:.4f}")
            all_results.append({
                "dataset": ds_name, "method": "top3_rank_blend",
                "ensemble_type": "top_k", "auc": round(t3_auc, 6),
                "pr_auc": round(t3_pr, 4), "n_models": 3,
                "selected_models": t3_names,
            })

        # 5. Top-5 rank blend
        if len(oofs) >= 5:
            t5_blend, t5_auc, t5_names = _top_k_blend(oofs, y, 5)
            t5_pr = float(average_precision_score(y, t5_blend))
            print(f"    {'top5_rank_blend':25s}  AUC={t5_auc:.6f}  PR-AUC={t5_pr:.4f}")
            all_results.append({
                "dataset": ds_name, "method": "top5_rank_blend",
                "ensemble_type": "top_k", "auc": round(t5_auc, 6),
                "pr_auc": round(t5_pr, 4), "n_models": 5,
                "selected_models": t5_names,
            })

    # Save results
    if all_results:
        output_dir.mkdir(parents=True, exist_ok=True)

        with open(output_dir / "ensemble_diverse_models.json", "w") as f:
            json.dump(all_results, f, indent=2, cls=_NumpyEncoder)

        flat_rows = []
        for r in all_results:
            row = {k: v for k, v in r.items()
                   if k not in ("selected_models", "source_config")}
            flat_rows.append(row)
        pd.DataFrame(flat_rows).to_csv(
            output_dir / "ensemble_diverse_models.csv", index=False)

        logger.info("Saved %d results to %s", len(all_results), output_dir)

    logger.info("Done.")


if __name__ == "__main__":
    main()
