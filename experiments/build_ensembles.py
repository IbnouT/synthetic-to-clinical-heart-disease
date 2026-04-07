"""Build ensemble predictions from base model OOFs.

Reads ensemble definitions from src/ensemble/definitions.py and builds
each one in dependency order. Each definition specifies a blending
technique, inputs, and parameters. New ensembles can be added by
editing the definitions file.

Usage:
    python -m experiments.02_build_ensembles
    python -m experiments.02_build_ensembles --only hillclimb_v4
    python -m experiments.02_build_ensembles --verify-only
    python -m experiments.02_build_ensembles --list
"""

import argparse

import numpy as np

from scipy.stats import rankdata

from sklearn.metrics import roc_auc_score

from src.ensemble import ENSEMBLE_DEFS, rank_blend, hillclimb, band_gate_blend

from src.config import RESULTS_DIR
from src.data import load_train_test
from src.metrics import compute_metrics
from src.utils import save_oof, save_test_preds, save_metrics


# Base model OOFs live here (copied from competition outputs).
DEPS_DIR = RESULTS_DIR / "ensemble_deps"


def load_predictions(name, built):
    """
    Load OOF and test predictions for a model or ensemble.

    Checks the built cache first (for ensembles computed this run),
    then falls back to files in ensemble_deps/ or the main results
    directories.
    """
    if name in built:
        return built[name]

    oof_path = DEPS_DIR / f"{name}_oof.npy"
    test_path = DEPS_DIR / f"{name}_test.npy"
    if not oof_path.exists():
        oof_path = RESULTS_DIR / "oof" / f"{name}_oof.npy"
        test_path = RESULTS_DIR / "test_preds" / f"{name}_test.npy"

    oof = np.load(oof_path)
    test = np.load(test_path)
    return oof, test


def resolve_build_order(defs, targets=None):
    """
    Topological sort of ensemble definitions by dependencies.

    Each ensemble's inputs may reference other ensembles. This resolves
    the order so that dependencies are built before the ensembles that
    need them.
    """
    if targets is None:
        targets = set(defs.keys())
    else:
        targets = set(targets)

    # Expand targets to include transitive dependencies.
    needed = set()
    stack = list(targets)
    while stack:
        name = stack.pop()
        if name in needed:
            continue
        needed.add(name)
        if name in defs:
            for inp in defs[name]["inputs"]:
                if inp in defs:
                    stack.append(inp)

    # Topological sort via DFS.
    order = []
    visited = set()

    def visit(name):
        if name in visited or name not in defs or name not in needed:
            return
        visited.add(name)
        for inp in defs[name]["inputs"]:
            visit(inp)
        order.append(name)

    for name in needed:
        visit(name)

    return order


def build_ensemble(name, spec, y, built):
    """
    Build a single ensemble from its definition spec.

    Dispatches to the appropriate blending function based on the
    method field. Returns (oof, test) prediction arrays.
    """
    method = spec["method"]
    inputs = spec["inputs"]

    oofs = [load_predictions(inp, built)[0] for inp in inputs]
    tests = [load_predictions(inp, built)[1] for inp in inputs]

    if method == "rank_blend":
        normalize = spec.get("normalize", True)
        oof = rank_blend(oofs, normalize=normalize)
        test = rank_blend(tests, normalize=normalize)

    elif method == "weighted_rank_blend":
        weights = spec["weights"]
        normalize = spec.get("normalize", False)
        oof = rank_blend(oofs, weights=weights, normalize=normalize)
        test = rank_blend(tests, weights=weights, normalize=normalize)

    elif method == "band_gate":
        params = spec.get("params", {})
        normalize_anchor = spec.get("normalize_anchor", False)

        anchor_oof, anchor_test = oofs[0], tests[0]
        div_oof, div_test = oofs[1], tests[1]

        if normalize_anchor:
            def _normalize(arr):
                lo, hi = arr.min(), arr.max()
                return (arr - lo) / (hi - lo) if hi > lo else arr
            anchor_oof = _normalize(anchor_oof)
            anchor_test = _normalize(anchor_test)
            div_oof = _normalize(div_oof)
            div_test = _normalize(div_test)

        oof = band_gate_blend(anchor_oof, div_oof, **params)
        test = band_gate_blend(anchor_test, div_test, **params)

    elif method == "hillclimb":
        # For hillclimb, inputs are a candidate pool, not a fixed list.
        oof_dict = {inp: load_predictions(inp, built)[0] for inp in inputs}
        selected, oof, auc = hillclimb(oof_dict, y)
        # Build test from the selected models.
        test_arrays = [load_predictions(s, built)[1] for s in selected]
        test = np.mean([rankdata(t) for t in test_arrays], axis=0)

    else:
        raise ValueError(f"Unknown ensemble method: {method}")

    # Verify against expected AUC if provided.
    expected = spec.get("expected_auc")
    if expected is not None:
        actual = roc_auc_score(y, oof)
        diff = abs(actual - expected)
        tag = "EXACT" if diff < 1e-10 else f"DIFF={diff:.2e}"
        print(f"  {name}: AUC={actual:.10f} (expected {expected:.10f}) [{tag}]")
    else:
        actual = roc_auc_score(y, oof)
        print(f"  {name}: AUC={actual:.10f}")

    return oof, test


def main():
    parser = argparse.ArgumentParser(description="Build ensembles from definitions")
    parser.add_argument("--only", nargs="+",
                        help="Build only these ensembles (plus dependencies)")
    parser.add_argument("--verify-only", action="store_true",
                        help="Build and verify but don't save artifacts")
    parser.add_argument("--list", action="store_true",
                        help="List available ensemble definitions")
    args = parser.parse_args()

    if args.list:
        for name, spec in ENSEMBLE_DEFS.items():
            inputs = ", ".join(spec["inputs"])
            print(f"  {name} [{spec['method']}] <- {inputs}")
        return

    train_df, _ = load_train_test()
    y = (train_df["Heart Disease"] == "Presence").astype(int).values

    targets = args.only if args.only else None
    build_order = resolve_build_order(ENSEMBLE_DEFS, targets)

    print(f"Building {len(build_order)} ensembles...")
    built = {}

    for name in build_order:
        spec = ENSEMBLE_DEFS[name]
        oof, test = build_ensemble(name, spec, y, built)
        built[name] = (oof, test)

    if not args.verify_only:
        print("\nSaving artifacts...")
        for name, (oof, test) in built.items():
            save_oof(name, oof)
            save_test_preds(name, test)
            metrics = compute_metrics(y, oof)
            save_metrics(name, metrics, {"type": "ensemble"})
            print(f"  {name}: saved")

    print("\nDone.")


if __name__ == "__main__":
    main()
