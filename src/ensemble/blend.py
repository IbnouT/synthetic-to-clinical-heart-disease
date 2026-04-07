"""Ensemble blending functions for combining model predictions.

Three blending strategies:
  - Rank averaging (equal or weighted)
  - Greedy hill-climbing (forward selection maximizing AUC)
  - Band-gated blending (selective injection in uncertain regions)

All functions operate on prediction arrays (OOF or test). Rank-based
methods convert to ranks first, which normalizes different model output
scales and improves AUC-oriented blending.
"""

import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score


# ---------------------------------------------------------------------------
# Rank-based weighted blend
# ---------------------------------------------------------------------------

def rank_blend(arrays, weights=None, normalize=True):
    """Weighted rank average of prediction arrays.

    Each input array is converted to ranks (1..N), scaled by its weight,
    and summed. The result can optionally be min-max normalized to [0, 1].

    Rank transformation ensures models with different output scales
    (probabilities vs. raw scores) contribute equally to the blend.
    Used for multi-seed averaging and ensemble-level blending.

    Parameters
    ----------
    arrays : list of np.ndarray
        Prediction arrays to blend (all same length).
    weights : list of float or None
        Per-array weights. If None, equal weights are used.
    normalize : bool
        If True, min-max normalize the output to [0, 1].
        Blends are typically normalized; hillclimb keeps raw ranks.

    Returns
    -------
    np.ndarray
        Blended prediction array.
    """
    if weights is None:
        weights = [1.0] * len(arrays)
    weights = np.array(weights, dtype=np.float64)
    weights /= weights.sum()

    blended = np.zeros(len(arrays[0]), dtype=np.float64)
    for arr, w in zip(arrays, weights):
        blended += rankdata(arr) * w

    if normalize:
        lo, hi = blended.min(), blended.max()
        if hi > lo:
            blended = (blended - lo) / (hi - lo)

    return blended


# ---------------------------------------------------------------------------
# Greedy hill-climbing
# ---------------------------------------------------------------------------

def hillclimb(oof_dict, y, threshold=1e-7):
    """Greedy forward selection to maximize OOF AUC via rank averaging.

    Starts with the best single model and iteratively adds the candidate
    that produces the largest AUC gain, stopping when no candidate
    improves by more than the threshold.

    The output is a raw rank-averaged blend (not normalized) to
    preserve the rank scale for downstream weighted blending.

    Parameters
    ----------
    oof_dict : dict of str -> np.ndarray
        Mapping from model name to OOF prediction array.
    y : np.ndarray
        Binary ground truth labels (0/1).
    threshold : float
        Minimum AUC improvement to keep adding models.

    Returns
    -------
    selected : list of str
        Model names in the order they were selected.
    blend : np.ndarray
        Final blended OOF predictions (raw rank scale).
    auc : float
        Final OOF AUC of the blend.
    """
    ranked = {name: rankdata(oof) for name, oof in oof_dict.items()}

    # Start with the best single model by AUC
    best_name = max(oof_dict, key=lambda k: roc_auc_score(y, oof_dict[k]))
    selected = [best_name]
    current_blend = ranked[best_name].copy()
    best_auc = roc_auc_score(y, current_blend)

    print(f"Start: {best_name} = {best_auc:.6f}", flush=True)

    improved = True
    while improved:
        improved = False
        best_candidate = None
        best_new_auc = best_auc

        n_sel = len(selected)
        for name in oof_dict:
            if name in selected:
                continue
            trial = (current_blend * n_sel + ranked[name]) / (n_sel + 1)
            auc = roc_auc_score(y, trial)
            if auc > best_new_auc + threshold:
                best_new_auc = auc
                best_candidate = name

        if best_candidate:
            n_sel = len(selected)
            current_blend = (current_blend * n_sel + ranked[best_candidate]) / (n_sel + 1)
            selected.append(best_candidate)
            best_auc = best_new_auc
            print(f"  + {best_candidate} -> {best_auc:.6f} ({len(selected)} models)",
                  flush=True)
            improved = True

    print(f"Final: {len(selected)} models, AUC={best_auc:.15f}", flush=True)
    return selected, current_blend, best_auc


# ---------------------------------------------------------------------------
# Band-gated blending
# ---------------------------------------------------------------------------

def band_gate_blend(anchor, diversity, weight=0.05, lo=0.08, hi=0.32):
    """Selective blending within an uncertainty band.

    Only modifies the anchor predictions where they fall inside the
    [lo, hi] band. Outside this range (where the model is confident),
    predictions are left unchanged. Inside the band, a small weight of
    the diversity model's predictions is mixed in.

    This captures the idea that confident predictions near 0 or 1
    should not be perturbed, while uncertain predictions near the
    decision boundary can benefit from a complementary signal.

    Parameters
    ----------
    anchor : np.ndarray
        Primary prediction array (must be in [0, 1] probability scale).
    diversity : np.ndarray
        Diverse model prediction array (same scale as anchor).
    weight : float
        Blend weight for diversity model within the gated band.
    lo : float
        Lower bound of the uncertainty band.
    hi : float
        Upper bound of the uncertainty band.

    Returns
    -------
    np.ndarray
        Blended predictions (same scale as anchor).
    """
    anchor = np.asarray(anchor, dtype=np.float64)
    diversity = np.asarray(diversity, dtype=np.float64)

    gate_mask = (anchor >= lo) & (anchor <= hi)
    result = anchor.copy()
    result[gate_mask] = (
        anchor[gate_mask] + weight * (diversity[gate_mask] - anchor[gate_mask])
    )
    return result
