"""Zero-shot UCI evaluation.

Applies the full trained model (expansion + backbone + head) directly
to UCI data. No UCI labels are used — this tests whether the competition-
trained model generalizes to clinical data as-is.
"""

from .metrics import compute_metrics
from ..training.utils import predict


def evaluate_zeroshot(model, uci_datasets, device):
    """Run the trained model on each UCI dataset, return metrics per dataset."""
    results = {}
    for name, data in uci_datasets.items():
        probs = predict(model, data["X"], device)
        m = compute_metrics(data["y"], probs)
        m["n_samples"] = data["n"]
        results[name] = m
        print(f"    {name}: AUC={m['auc']:.4f}  Acc={m['accuracy']:.3f}")
    return results
