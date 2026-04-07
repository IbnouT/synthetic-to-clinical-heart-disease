"""Metric computation for model evaluation.

AUC is the primary evaluation metric. Accuracy, precision, recall,
and F1 are included for comparison with benchmark papers that report
threshold-based metrics on the UCI heart disease datasets.
"""

from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score

from src.config import DEFAULT_THRESHOLD


def compute_metrics(y_true, y_score, threshold=DEFAULT_THRESHOLD):
    """Compute AUC and threshold-based metrics from true labels and scores."""
    # Binarize scores at the given threshold for accuracy/precision/recall/F1.
    y_pred = (y_score >= threshold).astype(int)

    return {
        "auc": float(roc_auc_score(y_true, y_score)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "threshold": float(threshold),
    }
