"""Classification metrics following the project metrics policy (RULES.md)."""

from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score


def compute_metrics(y_true, y_prob, threshold=0.5):
    """Compute AUC, accuracy, precision, recall, F1 at the given threshold.

    If AUC < 0.5 the model is predicting inverted classes, so we flip
    both the AUC and the binary predictions before computing the rest.
    """
    y_pred = (y_prob > threshold).astype(int)
    auc = float(roc_auc_score(y_true, y_prob))

    # Inverted predictions: model learned the wrong class mapping
    if auc < 0.5:
        auc = 1 - auc
        y_pred = 1 - y_pred

    return {
        "auc": auc,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "threshold": threshold,
    }
