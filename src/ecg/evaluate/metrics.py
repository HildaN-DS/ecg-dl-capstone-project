
"""Reusable evaluation metrics for multi-label ECG classification."""

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    log_loss,
    roc_auc_score,
)


CLASSES = ("NORM", "MI", "STTC", "CD", "HYP")


def evaluate_multilabel(
    y_true,
    y_prob,
    threshold=0.5,
    class_names=CLASSES,
):
    """Evaluate multi-label predictions using a fixed threshold.

    Parameters
    ----------
    y_true : array-like, shape (n_samples, n_classes)
        Binary diagnostic targets.
    y_prob : array-like, shape (n_samples, n_classes)
        Predicted probabilities between zero and one.
    threshold : float
        Decision threshold applied independently to each class.
    class_names : sequence of str
        Names corresponding to the columns of both arrays.

    Returns
    -------
    dict
        Summary metrics and per-class results.
    """

    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    class_names = tuple(class_names)

    if y_true.ndim != 2 or y_prob.ndim != 2:
        raise ValueError("Targets and probabilities must be 2D.")

    if y_true.shape != y_prob.shape:
        raise ValueError("Target/probability shapes do not match.")

    if y_true.shape[0] == 0:
        raise ValueError("Cannot evaluate an empty dataset.")

    if y_true.shape[1] != len(class_names):
        raise ValueError("Incorrect number of class names.")

    if len(set(class_names)) != len(class_names):
        raise ValueError("Class names must be unique.")

    if not np.isfinite(y_true).all():
        raise ValueError("Targets contain non-finite values.")

    if not np.isin(y_true, [0, 1]).all():
        raise ValueError("Targets must be binary.")

    if not np.isfinite(y_prob).all():
        raise ValueError("Probabilities contain non-finite values.")

    if ((y_prob < 0) | (y_prob > 1)).any():
        raise ValueError("Probabilities must be in [0, 1].")

    if not np.isfinite(threshold) or not 0 < threshold < 1:
        raise ValueError("Threshold must be between 0 and 1.")

    y_true = y_true.astype(np.int64)
    y_pred = (y_prob >= threshold).astype(np.int64)

    per_class = []

    for i, name in enumerate(class_names):
        actual = y_true[:, i]
        probabilities = y_prob[:, i]
        predicted = y_pred[:, i]

        positive_count = int(actual.sum())

        # Ranking metrics require both positive and negative
        # observations. Mark undefined metrics explicitly.
        has_both_classes = (
            0 < positive_count < len(actual)
        )

        ap = (
            float(average_precision_score(
                actual, probabilities
            ))
            if has_both_classes
            else float("nan")
        )

        auc = (
            float(roc_auc_score(
                actual, probabilities
            ))
            if has_both_classes
            else float("nan")
        )

        per_class.append({
            "class": name,
            "support": positive_count,
            "prevalence": float(actual.mean()),
            "ap": ap,
            "roc_auc": auc,
            "f1": float(f1_score(
                actual,
                predicted,
                zero_division=0,
            )),
            "log_loss": float(log_loss(
                actual,
                probabilities,
                labels=[0, 1],
            )),
        })

    ap_values = np.array([
        row["ap"] for row in per_class
    ])

    auc_values = np.array([
        row["roc_auc"] for row in per_class
    ])

    if not np.isfinite(ap_values).any():
        raise ValueError("AP undefined for all classes.")

    if not np.isfinite(auc_values).any():
        raise ValueError("ROC-AUC undefined for all classes.")

    summary = {
        "n_samples": int(len(y_true)),
        "threshold": float(threshold),
        "macro_ap": float(np.nanmean(ap_values)),
        "macro_roc_auc": float(np.nanmean(auc_values)),
        "macro_f1": float(f1_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        )),
        "micro_f1": float(f1_score(
            y_true,
            y_pred,
            average="micro",
            zero_division=0,
        )),
        "mean_binary_log_loss": float(np.mean([
            row["log_loss"] for row in per_class
        ])),
    }

    return {
        "summary": summary,
        "per_class": per_class,
    }
