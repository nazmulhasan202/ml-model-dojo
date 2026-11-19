"""Metrics for classification and regression, with safe handling of edge cases."""
from __future__ import annotations

import numpy as np
from sklearn import metrics as skm

CLS_KEYS = ["accuracy", "precision", "recall", "f1", "balanced_accuracy", "roc_auc", "log_loss"]
REG_KEYS = ["r2", "rmse", "mae"]

LABELS = {
    "accuracy": "Accuracy", "precision": "Precision", "recall": "Recall", "f1": "F1",
    "balanced_accuracy": "Balanced accuracy", "roc_auc": "ROC AUC", "log_loss": "Log loss",
    "r2": "R squared", "rmse": "RMSE", "mae": "MAE",
}
# Metrics where a smaller number is better.
LOWER_IS_BETTER = {"log_loss", "rmse", "mae"}

GLOSSARY = {
    "accuracy": "Share of all predictions that are correct. Example: 90 of 100 correct gives 0.90, but a model that always predicts the majority class of a 95/5 split also scores 0.95.",
    "precision": "Of the points predicted positive, the share that really are positive. Example: 8 true positives and 2 false positives give 8/10 = 0.80.",
    "recall": "Of the truly positive points, the share that the model finds. Example: 8 found out of 16 positives gives 0.50.",
    "f1": "The harmonic mean of precision and recall. Example: precision 0.80 and recall 0.50 give 2*0.8*0.5/1.3 = 0.62.",
    "balanced_accuracy": "The average of recall on each class, so rare classes count as much as common ones. Example: recall 1.0 on class 0 and 0.0 on class 1 gives 0.50.",
    "roc_auc": "The chance that a random positive point is scored higher than a random negative point. 0.5 is guessing and 1.0 is perfect ranking.",
    "log_loss": "Average penalty for confident wrong probabilities. Predicting 0.9 for a true positive costs 0.105, while predicting 0.1 costs 2.30.",
    "r2": "Share of the variance in the target that the model explains. 1 is perfect, 0 equals predicting the mean, and negative is worse than the mean.",
    "rmse": "Root mean squared error, in the units of the target. Errors of 1 and 3 give sqrt((1+9)/2) = 2.24. It punishes large errors.",
    "mae": "Mean absolute error, in the units of the target. Errors of 1 and 3 give 2.0. It treats all errors equally.",
}


def classification_metrics(y_true, y_pred, p1) -> dict:
    """Binary metrics. ``p1`` is the score or probability for class 1."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    p1 = np.clip(np.asarray(p1, float), 1e-12, 1 - 1e-12)
    out = {
        "accuracy": float(skm.accuracy_score(y_true, y_pred)),
        "precision": float(skm.precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(skm.recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(skm.f1_score(y_true, y_pred, zero_division=0)),
        "balanced_accuracy": float(skm.balanced_accuracy_score(y_true, y_pred)),
    }
    if len(np.unique(y_true)) == 2:
        out["roc_auc"] = float(skm.roc_auc_score(y_true, p1))
    else:
        out["roc_auc"] = float("nan")
    out["log_loss"] = float(skm.log_loss(y_true, p1, labels=[0, 1]))
    out["confusion"] = skm.confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()
    out["baseline"] = majority_baseline(y_true)
    return out


def majority_baseline(y_true) -> float:
    """Accuracy of always predicting the most common class."""
    y_true = np.asarray(y_true).astype(int)
    if len(y_true) == 0:
        return float("nan")
    return float(np.bincount(y_true).max() / len(y_true))


def regression_metrics(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    return {
        "r2": float(skm.r2_score(y_true, y_pred)),
        "rmse": float(np.sqrt(np.mean((y_true - y_pred) ** 2))),
        "mae": float(np.mean(np.abs(y_true - y_pred))),
    }


def main_score_key(problem: str) -> str:
    return "accuracy" if problem == "classification" else "r2"


def gap(train: dict, test: dict, problem: str) -> float:
    """Train minus test for the main score (accuracy or R squared)."""
    k = main_score_key(problem)
    return float(train[k] - test[k])


OVERFIT_THRESHOLD = 0.10


def is_overfit(train: dict, test: dict, problem: str) -> bool:
    return gap(train, test, problem) > OVERFIT_THRESHOLD
