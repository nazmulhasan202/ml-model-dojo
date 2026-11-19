import numpy as np
import pytest

from ml_model_dojo import metrics as M


def test_classification_hand_computed():
    y = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
    pred = np.array([1, 1, 0, 0, 1, 0, 0, 0, 0, 0])
    p = np.where(pred == 1, 0.9, 0.1)
    m = M.classification_metrics(y, pred, p)
    assert m["accuracy"] == pytest.approx(0.7)
    assert m["precision"] == pytest.approx(2 / 3)
    assert m["recall"] == pytest.approx(0.5)
    assert m["f1"] == pytest.approx(2 * (2 / 3) * 0.5 / (2 / 3 + 0.5))
    assert m["balanced_accuracy"] == pytest.approx((0.5 + 5 / 6) / 2)
    assert m["confusion"] == [[5, 1], [2, 2]]
    assert m["baseline"] == pytest.approx(0.6)


def test_log_loss_value():
    y = np.array([1, 0])
    m = M.classification_metrics(y, np.array([1, 0]), np.array([0.9, 0.2]))
    assert m["log_loss"] == pytest.approx(-(np.log(0.9) + np.log(0.8)) / 2)


def test_auc_perfect_and_reversed():
    y = np.array([0, 0, 1, 1])
    assert M.classification_metrics(y, y, np.array([0.1, 0.2, 0.8, 0.9]))["roc_auc"] == pytest.approx(1.0)
    assert M.classification_metrics(y, 1 - y, np.array([0.9, 0.8, 0.2, 0.1]))["roc_auc"] == pytest.approx(0.0)


def test_majority_baseline_and_accuracy_trap():
    y = np.r_[np.zeros(95, int), np.ones(5, int)]
    m = M.classification_metrics(y, np.zeros(100, int), np.full(100, 0.1))
    assert m["accuracy"] == 0.95 and m["baseline"] == 0.95
    assert m["recall"] == 0.0 and m["f1"] == 0.0 and m["balanced_accuracy"] == 0.5


def test_regression_hand_computed():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    p = np.array([1.0, 2.0, 3.0, 6.0])
    m = M.regression_metrics(y, p)
    assert m["mae"] == pytest.approx(0.5)
    assert m["rmse"] == pytest.approx(1.0)
    assert m["r2"] == pytest.approx(1 - 4 / 5)


def test_gap_and_overfit_flag():
    tr, te = {"accuracy": 1.0}, {"accuracy": 0.8}
    assert M.gap(tr, te, "classification") == pytest.approx(0.2)
    assert M.is_overfit(tr, te, "classification")
    assert not M.is_overfit({"r2": 0.9}, {"r2": 0.85}, "regression")
