"""Evaluate a run on its train and test sets."""
from __future__ import annotations

import numpy as np

from .. import metrics as M
from .base import CLS, Run


def evaluate(run: Run) -> dict:
    """Return {"train": {...}, "test": {...}} with metrics and the predictions used."""
    d = run.data
    out = {}
    for name, X, y in (("train", d.X_train, d.y_train), ("test", d.X_test, d.y_test)):
        p = run.predict(X)
        if run.problem == CLS:
            pred = run.decide(p)
            m = M.classification_metrics(y, pred, p)
        else:
            pred = p
            m = M.regression_metrics(y, p)
        m["pred"] = np.asarray(pred)
        out[name] = m
    return out


def summarize(run: Run) -> dict:
    """Metrics plus timing and facts, without the bulky arrays (safe to pickle and tabulate)."""
    res = evaluate(run)
    for m in res.values():
        m.pop("pred", None)
    res["train_time"] = run.train_time
    res["predict_time"] = run.predict_time(run.data.X_test)
    res["facts"] = run.facts()
    return res
