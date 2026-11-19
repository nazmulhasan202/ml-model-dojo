"""Comparison jobs. Everything here is picklable and free of Qt, so it can run in a process pool.

A job regenerates the dataset from (config, seed), so no arrays are sent to the worker.
The held-out test set is never used by cross-validation: folds are cut from the training set only.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import replace

import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold

from .. import metrics as M
from ..data.datasets import DataConfig, generate
from ..models import get_spec
from ..models.base import CLS, make_grid
from ..models.evaluate import evaluate

GRID_2D, GRID_1D = 60, 200
CV_FOLDS = 5


def job_key(job: dict) -> str:
    """Cache key: a hash of the data config, seed, model settings and options."""
    return hashlib.sha1(json.dumps(job, sort_keys=True, default=str).encode()).hexdigest()


def make_job(config: DataConfig, seed: int, model_id: str, params: dict, main: bool, cv: bool) -> dict:
    return {"config": config.to_dict(), "seed": int(seed), "model_id": model_id, "params": params,
            "main": bool(main), "cv": bool(cv and main)}


def _fit(spec, params, problem, seed, data):
    run = spec.start(params, problem, seed, data)
    run.fit_initial()
    run.go_to_end()
    return run


def cross_validate(spec, params, problem, seed, data, k: int = CV_FOLDS) -> dict:
    """k-fold cross-validation on the training set only."""
    X, y = data.X_train, data.y_train
    if problem == CLS:
        splits = StratifiedKFold(k, shuffle=True, random_state=seed).split(X, y)
    else:
        splits = KFold(k, shuffle=True, random_state=seed).split(X)
    key = M.main_score_key(problem)
    scores = []
    for tr, va in splits:
        fold = replace(data, X_train=X[tr], y_train=y[tr], X_test=X[va], y_test=y[va])
        scores.append(evaluate(_fit(spec, params, problem, seed, fold))["test"][key])
    return {"key": key, "scores": scores, "mean": float(np.mean(scores)), "std": float(np.std(scores))}


def run_job(job: dict) -> dict:
    """Train one model for one seed and return a plain-data result."""
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(v, "1")
    cfg = DataConfig.from_dict(job["config"])
    data = generate(cfg, job["seed"])
    spec = get_spec(job["model_id"])
    run = _fit(spec, job["params"], cfg.problem, job["seed"], data)
    ev = evaluate(run)
    out = {
        "model_id": job["model_id"], "seed": job["seed"], "main": job["main"],
        "train": {k: v for k, v in ev["train"].items() if k != "pred"},
        "test": {k: v for k, v in ev["test"].items() if k != "pred"},
        "train_time": run.train_time, "predict_time": run.predict_time(data.X_test),
        "facts": run.facts(), "notice": run.notice, "surface": None, "cv": None,
    }
    if job["main"]:
        out["pred_train"], out["pred_test"] = ev["train"]["pred"], ev["test"]["pred"]
        g, G = make_grid(data.n_inputs, GRID_1D if data.n_inputs == 1 else GRID_2D)
        out["surface"] = (g, run.predict(G))
        out["threshold"] = run.threshold
        out["curves"] = {k: list(v) for k, v in run.curves.items()} if run.curves_full else None
    if job["cv"]:
        out["cv"] = cross_validate(spec, job["params"], cfg.problem, job["seed"], data)
    return out


def warm_up() -> bool:
    """Import the heavy libraries in a worker so the first real job starts quickly."""
    import sklearn.ensemble  # noqa: F401
    import sklearn.neural_network  # noqa: F401
    import xgboost  # noqa: F401
    return True
