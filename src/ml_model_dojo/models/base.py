"""Common interface for models.

A ``ModelSpec`` describes a model family (parameters, concept card, facts).
A ``Run`` is one trained instance on one dataset. Runs support staged
reveal: ``fit_initial`` does the heavy work once, then ``step`` moves one
stage forward (one tree, one epoch, one split). Non-iterative models have a
single stage.

Animation strategy per model is documented in each model module.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
from sklearn.metrics import accuracy_score, log_loss, r2_score

from ..data.datasets import Dataset
from ..seeding import model_seed

CLS, REG = "classification", "regression"


@dataclass
class ParamSpec:
    """One hyperparameter control. The left panel is generated from a list of these."""
    key: str
    label: str
    kind: str                       # int, float, log, choice, bool, layers, flags
    default: Any
    tooltip: str
    lo: float | None = None
    hi: float | None = None
    choices: list[tuple[Any, str]] | None = None
    problems: tuple[str, ...] = (CLS, REG)
    show_if: Callable[[dict], bool] | None = None
    zero_label: str | None = None   # text shown when an int slider is at its minimum
    decimals: int = 2


class ModelSpec:
    id = ""
    display_name = ""
    supports: set[str] = {CLS, REG}
    concept = ""
    interpretability = "Medium"      # general guideline, not a measurement
    curve_xlabel = "Step"
    run_class: type["Run"]

    def _schema(self, problem: str) -> list[ParamSpec]:
        raise NotImplementedError

    def param_schema(self, problem: str) -> list[ParamSpec]:
        return [p for p in self._schema(problem) if problem in p.problems]

    def default_params(self, problem: str) -> dict:
        return {p.key: p.default for p in self.param_schema(problem)}

    def clean_params(self, params: dict, problem: str) -> dict:
        """Fill missing keys with defaults and drop unknown keys."""
        d = self.default_params(problem)
        d.update({k: v for k, v in (params or {}).items() if k in d})
        return d

    def iterative(self, problem: str) -> bool:
        return False

    def start(self, params: dict, problem: str, seed: int, data: Dataset) -> "Run":
        return self.run_class(self, self.clean_params(params, problem), problem, seed, data)

    def summary(self, params: dict, problem: str) -> str:
        raise NotImplementedError

    def build(self, params: dict, problem: str, seed: int, data: Dataset):
        """Return the fitted estimator after running to the end (used by tests and Compare)."""
        run = self.start(params, problem, seed, data)
        run.fit_initial()
        run.go_to_end()
        return run


class Run:
    replayable = False       # True when stages can be replayed without refitting

    def __init__(self, spec: ModelSpec, params: dict, problem: str, seed: int, data: Dataset):
        self.spec, self.params, self.problem = spec, params, problem
        self.seed = seed
        self.est_seed = model_seed(seed)
        self.data = data
        self.stage = 0
        self.n_stages = 1
        self.ready = False
        self.train_time = 0.0
        self.notice = ""
        self.threshold = float(params.get("threshold", 0.5)) if problem == CLS else 0.5
        self.curves: dict[str, list[float]] = {}
        self.curves_full = False     # True when curves cover all stages already

    # ---- lifecycle -------------------------------------------------------
    def fit_initial(self) -> None:
        t0 = time.perf_counter()
        self._fit()
        self.train_time += time.perf_counter() - t0
        self.ready = True

    def step(self) -> bool:
        """Advance one stage. Returns True while more stages remain."""
        if self.stage >= self.n_stages:
            return False
        t0 = time.perf_counter()
        self.stage += 1
        self._advance()
        self.train_time += time.perf_counter() - t0
        if not self.curves_full:
            self._record()
        return self.stage < self.n_stages

    def go_to_end(self) -> None:
        while self.stage < self.n_stages:
            self.step()

    @property
    def finished(self) -> bool:
        return self.ready and self.stage >= self.n_stages

    def reset(self) -> None:
        """Back to stage 0. Only valid for replayable runs."""
        self.stage = 0
        if not self.curves_full:
            self.curves = {}
        self._reset()

    # ---- hooks for subclasses -------------------------------------------
    def _fit(self) -> None: ...
    def _advance(self) -> None: ...
    def _reset(self) -> None: ...

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Class 1 probability (or pseudo probability) for classification, value for regression."""
        raise NotImplementedError

    def facts(self) -> dict:
        return {}

    def extra(self) -> dict:
        """Model-specific data for the details panel."""
        return {}

    def curve_x(self, n: int) -> np.ndarray:
        """x values for the first ``n`` points of the training curves."""
        return np.arange(1, n + 1)

    # ---- helpers ---------------------------------------------------------
    def decide(self, p: np.ndarray) -> np.ndarray:
        return (np.asarray(p) >= self.threshold).astype(int)

    def score_and_loss(self, X, y) -> tuple[float, float]:
        p = self.predict(X)
        if self.problem == CLS:
            return (float(accuracy_score(y, self.decide(p))),
                    float(log_loss(y, np.clip(p, 1e-9, 1 - 1e-9), labels=[0, 1])))
        return float(r2_score(y, p)), float(np.mean((y - p) ** 2))

    def _record(self) -> None:
        d = self.data
        for name, (X, y) in {"train": (d.X_train, d.y_train), "test": (d.X_test, d.y_test)}.items():
            s, l = self.score_and_loss(X, y)
            self.curves.setdefault(f"{name}_score", []).append(s)
            self.curves.setdefault(f"{name}_loss", []).append(l)

    def _timed_predict(self, X):
        return self.predict(X)

    def predict_time(self, X: np.ndarray, repeats: int = 3) -> float:
        best = float("inf")
        for _ in range(repeats):
            t0 = time.perf_counter()
            self._timed_predict(X)
            best = min(best, time.perf_counter() - t0)
        return best


def oversample_minority(X: np.ndarray, y: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Duplicate minority rows (training set only) until the classes are balanced."""
    rng = np.random.default_rng([seed, 909])
    n0, n1 = int((y == 0).sum()), int((y == 1).sum())
    if n0 == 0 or n1 == 0 or n0 == n1:
        return X, y
    minority = 1 if n1 < n0 else 0
    idx = np.flatnonzero(y == minority)
    extra = rng.choice(idx, abs(n0 - n1), replace=True)
    keep = np.r_[np.arange(len(y)), extra]
    return X[keep], y[keep]


def balanced_weights(y: np.ndarray) -> dict[int, float]:
    n = len(y)
    out = {}
    for c in (0, 1):
        k = int((y == c).sum())
        out[c] = n / (2.0 * k) if k else 1.0
    return out


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -50, 50)))


def make_grid(n_inputs: int, res: int) -> tuple[np.ndarray, np.ndarray]:
    """Grid over [-1, 1]. Returns (axis values, points). Points are (res, 1) or (res*res, 2)."""
    g = np.linspace(-1, 1, res)
    if n_inputs == 1:
        return g, g[:, None]
    xx, yy = np.meshgrid(g, g)
    return g, np.c_[xx.ravel(), yy.ravel()]


def choice(values: list[str], labels: list[str] | None = None) -> list[tuple[str, str]]:
    return list(zip(values, labels or values))


def class_weight_param(tooltip_extra: str = "") -> ParamSpec:
    return ParamSpec("class_weight", "Class weight", "choice", "none",
                     "Balanced gives the rare class more influence, which usually raises recall. " + tooltip_extra,
                     choices=choice(["none", "balanced"], ["None", "Balanced"]), problems=(CLS,))


def threshold_param() -> ParamSpec:
    return ParamSpec("threshold", "Decision threshold", "float", 0.5,
                     "Predict class 1 when the probability is at least this value. Lower it to catch more positives.",
                     lo=0.05, hi=0.95, problems=(CLS,))


def standardize_param() -> ParamSpec:
    return ParamSpec("standardize", "Standardize inputs", "bool", True,
                     "Rescale each input to mean 0 and spread 1, using the training data only.")
