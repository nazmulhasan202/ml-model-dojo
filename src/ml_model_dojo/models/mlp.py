"""Neural network (multilayer perceptron) from scikit-learn.

Animation: ``partial_fit`` is called once per epoch, and one stage runs
``epochs_per_step`` epochs. Input feature toggles (x1, x2, squares, product, sine)
are applied by a transformer fitted before training, then optionally scaled.
"""
from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import StandardScaler

from .base import CLS, REG, ModelSpec, ParamSpec, Run, choice, oversample_minority, standardize_param, threshold_param

FEATURES_2D = [("x1", "x1"), ("x2", "x2"), ("x1sq", "x1 squared"), ("x2sq", "x2 squared"),
               ("x1x2", "x1 times x2"), ("sinx1", "sin(x1)"), ("sinx2", "sin(x2)")]
FEATURES_1D = [("x1", "x"), ("x1sq", "x squared"), ("sinx1", "sin(x)")]
DEFAULT_FEATURES = ["x1", "x2"]


def feature_options(n_inputs: int) -> list[tuple[str, str]]:
    return FEATURES_2D if n_inputs == 2 else FEATURES_1D


class FeatureTransformer(BaseEstimator, TransformerMixin):
    """Builds the engineered inputs chosen by the toggles."""

    def __init__(self, features=("x1", "x2")):
        self.features = features

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        x1 = X[:, 0]
        x2 = X[:, 1] if X.shape[1] > 1 else np.zeros(len(X))
        table = {"x1": x1, "x2": x2, "x1sq": x1 ** 2, "x2sq": x2 ** 2, "x1x2": x1 * x2,
                 "sinx1": np.sin(x1), "sinx2": np.sin(x2)}
        cols = [table[f] for f in self.features if f in table]
        return np.column_stack(cols)


class MlpRun(Run):
    def _fit(self) -> None:
        p, d = self.params, self.data
        valid = {k for k, _ in feature_options(d.n_inputs)}
        feats = [f for f in p["features"] if f in valid] or (["x1", "x2"] if d.n_inputs == 2 else ["x1"])
        self.ft = FeatureTransformer(tuple(feats))
        X, y = d.X_train, d.y_train
        if self.problem == CLS and p["class_weight"] == "balanced":
            X, y = oversample_minority(X, y, self.seed)
        F = self.ft.transform(X)
        self.scaler = StandardScaler().fit(F) if p["standardize"] else None
        self.Z, self.y = self._scale(F), y
        common = dict(hidden_layer_sizes=tuple(int(n) for n in p["layers"]), activation=p["activation"],
                      solver=p["solver"], learning_rate_init=float(p["lr"]), alpha=float(p["alpha"]),
                      batch_size=min(int(p["batch_size"]), len(self.y)), random_state=self.est_seed,
                      shuffle=True, max_iter=1)
        self.est = MLPClassifier(**common) if self.problem == CLS else MLPRegressor(**common)
        self.eps = int(p["epochs_per_step"])
        self.n_stages = int(np.ceil(int(p["max_epochs"]) / self.eps))
        self.epochs = 0
        self.initialized = False

    def _scale(self, F):
        return self.scaler.transform(F) if self.scaler is not None else F

    def _advance(self) -> None:
        import warnings
        from sklearn.exceptions import ConvergenceWarning
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            for _ in range(self.eps):
                if self.problem == CLS:
                    self.est.partial_fit(self.Z, self.y, classes=np.array([0, 1]))
                else:
                    self.est.partial_fit(self.Z, self.y)
                self.epochs += 1

    def predict(self, X):
        Z = self._scale(self.ft.transform(X))
        if self.problem == CLS:
            return self.est.predict_proba(Z)[:, 1]
        return self.est.predict(Z)

    def facts(self) -> dict:
        n_par = 0
        if hasattr(self.est, "coefs_"):
            n_par = sum(w.size for w in self.est.coefs_) + sum(b.size for b in self.est.intercepts_)
        return {"Epochs": self.epochs, "Parameters": int(n_par), "Input features": len(self.ft.features)}

    def curve_x(self, n):
        return np.arange(1, n + 1) * self.eps

    def extra(self) -> dict:
        return {"features": list(self.ft.features), "layers": [int(n) for n in self.params["layers"]],
                "coefs": list(getattr(self.est, "coefs_", [])) or None}


class MlpSpec(ModelSpec):
    id = "mlp"
    display_name = "Neural network"
    supports = {CLS, REG}
    interpretability = "Low"
    concept = ("Layers of weighted sums and bends learn their own features from the inputs. "
               "It shines on complex, smooth patterns when there is enough data. "
               "It struggles on tiny data (it overfits), needs tuning and scaling, and trains more slowly.")
    run_class = MlpRun
    curve_xlabel = "Epoch"

    def iterative(self, problem):
        return True

    def _schema(self, problem):
        return [
            ParamSpec("features", "Input features", "flags", list(DEFAULT_FEATURES),
                      "Choose which inputs the network sees. Engineered features such as x1 times x2 can replace hidden capacity."),
            ParamSpec("layers", "Hidden layers", "layers", [8, 8],
                      "Use plus and minus to add layers (1 to 5) and neurons (1 to 16). Bigger networks can fit more, and overfit more.",
                      lo=1, hi=16),
            ParamSpec("activation", "Activation", "choice", "tanh", "The bend applied after each weighted sum.",
                      choices=choice(["relu", "tanh", "logistic"], ["ReLU", "Tanh", "Logistic"])),
            ParamSpec("solver", "Solver", "choice", "adam", "How the weights are updated. Adam adapts step sizes, SGD uses a fixed step.",
                      choices=choice(["adam", "sgd"], ["Adam", "SGD"])),
            ParamSpec("lr", "Learning rate", "log", 0.01, "Size of each weight update.", lo=1e-4, hi=1.0),
            ParamSpec("batch_size", "Batch size", "choice", 32, "Rows per update. Smaller batches give noisier but more frequent updates.",
                      choices=[(b, str(b)) for b in (8, 16, 32, 64, 128, 256)]),
            ParamSpec("alpha", "L2 regularization (alpha)", "log", 1e-4, "Penalty on large weights. Larger values give smoother boundaries.", lo=1e-6, hi=1.0),
            ParamSpec("epochs_per_step", "Epochs per step", "int", 5, "How many passes over the data happen in one animation step.", lo=1, hi=50),
            ParamSpec("max_epochs", "Max epochs", "int", 300, "Training stops after this many epochs.", lo=10, hi=2000),
            standardize_param(),
            ParamSpec("class_weight", "Class weight", "choice", "none",
                      "MLP has no class weights, so Balanced duplicates minority points in the training set only (oversampling).",
                      choices=choice(["none", "balanced"], ["None", "Balanced (oversample)"]), problems=(CLS,)),
            threshold_param(),
        ]

    def summary(self, p, problem):
        return f"layers = {'-'.join(str(int(n)) for n in p['layers'])}, {p['activation']}, {p['solver']}"
