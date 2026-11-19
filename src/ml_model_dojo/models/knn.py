"""k-nearest neighbors. Non-iterative: one stage. Fit stores the training set."""
from __future__ import annotations

import numpy as np
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.preprocessing import StandardScaler

from .base import CLS, REG, ModelSpec, ParamSpec, Run, choice, oversample_minority, standardize_param, threshold_param


class KnnRun(Run):
    def _fit(self) -> None:
        p, d = self.params, self.data
        X, y = d.X_train, d.y_train
        if self.problem == CLS and p["class_weight"] == "balanced":
            X, y = oversample_minority(X, y, self.seed)
        self.X_fit = X
        self.scaler = StandardScaler().fit(X) if p["standardize"] else None
        k = min(int(p["k"]), len(X))
        cls = KNeighborsClassifier if self.problem == CLS else KNeighborsRegressor
        self.est = cls(n_neighbors=k, metric=p["metric"], weights=p["weights"]).fit(self._t(X), y)
        self.n_stages = 1

    def _t(self, X):
        return self.scaler.transform(X) if self.scaler is not None else X

    def predict(self, X):
        if self.problem == CLS:
            proba = self.est.predict_proba(self._t(X))
            return proba[:, 1] if proba.shape[1] == 2 else np.zeros(len(X))
        return self.est.predict(self._t(X))

    def neighbors(self, point: np.ndarray) -> np.ndarray:
        """Training points (original units) that are the k nearest to ``point``."""
        idx = self.est.kneighbors(self._t(np.atleast_2d(point)), return_distance=False)[0]
        return self.X_fit[idx]

    def facts(self) -> dict:
        return {"Stored training points": int(len(self.X_fit)), "k": int(self.est.n_neighbors)}


class KnnSpec(ModelSpec):
    id = "knn"
    display_name = "kNN"
    supports = {CLS, REG}
    interpretability = "Medium"
    concept = ("Predicts from the k closest training points. It shines when the boundary is irregular and data are plentiful. "
               "It struggles with tiny noisy data at small k (it memorizes), with many irrelevant inputs, and with slow prediction on large data.")
    run_class = KnnRun

    def _schema(self, problem):
        return [
            ParamSpec("k", "Neighbors (k)", "int", 5,
                      "How many nearby training points vote. k = 1 memorizes the training data.", lo=1, hi=50),
            ParamSpec("metric", "Distance", "choice", "euclidean", "How distance between points is measured.",
                      choices=choice(["euclidean", "manhattan"], ["Euclidean", "Manhattan"])),
            ParamSpec("weights", "Weighting", "choice", "uniform",
                      "Distance weighting lets closer neighbors count more.",
                      choices=choice(["uniform", "distance"], ["Uniform", "Distance"])),
            standardize_param(),
            ParamSpec("class_weight", "Class weight", "choice", "none",
                      "kNN has no class weights, so Balanced duplicates minority points in the training set only (oversampling).",
                      choices=choice(["none", "balanced"], ["None", "Balanced (oversample)"]), problems=(CLS,)),
            threshold_param(),
        ]

    def summary(self, p, problem):
        return f"k = {p['k']}, {p['metric']}, {p['weights']}"
