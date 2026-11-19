"""Support vector machine.

Non-iterative (one stage). The heatmap uses the decision function passed through
a sigmoid, which avoids the slow ``probability=True`` option. The threshold
slider shifts the decision offset. For more than ``APPROX_LIMIT`` training
points an approximation is used (Nystroem features plus a linear SVM).
"""
from __future__ import annotations

import numpy as np
from sklearn.kernel_approximation import Nystroem
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR, LinearSVC, LinearSVR

from .base import CLS, REG, ModelSpec, ParamSpec, Run, choice, class_weight_param, sigmoid, standardize_param, threshold_param

APPROX_LIMIT = 3000


class SvmRun(Run):
    def _fit(self) -> None:
        p, d = self.params, self.data
        self.scaler = StandardScaler().fit(d.X_train) if p["standardize"] else None
        Z = self._t(d.X_train)
        kernel, C = p["kernel"], float(p["C"])
        gamma, deg = float(p["gamma"]), int(p["degree"])
        self.approx = len(Z) > APPROX_LIMIT and kernel != "linear"
        cw = "balanced" if (self.problem == CLS and p["class_weight"] == "balanced") else None
        if self.approx:
            self.notice = (f"Approximation in use: more than {APPROX_LIMIT:,} training points, so the {kernel} kernel "
                           "is approximated with Nystroem features and a linear SVM.")
            nys = Nystroem(kernel=kernel, gamma=gamma, degree=deg, n_components=300, random_state=self.est_seed)
            lin = (LinearSVC(C=C, class_weight=cw, random_state=self.est_seed, max_iter=5000) if self.problem == CLS
                   else LinearSVR(C=C, epsilon=float(p["epsilon"]), random_state=self.est_seed, max_iter=5000))
            self.est = make_pipeline(nys, lin).fit(Z, d.y_train)
        elif kernel == "linear" and len(Z) > APPROX_LIMIT:
            self.notice = "Large data: the linear kernel uses a fast linear solver."
            self.est = (LinearSVC(C=C, class_weight=cw, random_state=self.est_seed, max_iter=5000) if self.problem == CLS
                        else LinearSVR(C=C, epsilon=float(p["epsilon"]), random_state=self.est_seed, max_iter=5000)).fit(Z, d.y_train)
        else:
            if self.problem == CLS:
                self.est = SVC(kernel=kernel, C=C, gamma=gamma, degree=deg, coef0=1.0, class_weight=cw,
                               random_state=self.est_seed, cache_size=500).fit(Z, d.y_train)
            else:
                self.est = SVR(kernel=kernel, C=C, gamma=gamma, degree=deg, coef0=1.0,
                               epsilon=float(p["epsilon"]), cache_size=500).fit(Z, d.y_train)
        self.n_stages = 1

    def _t(self, X):
        return self.scaler.transform(X) if self.scaler is not None else X

    def decision(self, X):
        return self.est.decision_function(self._t(X))

    def predict(self, X):
        if self.problem == CLS:
            return sigmoid(self.decision(X))
        return self.est.predict(self._t(X))

    def support_points(self) -> np.ndarray | None:
        if self.approx or not hasattr(self.est, "support_"):
            return None
        return self.data.X_train[self.est.support_]

    def facts(self) -> dict:
        sv = self.support_points()
        out = {"Kernel": self.params["kernel"]}
        if sv is not None:
            out["Support vectors"] = int(len(sv))
        elif self.approx:
            out["Approximation"] = "Nystroem, 300 components"
        return out

    def extra(self) -> dict:
        return {"support": self.support_points(), "linear_margin": self.params["kernel"] == "linear" and self.problem == CLS}


class SvmSpec(ModelSpec):
    id = "svm"
    display_name = "SVM"
    supports = {CLS, REG}
    interpretability = "Low"
    concept = ("Finds the boundary with the widest margin, and a kernel lets it bend. "
               "It shines on small to medium data with smooth non-linear boundaries. "
               "It struggles on very large data, needs scaled inputs, and its C and gamma settings matter a lot.")
    run_class = SvmRun

    def _schema(self, problem):
        return [
            ParamSpec("kernel", "Kernel", "choice", "rbf", "The shape of boundary it can learn. Linear is straight, RBF is smooth and flexible.",
                      choices=choice(["linear", "rbf", "poly"], ["Linear", "RBF", "Polynomial"])),
            ParamSpec("C", "C (penalty)", "log", 1.0, "Cost of errors. Large C fits the training data tightly, small C allows a wider margin.", lo=0.01, hi=1000.0),
            ParamSpec("gamma", "Gamma", "log", 0.5, "How local the RBF or polynomial kernel is. Large gamma gives wiggly, local boundaries.",
                      lo=0.01, hi=100.0, show_if=lambda p: p["kernel"] != "linear"),
            ParamSpec("degree", "Degree", "int", 3, "Degree of the polynomial kernel.", lo=2, hi=5, show_if=lambda p: p["kernel"] == "poly"),
            ParamSpec("epsilon", "Epsilon", "log", 0.1, "Width of the zone around the fit where errors are ignored.", lo=0.001, hi=1.0, problems=(REG,)),
            standardize_param(),
            class_weight_param(),
            threshold_param(),
        ]

    def summary(self, p, problem):
        s = f"kernel = {p['kernel']}, C = {p['C']:.3g}"
        if p["kernel"] != "linear":
            s += f", gamma = {p['gamma']:.3g}"
        return s
