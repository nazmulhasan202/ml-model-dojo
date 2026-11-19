"""Linear and logistic regression.

Animation: classification uses ``SGDClassifier`` (log loss) with one
``partial_fit`` epoch per stage, so the boundary moves while training.
Regression models (OLS, ridge, lasso) have a closed form and fit in one stage.
Polynomial features and the scaler are fitted on the training set only.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import Lasso, LinearRegression, Ridge, SGDClassifier
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from .base import (CLS, REG, ModelSpec, ParamSpec, Run, balanced_weights, choice, class_weight_param,
                   sigmoid, standardize_param, threshold_param)


class LinearRun(Run):
    def _fit(self) -> None:
        p, d = self.params, self.data
        self.names_in = ["x1", "x2"] if d.n_inputs == 2 else ["x"]
        self.poly = PolynomialFeatures(int(p["degree"]), include_bias=False).fit(d.X_train)
        Z = self.poly.transform(d.X_train)
        self.scaler = StandardScaler().fit(Z) if p["standardize"] else None
        Z = self._scale(Z)
        pen = p["penalty"]
        if self.problem == CLS:
            self.n_stages = int(p["max_epochs"])
            cw = balanced_weights(d.y_train) if p["class_weight"] == "balanced" else None
            self.est = SGDClassifier(loss="log_loss", penalty=None if pen == "none" else pen,
                                     alpha=float(p["alpha"]), learning_rate="constant",
                                     eta0=float(p["lr"]), random_state=self.est_seed, class_weight=cw,
                                     shuffle=True)
            self.Z = Z
            self.est.partial_fit(Z, d.y_train, classes=np.array([0, 1]))
            # Start from zero weights so that stage 0 is a neutral model.
            self.est.coef_[:] = 0.0
            self.est.intercept_[:] = 0.0
        else:
            self.n_stages = 1
            a = float(p["alpha"])
            self.est = {"none": LinearRegression(), "l2": Ridge(alpha=a),
                        "l1": Lasso(alpha=a, max_iter=20000)}[pen]
            self.est.fit(Z, d.y_train)

    def _scale(self, Z):
        return self.scaler.transform(Z) if self.scaler is not None else Z

    def _advance(self) -> None:
        if self.problem == CLS:
            self.est.partial_fit(self.Z, self.data.y_train)

    def predict(self, X):
        Z = self._scale(self.poly.transform(X))
        if self.problem == CLS:
            return sigmoid(self.est.decision_function(Z))
        return self.est.predict(Z)

    def _weights(self):
        return np.ravel(self.est.coef_), float(np.ravel(self.est.intercept_)[0])

    def facts(self) -> dict:
        w, _ = self._weights()
        return {"Parameters": int(len(w) + 1), "Polynomial degree": int(self.params["degree"])}

    def extra(self) -> dict:
        w, b = self._weights()
        names = list(self.poly.get_feature_names_out(self.names_in))
        eq = ""
        degree1 = int(self.params["degree"]) == 1
        if degree1 and self.scaler is not None:       # convert back to original units for display
            s, m = self.scaler.scale_, self.scaler.mean_
            b = b - float(np.sum(w * m / s))
            w = w / s
        if degree1:
            terms = " ".join(f"{'+' if c >= 0 else '-'} {abs(c):.3f} {n}" for c, n in zip(w, names))
            lhs = "logit(p)" if self.problem == CLS else "y"
            eq = f"{lhs} = {b:.3f} {terms}"
        rows = [("intercept", b)] + list(zip(names, w))
        note = "Coefficients are in standardized units." if (not degree1 and self.scaler is not None) else ""
        return {"coef_rows": rows, "equation": eq, "coef_note": note}


class LinearSpec(ModelSpec):
    id = "linear"
    display_name = "Linear / logistic"
    supports = {CLS, REG}
    interpretability = "High"
    concept = ("Fits a weighted sum of the inputs (logistic regression passes it through a sigmoid). "
               "It shines when relationships are close to linear, data are scarce, or you need readable coefficients. "
               "It struggles with curved boundaries and interactions unless you add polynomial features.")
    run_class = LinearRun
    curve_xlabel = "Epoch"

    def iterative(self, problem):
        return problem == CLS

    def _schema(self, problem):
        return [
            ParamSpec("penalty", "Regularization", "choice", "none" if problem == REG else "l2",
                      "Penalty on large coefficients. L2 shrinks them, L1 can set some exactly to zero.",
                      choices=choice(["none", "l1", "l2"], ["None", "L1 (lasso)", "L2 (ridge)"])),
            ParamSpec("alpha", "Strength (alpha)", "log", 1e-3 if problem == CLS else 1e-2,
                      "How strongly large coefficients are penalized. Larger values give a simpler model.",
                      lo=1e-5, hi=10.0, show_if=lambda p: p["penalty"] != "none"),
            ParamSpec("degree", "Polynomial degree", "int", 1,
                      "Adds powers and products of the inputs. Degree 2 includes x1 times x2.", lo=1, hi=5),
            standardize_param(),
            class_weight_param(),
            ParamSpec("lr", "Learning rate", "log", 0.05, "Step size of each gradient update.", lo=1e-3, hi=1.0,
                      problems=(CLS,)),
            ParamSpec("max_epochs", "Max epochs", "int", 150, "Number of passes over the training data.",
                      lo=5, hi=500, problems=(CLS,)),
            threshold_param(),
        ]

    def summary(self, p, problem):
        s = f"degree = {p['degree']}, penalty = {p['penalty']}"
        if p["penalty"] != "none":
            s += f", alpha = {p['alpha']:.3g}"
        return s
