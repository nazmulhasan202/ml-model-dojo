"""Gradient boosting with XGBoost.

Animation: the booster is fitted once with an evaluation set. Stage k predicts
with ``iteration_range=(0, k)``, so Train reveals one tree at a time. Loss
curves come from XGBoost's own evaluation log.

Note: choosing the number of trees by the minimum test loss leaks information
from the test set. A real project would use a separate validation set.
"""
from __future__ import annotations

import numpy as np
from xgboost import XGBClassifier, XGBRegressor

from .base import CLS, REG, ModelSpec, ParamSpec, Run, class_weight_param, threshold_param


class BoostRun(Run):
    replayable = True

    def _fit(self) -> None:
        p, d = self.params, self.data
        T = int(p["n_estimators"])
        kw = dict(n_estimators=T, learning_rate=float(p["learning_rate"]), max_depth=int(p["max_depth"]),
                  subsample=float(p["subsample"]), colsample_bytree=float(p["colsample"]),
                  reg_lambda=float(p["reg_lambda"]), gamma=float(p["gamma"]),
                  min_child_weight=float(p["min_child_weight"]), random_state=self.est_seed,
                  n_jobs=1, tree_method="hist", verbosity=0)
        if self.problem == CLS:
            n1 = max(int((d.y_train == 1).sum()), 1)
            n0 = max(int((d.y_train == 0).sum()), 1)
            if p["class_weight"] == "balanced":
                kw["scale_pos_weight"] = n0 / n1
            self.est = XGBClassifier(eval_metric="logloss", **kw)
        else:
            self.est = XGBRegressor(eval_metric="rmse", **kw)
        self.est.fit(d.X_train, d.y_train, eval_set=[(d.X_train, d.y_train), (d.X_test, d.y_test)], verbose=False)
        self.n_stages = T
        self._leaves = self._leaf_counts()
        self._compute_curves()

    def _leaf_counts(self) -> np.ndarray:
        dump = self.est.get_booster().get_dump()
        return np.array([max(t.count("leaf="), 1) for t in dump])

    def _compute_curves(self) -> None:
        ev = self.est.evals_result()
        key = "logloss" if self.problem == CLS else "rmse"
        tr = np.asarray(ev["validation_0"][key], float)
        te = np.asarray(ev["validation_1"][key], float)
        if self.problem == REG:
            tr, te = tr ** 2, te ** 2
        self.curves = {"train_loss": tr.tolist(), "test_loss": te.tolist()}
        # Scores are computed on a stride of stages, then interpolated.
        d = self.data
        T = self.n_stages
        ks = np.unique(np.r_[np.arange(1, T + 1, max(1, T // 40)), T])
        for name, X, y in (("train", d.X_train, d.y_train), ("test", d.X_test, d.y_test)):
            vals = []
            for k in ks:
                p = self._predict_k(X, int(k))
                vals.append(float(np.mean((p >= self.threshold).astype(int) == y)) if self.problem == CLS
                            else 1 - float(np.sum((y - p) ** 2) / max(np.sum((y - y.mean()) ** 2), 1e-12)))
            self.curves[f"{name}_score"] = np.interp(np.arange(1, T + 1), ks, vals).tolist()
        self.curves_full = True
        self.best_iter = int(np.argmin(te)) + 1

    def _predict_k(self, X, k):
        if self.problem == CLS:
            return self.est.predict_proba(X, iteration_range=(0, k))[:, 1]
        return self.est.predict(X, iteration_range=(0, k))

    def predict(self, X):
        return self._predict_k(X, max(self.stage, 1))

    def _timed_predict(self, X):
        return self.est.predict(X)

    def facts(self) -> dict:
        k = max(self.stage, 1)
        return {"Trees": k, "Leaves (total)": int(self._leaves[:k].sum())}

    def extra(self) -> dict:
        return {"best_iter": self.best_iter}


class BoostSpec(ModelSpec):
    id = "boosting"
    display_name = "Gradient boosting"
    supports = {CLS, REG}
    interpretability = "Low"
    concept = ("Adds small trees one after another, each correcting the errors of the ones before it. "
               "It shines on tabular data with interactions and often gives the best accuracy. "
               "It struggles when noise or outliers are large and the model has too many trees or a high learning rate.")
    run_class = BoostRun
    curve_xlabel = "Trees"

    def iterative(self, problem):
        return True

    def _schema(self, problem):
        return [
            ParamSpec("n_estimators", "Number of trees", "int", 100,
                      "How many boosting rounds. Watch the test loss curve: the minimum shows where more trees stop helping. "
                      "Picking that point from the test loss leaks information, so a real project would use a validation set.",
                      lo=1, hi=300),
            ParamSpec("learning_rate", "Learning rate", "log", 0.2, "How much each tree contributes. Small values need more trees.", lo=0.01, hi=1.0),
            ParamSpec("max_depth", "Max depth", "int", 3, "Depth of each small tree.", lo=1, hi=10),
            ParamSpec("subsample", "Row subsample", "float", 1.0, "Fraction of rows used for each tree. Below 1 adds randomness.", lo=0.3, hi=1.0),
            ParamSpec("colsample", "Column subsample", "float", 1.0, "Fraction of inputs used for each tree.", lo=0.3, hi=1.0),
            ParamSpec("reg_lambda", "L2 regularization", "log", 1.0, "Penalty on leaf values. Larger values make the model more cautious.", lo=0.01, hi=100.0),
            ParamSpec("gamma", "Min split gain", "float", 0.0, "A split must reduce the loss by at least this much.", lo=0.0, hi=5.0),
            ParamSpec("min_child_weight", "Min child weight", "float", 1.0, "Minimum total weight in a leaf. Larger values prevent tiny leaves.", lo=0.0, hi=20.0, decimals=1),
            class_weight_param("Implemented with scale_pos_weight."),
            threshold_param(),
        ]

    def summary(self, p, problem):
        return f"trees = {p['n_estimators']}, lr = {p['learning_rate']:.2g}, depth = {p['max_depth']}"
