"""Random forest.

Animation: the whole forest is fitted once. Stage k averages the predictions of
the first k trees, so the ensemble boundary smooths out as trees are added.
Per-tree predictions are cached for each set of points that is displayed.
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from .base import CLS, REG, ModelSpec, ParamSpec, Run, choice, class_weight_param, threshold_param

_MAXF = {"sqrt": "sqrt", "log2": "log2", "all": None, "half": 0.5}


class ForestRun(Run):
    replayable = True

    def _fit(self) -> None:
        p, d = self.params, self.data
        T = int(p["n_estimators"])
        kw = dict(n_estimators=T, max_depth=int(p["max_depth"]), min_samples_leaf=int(p["min_samples_leaf"]),
                  max_features=_MAXF[p["max_features"]], bootstrap=bool(p["bootstrap"]),
                  oob_score=bool(p["bootstrap"]) and len(d.y_train) > 10, random_state=self.est_seed, n_jobs=1)
        if self.problem == CLS:
            kw["class_weight"] = "balanced" if p["class_weight"] == "balanced" else None
            self.est = RandomForestClassifier(**kw)
        else:
            self.est = RandomForestRegressor(**kw)
        self.est.fit(d.X_train, d.y_train)
        self.n_stages = T
        self._cache: dict[int, np.ndarray] = {}
        self._compute_curves()

    # per-tree values for a set of points: class 1 probability or regression value
    def tree_values(self, X: np.ndarray) -> np.ndarray:
        key = hash((X.shape, X.tobytes()))
        if key in self._cache:
            return self._cache[key]
        out = np.empty((len(self.est.estimators_), len(X)), np.float32)
        for i, t in enumerate(self.est.estimators_):
            out[i] = self._tree_pred(t, X)
        if len(self._cache) >= 5:
            self._cache.pop(next(iter(self._cache)))
        self._cache[key] = out
        return out

    def _tree_pred(self, t, X):
        if self.problem == CLS:
            pr = t.predict_proba(X)
            return pr[:, 1] if pr.shape[1] == 2 else np.full(len(X), float(self.est.classes_[0]))
        return t.predict(X)

    def tree_predict(self, i: int, X: np.ndarray) -> np.ndarray:
        return self._tree_pred(self.est.estimators_[i], X)

    def _compute_curves(self) -> None:
        d = self.data
        T = self.n_stages
        ks = np.arange(1, T + 1)[:, None]
        self.curves = {}
        for name, X, y in (("train", d.X_train, d.y_train), ("test", d.X_test, d.y_test)):
            cum = np.cumsum(self.tree_values(X), axis=0, dtype=np.float64) / ks
            if self.problem == CLS:
                p = np.clip(cum, 1e-9, 1 - 1e-9)
                self.curves[f"{name}_score"] = (cum >= self.threshold).astype(int).__eq__(y[None, :]).mean(1).tolist()
                self.curves[f"{name}_loss"] = (-(y * np.log(p) + (1 - y) * np.log(1 - p))).mean(1).tolist()
            else:
                ss_tot = max(float(np.sum((y - y.mean()) ** 2)), 1e-12)
                sse = ((cum - y[None, :]) ** 2).sum(1)
                self.curves[f"{name}_score"] = (1 - sse / ss_tot).tolist()
                self.curves[f"{name}_loss"] = (sse / len(y)).tolist()
        self.curves_full = True

    def predict(self, X):
        k = max(self.stage, 1)
        return self.tree_values(X)[:k].mean(0, dtype=np.float64)

    def _timed_predict(self, X):
        return self.est.predict(X)

    def facts(self) -> dict:
        k = max(self.stage, 1)
        trees = self.est.estimators_[:k]
        out = {"Trees": k, "Leaves (total)": int(sum(t.get_n_leaves() for t in trees))}
        oob = getattr(self.est, "oob_score_", None)
        if oob is not None:
            out["Out-of-bag score"] = round(float(oob), 3)
        return out

    def extra(self) -> dict:
        return {"oob": getattr(self.est, "oob_score_", None)}


class ForestSpec(ModelSpec):
    id = "forest"
    display_name = "Random forest"
    supports = {CLS, REG}
    interpretability = "Low"
    concept = ("Averages many deep trees, each trained on a random resample of the rows and a random subset of inputs. "
               "It shines as a strong, low-tuning default that resists noise. "
               "It struggles to extrapolate beyond the training range and is harder to read than one tree.")
    run_class = ForestRun
    curve_xlabel = "Trees"

    def iterative(self, problem):
        return True

    def _schema(self, problem):
        return [
            ParamSpec("n_estimators", "Number of trees", "int", 100, "More trees make the average smoother and rarely hurt.", lo=1, hi=300),
            ParamSpec("max_depth", "Max depth", "int", 8, "Longest path in each tree.", lo=1, hi=20),
            ParamSpec("min_samples_leaf", "Min samples per leaf", "int", 1, "Larger leaves make each tree smoother.", lo=1, hi=50),
            ParamSpec("max_features", "Max features per split", "choice", "sqrt" if problem == CLS else "all",
                      "How many inputs each split may look at. Fewer inputs make trees more different from each other.",
                      choices=choice(["sqrt", "log2", "all", "half"], ["Square root", "Log2", "All", "Half"])),
            ParamSpec("bootstrap", "Bootstrap rows", "bool", True,
                      "Train each tree on a random resample of the rows. This also enables the out-of-bag score."),
            class_weight_param(),
            threshold_param(),
        ]

    def summary(self, p, problem):
        return f"trees = {p['n_estimators']}, depth = {p['max_depth']}, features = {p['max_features']}"
