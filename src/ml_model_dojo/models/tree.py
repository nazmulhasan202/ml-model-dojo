"""Decision tree.

Animation: the full tree is fitted once to find its leaf count L. Stage s
refits a tree with ``max_leaf_nodes`` equal to the s-th target, which grows the
tree best-first (the most useful split comes first). The last stage is the tree
fitted with the user settings, so the final result is exact.
"""
from __future__ import annotations

import numpy as np
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from .base import CLS, REG, ModelSpec, ParamSpec, Run, choice, class_weight_param, threshold_param

MAX_STAGES = 60


def tree_kwargs(problem: str, p: dict, seed: int) -> dict:
    kw = dict(criterion=p["criterion"], max_depth=int(p["max_depth"]),
              min_samples_split=int(p["min_samples_split"]), min_samples_leaf=int(p["min_samples_leaf"]),
              random_state=seed)
    if int(p["max_leaf_nodes"]) >= 2:
        kw["max_leaf_nodes"] = int(p["max_leaf_nodes"])
    if problem == CLS and p.get("class_weight") == "balanced":
        kw["class_weight"] = "balanced"
    return kw


class TreeRun(Run):
    replayable = True

    def _make(self, **over):
        kw = tree_kwargs(self.problem, self.params, self.est_seed)
        kw.update(over)
        return (DecisionTreeClassifier if self.problem == CLS else DecisionTreeRegressor)(**kw)

    def _fit(self) -> None:
        d = self.data
        self.full = self._make().fit(d.X_train, d.y_train)
        L = int(self.full.get_n_leaves())
        targets = np.unique(np.round(np.linspace(1, L, min(L, MAX_STAGES))).astype(int))
        self.targets = targets.tolist()
        self.n_stages = len(self.targets)
        self.prior = float(np.mean(d.y_train))
        self.est = None

    def _advance(self) -> None:
        self._set_stage_tree()

    def _set_stage_tree(self) -> None:
        if self.stage <= 0:
            self.est = None
            return
        leaves = self.targets[self.stage - 1]
        if leaves >= self.targets[-1]:
            self.est = self.full
        elif leaves <= 1:
            self.est = None
        else:
            cap = int(self.params["max_leaf_nodes"])
            self.est = self._make(max_leaf_nodes=min(leaves, cap) if cap >= 2 else leaves).fit(
                self.data.X_train, self.data.y_train)

    def _reset(self) -> None:
        self.est = None

    def curve_x(self, n):
        return np.array(self.targets[:n])

    def set_stage(self, k: int) -> None:
        """Random access, used when replaying (Back or jump to the end)."""
        self.stage = int(np.clip(k, 0, self.n_stages))
        self._set_stage_tree()

    def predict(self, X):
        if self.est is None:
            return np.full(len(X), self.prior)
        if self.problem == CLS:
            pr = self.est.predict_proba(X)
            return pr[:, 1] if pr.shape[1] == 2 else np.full(len(X), float(self.est.classes_[0]))
        return self.est.predict(X)

    def facts(self) -> dict:
        e = self.est
        if e is None:
            return {"Leaves": 1, "Depth": 0, "Nodes": 1}
        return {"Leaves": int(e.get_n_leaves()), "Depth": int(e.get_depth()), "Nodes": int(e.tree_.node_count)}

    def extra(self) -> dict:
        names = ["x1", "x2"] if self.data.n_inputs == 2 else ["x"]
        if self.est is None:
            return {"tree": None, "n": len(self.data.y_train), "prior": self.prior, "names": names}
        t = self.est.tree_
        val = t.value.reshape(t.node_count, -1)
        if self.problem == CLS:
            val = val / np.maximum(val.sum(1, keepdims=True), 1e-12)
        return {"tree": dict(left=t.children_left.copy(), right=t.children_right.copy(),
                             feature=t.feature.copy(), threshold=t.threshold.copy(),
                             n=t.n_node_samples.copy(), value=val), "names": names}


class TreeSpec(ModelSpec):
    id = "tree"
    display_name = "Decision tree"
    supports = {CLS, REG}
    interpretability = "High"
    concept = ("Splits the data with yes/no questions on one input at a time, building boxes with a constant prediction. "
               "It shines on threshold rules and mixed interactions and is easy to read. "
               "It struggles with diagonal or smooth boundaries, extrapolation, and overfits when deep.")
    run_class = TreeRun
    curve_xlabel = "Leaves"

    def iterative(self, problem):
        return True

    def _schema(self, problem):
        return [
            ParamSpec("criterion", "Split criterion", "choice", "gini" if problem == CLS else "squared_error",
                      "The rule used to pick the best split. Gini and entropy measure class impurity.",
                      choices=choice(["gini", "entropy"], ["Gini", "Entropy"]) if problem == CLS
                      else choice(["squared_error"], ["Squared error"])),
            ParamSpec("max_depth", "Max depth", "int", 4, "Longest path from root to leaf. Deeper trees fit more detail and overfit more.", lo=1, hi=20),
            ParamSpec("min_samples_split", "Min samples to split", "int", 2, "A node needs at least this many points to be split.", lo=2, hi=50),
            ParamSpec("min_samples_leaf", "Min samples per leaf", "int", 1, "Every leaf keeps at least this many points. Larger values smooth the model.", lo=1, hi=50),
            ParamSpec("max_leaf_nodes", "Max leaf nodes", "int", 0, "Caps the number of regions. The tree grows its best splits first.", lo=0, hi=100, zero_label="Unlimited"),
            class_weight_param(),
            threshold_param(),
        ]

    def summary(self, p, problem):
        return f"depth = {p['max_depth']}, min leaf = {p['min_samples_leaf']}"
