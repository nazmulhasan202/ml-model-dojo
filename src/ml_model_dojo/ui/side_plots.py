"""Small plots used in the metrics and details panels."""
from __future__ import annotations

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtWidgets import QSizePolicy

from ..models.base import CLS, make_grid
from .theme import BLUE, CLS_CMAP, INK, ORANGE, REG_CMAP


class SmallCanvas(FigureCanvasQTAgg):
    def __init__(self, parent=None, size=(3, 2.2), min_h=150):
        self.fig = Figure(figsize=size, dpi=100)
        super().__init__(self.fig)
        self.setParent(parent)
        self.ax = self.fig.add_subplot(111)
        self.setMinimumHeight(min_h)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)

    def export_png(self, path: str):
        self.fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")


class CurvePlot(SmallCanvas):
    """Train and test curves versus training stage, with a marker at the current stage."""

    def __init__(self, parent=None):
        super().__init__(parent, (3.2, 2.2), 170)
        self.mode = "loss"

    def update_run(self, run, mode: str | None = None):
        self.mode = mode or self.mode
        ax = self.ax
        ax.clear()
        self.fig.subplots_adjust(left=0.17, right=0.96, top=0.9, bottom=0.24)
        if run is None or not run.curves or run.stage == 0:
            ax.text(0.5, 0.5, "No curve yet", ha="center", va="center", transform=ax.transAxes, color="#666")
            ax.set_xticks([]); ax.set_yticks([])
            self.draw_idle()
            return
        tr = np.asarray(run.curves[f"train_{self.mode}"], float)
        te = np.asarray(run.curves[f"test_{self.mode}"], float)
        x = run.curve_x(len(tr))
        ax.plot(x, tr, color=BLUE, lw=1.6, label="Train")
        ax.plot(x, te, color=ORANGE, lw=1.6, label="Test")
        if run.curves_full and run.stage >= 1:
            ax.axvline(x[min(run.stage, len(x)) - 1], color="#555", lw=1, ls="--")
        best = run.extra().get("best_iter") if run.spec.id == "boosting" and self.mode == "loss" else None
        if best:
            ax.plot([x[best - 1]], [te[best - 1]], marker="v", color="#222", ms=7, ls="", label=f"Min test loss ({best})")
        reg = run.problem != CLS
        ylab = ("Loss (MSE)" if reg else "Loss (log loss)") if self.mode == "loss" else ("R squared" if reg else "Accuracy")
        ax.set_ylabel(ylab)
        ax.set_xlabel(run.spec.curve_xlabel)
        if self.mode == "score" and not reg:
            ax.set_ylim(min(0.45, float(min(tr.min(), te.min())) - 0.02), 1.02)
        elif self.mode == "loss":
            hi = float(np.percentile(np.r_[tr, te], 98)) * 1.2
            ax.set_ylim(0, max(hi, 1e-6))
        ax.legend(loc="best", frameon=False)
        self.draw_idle()


class ConfusionPlot(SmallCanvas):
    def __init__(self, parent=None):
        super().__init__(parent, (2.4, 2.2), 150)

    def update_cm(self, cm, title="Test confusion matrix"):
        ax = self.ax
        ax.clear()
        self.fig.subplots_adjust(left=0.25, right=0.95, top=0.85, bottom=0.22)
        if cm is None:
            ax.text(0.5, 0.5, "No predictions yet", ha="center", va="center", transform=ax.transAxes, color="#666")
            ax.set_xticks([]); ax.set_yticks([])
            self.draw_idle()
            return
        cm = np.asarray(cm)
        ax.imshow(cm, cmap="Blues", vmin=0, vmax=max(cm.max(), 1))
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=11,
                        color="white" if cm[i, j] > cm.max() * 0.55 else INK)
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(["pred 0", "pred 1"]); ax.set_yticklabels(["true 0", "true 1"])
        ax.set_title(title)
        self.draw_idle()


class PredActualPlot(SmallCanvas):
    def __init__(self, parent=None):
        super().__init__(parent, (2.6, 2.2), 150)

    def update_run(self, data, ev):
        ax = self.ax
        ax.clear()
        self.fig.subplots_adjust(left=0.2, right=0.95, top=0.9, bottom=0.22)
        if ev is None:
            ax.text(0.5, 0.5, "No predictions yet", ha="center", va="center", transform=ax.transAxes, color="#666")
            ax.set_xticks([]); ax.set_yticks([])
            self.draw_idle()
            return
        ax.scatter(data.y_train, ev["train"]["pred"], s=8, color=BLUE, alpha=0.4, label="Train")
        ax.scatter(data.y_test, ev["test"]["pred"], s=12, facecolors="none", edgecolors=ORANGE, label="Test")
        lo, hi = float(np.min(data.y)), float(np.max(data.y))
        ax.plot([lo, hi], [lo, hi], color="#555", lw=1)
        ax.set_xlabel("Actual"); ax.set_ylabel("Predicted"); ax.set_title("Predicted versus actual")
        ax.legend(frameon=False)
        self.draw_idle()


def _node_text(t, i, problem):
    n = int(t["n"][i])
    v = t["value"][i]
    stat = f"p1 = {v[1]:.2f}" if problem == CLS and len(v) > 1 else f"mean = {v[0]:.2f}"
    if t["left"][i] == -1:
        return f"n = {n}\n{stat}"
    name = t["names"][t["feature"][i]]
    return f"{name} <= {t['threshold'][i]:.2f}\nn = {n}\n{stat}"


class TreeDiagram(SmallCanvas):
    MAX_DEPTH = 4

    def __init__(self, parent=None):
        super().__init__(parent, (6, 3.4), 260)

    def update_run(self, run):
        ax = self.ax
        ax.clear()
        ax.axis("off")
        self.fig.subplots_adjust(left=0.01, right=0.99, top=0.98, bottom=0.02)
        ex = run.extra() if run is not None and run.stage > 0 else None
        if ex is None:
            ax.text(0.5, 0.5, "Train the tree to see its diagram", ha="center", va="center", transform=ax.transAxes, color="#666")
            self.draw_idle()
            return
        t = ex["tree"]
        if t is None:
            ax.text(0.5, 0.5, f"Root only: n = {ex['n']}, value = {ex['prior']:.2f}", ha="center", va="center", transform=ax.transAxes)
            self.draw_idle()
            return
        t = dict(t, names=ex["names"])
        pos, truncated = {}, False
        counter = [0]

        def layout(i, depth):
            nonlocal truncated
            if t["left"][i] == -1 or depth >= self.MAX_DEPTH:
                if t["left"][i] != -1:
                    truncated = True
                pos[i] = (counter[0], -depth)
                counter[0] += 1
                return pos[i][0]
            a, b = layout(t["left"][i], depth + 1), layout(t["right"][i], depth + 1)
            pos[i] = ((a + b) / 2, -depth)
            return pos[i][0]

        layout(0, 0)
        for i, (x, y) in pos.items():
            if t["left"][i] != -1 and (i == 0 or True):
                for c in (t["left"][i], t["right"][i]):
                    if c in pos:
                        ax.plot([x, pos[c][0]], [y, pos[c][1]], color="#888", lw=0.8, zorder=1)
        for i, (x, y) in pos.items():
            v = t["value"][i]
            if run.problem == CLS and len(v) > 1:
                fc = CLS_CMAP(float(v[1]))
            else:
                fc = "#f2f2f2"
            cut = t["left"][i] != -1 and not all(c in pos for c in (t["left"][i], t["right"][i]))
            txt = _node_text(t, i, run.problem) if not cut else _node_text(t, i, run.problem) + "\n..."
            ax.text(x, y, txt, ha="center", va="center", fontsize=6, zorder=2,
                    bbox=dict(boxstyle="round,pad=0.25", fc=fc, ec="#555", lw=0.6))
        ax.set_xlim(-0.8, max(counter[0] - 0.2, 1))
        ax.set_ylim(-self.MAX_DEPTH - 0.6, 0.6)
        if truncated:
            ax.text(0.01, 0.01, f"Display capped at depth {self.MAX_DEPTH}", transform=ax.transAxes, fontsize=7, color="#a00")
        self.setMinimumWidth(max(380, counter[0] * 62))
        self.draw_idle()


class NetworkDiagram(SmallCanvas):
    def __init__(self, parent=None):
        super().__init__(parent, (4.5, 2.4), 200)

    def update_run(self, features_labels, layers, coefs=None):
        ax = self.ax
        ax.clear()
        ax.axis("off")
        self.fig.subplots_adjust(left=0.02, right=0.98, top=0.95, bottom=0.03)
        sizes = [len(features_labels)] + [int(n) for n in layers] + [1]
        xs = np.linspace(0, 1, len(sizes))
        pos = [[(xs[l], 0.5 + (j - (n - 1) / 2) / max(max(sizes), 1) * 0.95) for j in range(n)] for l, n in enumerate(sizes)]
        for l in range(len(sizes) - 1):
            W = coefs[l] if coefs is not None and l < len(coefs) and coefs[l].shape == (sizes[l], sizes[l + 1]) else None
            wmax = np.abs(W).max() if W is not None else 1
            for a, pa in enumerate(pos[l]):
                for b, pb in enumerate(pos[l + 1]):
                    if W is None:
                        ax.plot([pa[0], pb[0]], [pa[1], pb[1]], color="#bbb", lw=0.5, zorder=1)
                    else:
                        w = W[a, b]
                        ax.plot([pa[0], pb[0]], [pa[1], pb[1]], color=ORANGE if w > 0 else BLUE,
                                lw=0.3 + 2.2 * abs(w) / wmax, alpha=0.8, zorder=1)
        for l, pts in enumerate(pos):
            for (x, y) in pts:
                ax.scatter([x], [y], s=110 if l else 90, c="white", edgecolors=INK, linewidths=1, zorder=3)
        for (x, y), lab in zip(pos[0], features_labels):
            ax.text(x - 0.02, y, lab, ha="right", va="center", fontsize=7)
        ax.text(xs[-1] + 0.02, 0.5, "output", ha="left", va="center", fontsize=7)
        ax.set_xlim(-0.25, 1.15)
        ax.set_ylim(-0.02, 1.02)
        self.draw_idle()


class SmallMultiples(SmallCanvas):
    """First six individual tree boundaries of a random forest."""

    def __init__(self, parent=None):
        self.fig = Figure(figsize=(4.5, 3.2), dpi=100)
        FigureCanvasQTAgg.__init__(self, self.fig)
        self.setParent(parent)
        self.setMinimumHeight(240)
        self.axes = self.fig.subplots(2, 3)

    def update_run(self, run, data):
        for a in self.axes.ravel():
            a.clear()
            a.set_xticks([]); a.set_yticks([])
        if run is None or run.stage == 0:
            self.draw_idle()
            return
        res = 40 if data.n_inputs == 2 else 150
        g, G = make_grid(data.n_inputs, res)
        for i, a in enumerate(self.axes.ravel()):
            if i >= len(run.est.estimators_):
                break
            p = run.tree_predict(i, G)
            if data.n_inputs == 1:
                a.plot(g, p, color=BLUE, lw=1)
                a.scatter(data.X_train[:, 0], data.y_train, s=2, color="#999")
            else:
                a.imshow(p.reshape(res, res), extent=(-1, 1, -1, 1), origin="lower",
                         cmap=CLS_CMAP if data.problem == CLS else REG_CMAP, vmin=None if data.problem != CLS else 0,
                         vmax=None if data.problem != CLS else 1)
            a.set_title(f"Tree {i + 1}", fontsize=7)
        self.fig.tight_layout(pad=0.5)
        self.draw_idle()
