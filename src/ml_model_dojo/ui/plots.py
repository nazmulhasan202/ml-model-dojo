"""Main plot: data, decision boundary or fitted function. Artists are reused between frames."""
from __future__ import annotations

import warnings

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.collections import LineCollection
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QSizePolicy

from ..data.datasets import EXTRAP_TEST, EXTRAP_TRAIN, Dataset
from ..models.base import CLS, make_grid
from .theme import BLUE, CLS_CMAP, INK, ORANGE, RED, REG_CMAP, TEST_REGION, TRAIN_REGION

CLS_COLORS = (BLUE, ORANGE)
CLS_MARKERS = ("o", "^")
COARSE, FINE = 60, 160
COARSE_1D, FINE_1D = 150, 400


def marker_size(n: int) -> float:
    return 36 if n <= 150 else 26 if n <= 600 else 14 if n <= 3000 else 6


def y_limits(ds: Dataset) -> tuple[float, float]:
    lo, hi = float(np.min(ds.y)), float(np.max(ds.y))
    pad = 0.15 * max(hi - lo, 1e-6)
    return lo - pad, hi + pad


def draw_data(ax, ds: Dataset, show_train=True, show_test=True, norm=None, size=None):
    """Draw data points on ``ax`` and return a dict of artists."""
    s = size or marker_size(len(ds.y))
    art: dict = {}
    if ds.problem == CLS:
        for split, X, y in (("train", ds.X_train, ds.y_train), ("test", ds.X_test, ds.y_test)):
            for k in (0, 1):
                P = X[y == k]
                if split == "train":
                    a = ax.scatter(P[:, 0], P[:, 1], s=s, marker=CLS_MARKERS[k], facecolors=CLS_COLORS[k],
                                   edgecolors=INK, linewidths=0.5, zorder=4)
                else:
                    a = ax.scatter(P[:, 0], P[:, 1], s=s * 1.2, marker=CLS_MARKERS[k], facecolors="none",
                                   edgecolors=CLS_COLORS[k], linewidths=1.4, zorder=5)
                art[(split, k)] = a
    elif ds.n_inputs == 1:
        art[("train", 0)] = ax.scatter(ds.X_train[:, 0], ds.y_train, s=s, facecolors="#4a4a4a", edgecolors="white",
                                       linewidths=0.4, zorder=4)
        art[("test", 0)] = ax.scatter(ds.X_test[:, 0], ds.y_test, s=s * 1.2, facecolors="none", edgecolors=ORANGE,
                                      linewidths=1.4, zorder=5)
    else:
        art[("train", 0)] = ax.scatter(ds.X_train[:, 0], ds.X_train[:, 1], s=s * 1.3, c=ds.y_train, cmap=REG_CMAP,
                                       norm=norm, edgecolors=INK, linewidths=0.6, zorder=4)
        cmap = art[("train", 0)].get_cmap()
        art[("test", 0)] = ax.scatter(ds.X_test[:, 0], ds.X_test[:, 1], s=s * 1.8, facecolors="none",
                                      edgecolors=cmap(norm(ds.y_test)), linewidths=2.0, zorder=5)
    for (split, _), a in art.items():
        a.set_visible(show_train if split == "train" else show_test)
    return art


def reg_norm(ds: Dataset) -> Normalize:
    lo, hi = np.percentile(ds.y, [1, 99])
    return Normalize(lo, hi if hi > lo else lo + 1)


class BoundaryPlot(FigureCanvasQTAgg):
    hovered = Signal(str)
    point_clicked = Signal(float, float)

    def __init__(self, parent=None):
        fig = Figure(figsize=(5, 5), dpi=100)
        super().__init__(fig)
        self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumSize(300, 300)
        self.ax = fig.add_subplot(111)
        self.dataset: Dataset | None = None
        self.run = None
        self.ev = None
        self.flags = {"train": True, "test": True, "errors": True, "truth": True}
        self.hover_hook = None
        self.mpl_connect("motion_notify_event", self._on_move)
        self.mpl_connect("button_press_event", self._on_click)
        self.setFocusPolicy(Qt.StrongFocus)
        self._reset_artists()

    # ---- setup -----------------------------------------------------------
    def _reset_artists(self):
        self.im = None
        self.cbar = None
        self.truth_cs = None
        self.line = None
        self.truth_line = None
        self.contours: list = []
        self.err_art = None
        self.resid = None
        self.overlay_pts = None
        self.neighbor_lines = None
        self.art: dict = {}
        self.legend = None
        self._grids: dict = {}

    def set_dataset(self, ds: Dataset | None):
        ax, fig = self.ax, self.figure
        if self.cbar is not None:
            self.cbar.remove()
        ax.clear()
        for t in list(fig.legends):
            t.remove()
        self._reset_artists()
        self.dataset, self.run, self.ev = ds, None, None
        if ds is None:
            self.draw_idle()
            return
        cls, d1 = ds.problem == CLS, ds.n_inputs == 1
        ax.set_xlim(-1, 1)
        if d1:
            ax.set_ylim(*y_limits(ds))
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_aspect("auto")
            if ds.extrapolation:
                ax.axvspan(-1, EXTRAP_TRAIN[1], color=TRAIN_REGION, zorder=0)
                ax.axvspan(EXTRAP_TEST[0], 1, color=TEST_REGION, zorder=0)
                ax.text(-0.98, 0.97, "training range", transform=ax.get_xaxis_transform(), va="top", fontsize=7)
                ax.text(0.62, 0.97, "test range", transform=ax.get_xaxis_transform(), va="top", fontsize=7)
        else:
            ax.set_ylim(-1, 1)
            ax.set_xlabel("x1")
            ax.set_ylabel("x2")
            ax.set_aspect("equal", adjustable="box")
        self.norm = None if cls or d1 else reg_norm(ds)
        self.art = draw_data(ax, ds, self.flags["train"], self.flags["test"], self.norm)
        if not cls and not d1:
            self._add_colorbar_and_truth(ds)
        self.err_art = ax.scatter([], [], s=marker_size(len(ds.y)) * 4, facecolors="none", edgecolors=RED,
                                  linewidths=1.6, zorder=7)
        self.resid = LineCollection([], colors=RED, linewidths=0.8, zorder=3)
        ax.add_collection(self.resid)
        if d1:
            self.truth_line, = ax.plot([], [], ls=":", color="#777", lw=1.2, zorder=2)
            g = np.linspace(-1, 1, 300)[:, None]
            self.truth_line.set_data(g[:, 0], ds.info.truth(g))
            self.truth_line.set_visible(self.flags["truth"])
            self.line, = ax.plot([], [], color=BLUE, lw=2.2, zorder=3)
        self._build_legend(ds)
        self.figure.subplots_adjust(left=0.1, right=0.97 if self.cbar is None else 0.9, top=0.97, bottom=0.2)
        self.draw_idle()

    def _add_colorbar_and_truth(self, ds: Dataset):
        """Two-input regression: a color scale for y and dashed contours of the noise-free surface."""
        sm = ScalarMappable(norm=self.norm, cmap=REG_CMAP)
        self.cbar = self.figure.colorbar(sm, ax=self.ax, fraction=0.046, pad=0.03)
        self.cbar.set_label("y (true or predicted)", fontsize=7)
        self.cbar.ax.tick_params(labelsize=7)
        g, G = make_grid(2, 60)
        Zt = np.asarray(ds.info.truth(G)).reshape(60, 60)
        levels = np.linspace(self.norm.vmin, self.norm.vmax, 9)[1:-1]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.truth_cs = self.ax.contour(g, g, Zt, levels=levels, colors="#222", linestyles="dashed",
                                            linewidths=0.9, alpha=0.75, zorder=3)
        self.truth_cs.set_visible(self.flags["truth"])

    def _build_legend(self, ds: Dataset):
        h = []
        if ds.problem == CLS:
            for k in (0, 1):
                h.append(Line2D([], [], ls="", marker=CLS_MARKERS[k], mfc=CLS_COLORS[k], mec=INK, mew=0.5, ms=6,
                                label=f"Class {k} (train)"))
            for k in (0, 1):
                h.append(Line2D([], [], ls="", marker=CLS_MARKERS[k], mfc="none", mec=CLS_COLORS[k], mew=1.4, ms=6,
                                label=f"Class {k} (test)"))
            h.append(Line2D([], [], ls="", marker="o", mfc="none", mec=RED, mew=1.6, ms=9, label="Test error"))
        elif ds.n_inputs == 1:
            h = [Line2D([], [], ls="", marker="o", mfc="#4a4a4a", mec="white", ms=6, label="Train"),
                 Line2D([], [], ls="", marker="o", mfc="none", mec=ORANGE, mew=1.4, ms=6, label="Test"),
                 Line2D([], [], color=BLUE, lw=2, label="Model"),
                 Line2D([], [], color="#777", ls=":", label="True function")]
        else:
            h = [Line2D([], [], ls="", marker="o", mfc="#888", mec=INK, ms=6, label="Train (color = true y)"),
                 Line2D([], [], ls="", marker="o", mfc="none", mec="#888", mew=2, ms=7, label="Test (ring = true y)"),
                 Line2D([], [], color="#222", ls="--", lw=1, label="True surface (contours)"),
                 Line2D([], [], color="#9aa5b1", lw=1.5, label="Model surface (contours)")]
        self.legend = self.figure.legend(handles=h, loc="lower center", ncol=3, frameon=False, fontsize=7,
                                         handletextpad=0.3, columnspacing=1.0)

    # ---- toggles ---------------------------------------------------------
    def set_flag(self, name: str, value: bool):
        self.flags[name] = value
        for (split, _), a in self.art.items():
            if split in ("train", "test"):
                a.set_visible(self.flags[split])
        if self.truth_line is not None:
            self.truth_line.set_visible(self.flags["truth"])
        if self.truth_cs is not None:
            self.truth_cs.set_visible(self.flags["truth"])
        if self.err_art is not None:
            self.err_art.set_visible(self.flags["errors"])
            self.resid.set_visible(self.flags["errors"])
        self.draw_idle()

    # ---- prediction ------------------------------------------------------
    def _grid(self, res: int):
        if res not in self._grids:
            self._grids[res] = make_grid(self.dataset.n_inputs, res)
        return self._grids[res]

    def update_run(self, run, ev=None, fine: bool = True):
        """Redraw the model layer. ``run`` may be None or at stage 0 (data only)."""
        ds = self.dataset
        self.run, self.ev = run, ev
        if ds is None:
            return
        if run is None or run.stage == 0:
            self._clear_model()
            self.draw_idle()
            return
        d1 = ds.n_inputs == 1
        res = (FINE_1D if fine else COARSE_1D) if d1 else (FINE if fine else COARSE)
        g, G = self._grid(res)
        p = run.predict(G)
        if d1:
            self.line.set_data(g, p)
            self.line.set_visible(True)
        else:
            Z = p.reshape(res, res)
            if ds.problem == CLS:
                thr = float(np.clip(run.threshold, 0.02, 0.98))
                norm = TwoSlopeNorm(vcenter=thr, vmin=0.0, vmax=1.0)
                self._set_image(Z, CLS_CMAP, norm)
                self._set_contours(g, Z, [thr], "black", 1.8)
                self._svm_overlay(run, g, Z)
            else:
                self._set_image(Z, REG_CMAP, self.norm)
                self._set_contours(g, Z, np.linspace(self.norm.vmin, self.norm.vmax, 9)[1:-1], "white", 0.9)
        self._update_errors(run, ev)
        self.draw_idle()

    def _clear_model(self):
        if self.im is not None:
            self.im.set_visible(False)
        if self.line is not None:
            self.line.set_visible(False)
        self._remove_contours()
        self.err_art.set_offsets(np.empty((0, 2)))
        self.resid.set_segments([])
        self._clear_overlays()

    def _set_image(self, Z, cmap, norm):
        if self.im is None:
            self.im = self.ax.imshow(Z, extent=(-1, 1, -1, 1), origin="lower", cmap=cmap, norm=norm,
                                     interpolation="bilinear", zorder=1, aspect="equal")
            self.ax.set_xlim(-1, 1)
            self.ax.set_ylim(-1, 1)
        self.im.set_data(Z)
        self.im.set_cmap(cmap)
        self.im.set_norm(norm)
        self.im.set_visible(True)

    def _remove_contours(self):
        for c in self.contours:
            try:
                c.remove()
            except (ValueError, AttributeError, NotImplementedError):
                for coll in getattr(c, "collections", []):
                    coll.remove()
        self.contours = []

    def _set_contours(self, g, Z, levels, color, lw, styles="solid", keep=False):
        if not keep:
            self._remove_contours()
        levels = [l for l in levels if Z.min() < l < Z.max()]
        if levels:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                cs = self.ax.contour(g, g, Z, levels=levels, colors=color, linewidths=lw, linestyles=styles, zorder=2)
            self.contours.append(cs)

    def _clear_overlays(self):
        if self.overlay_pts is not None:
            self.overlay_pts.remove()
            self.overlay_pts = None
        self.clear_neighbors()

    def _svm_overlay(self, run, g, Z):
        if run.spec.id != "svm":
            self._clear_overlays()
            return
        ex = run.extra()
        if ex["linear_margin"]:
            s = 1 / (1 + np.exp(-1.0))
            self._set_contours(g, Z, [1 - s, s], "#555", 1.0, styles="dashed", keep=True)
        sv = ex["support"]
        if self.overlay_pts is not None:
            self.overlay_pts.remove()
            self.overlay_pts = None
        if sv is not None and len(sv):
            self.overlay_pts = self.ax.scatter(sv[:, 0], sv[:, 1], s=marker_size(len(self.dataset.y)) * 3.2,
                                               facecolors="none", edgecolors="#222", linewidths=0.8, zorder=6)

    def _update_errors(self, run, ev):
        ds = self.dataset
        if ev is None:
            self.err_art.set_offsets(np.empty((0, 2)))
            self.resid.set_segments([])
            return
        pred = ev["test"]["pred"]
        if ds.problem == CLS:
            bad = pred != ds.y_test
            self.err_art.set_offsets(ds.X_test[bad] if bad.any() else np.empty((0, 2)))
            self.resid.set_segments([])
        else:
            self.err_art.set_offsets(np.empty((0, 2)))
            if ds.n_inputs == 1:
                x = ds.X_test[:, 0]
                segs = [[(xi, yi), (xi, pi)] for xi, yi, pi in zip(x, ds.y_test, pred)]
                self.resid.set_segments(segs)
            else:
                self.resid.set_segments([])
        self.err_art.set_visible(self.flags["errors"])
        self.resid.set_visible(self.flags["errors"])

    # ---- neighbors (kNN) -------------------------------------------------
    def set_neighbors(self, point, nbrs):
        self.clear_neighbors()
        segs = [[tuple(point), tuple(n)] for n in nbrs]
        self.neighbor_lines = LineCollection(segs, colors="#333", linewidths=1.0, zorder=6)
        self.ax.add_collection(self.neighbor_lines)

    def clear_neighbors(self):
        if self.neighbor_lines is not None:
            self.neighbor_lines.remove()
            self.neighbor_lines = None

    # ---- interaction -----------------------------------------------------
    def _on_move(self, ev):
        ds, run = self.dataset, self.run
        if ds is None or ev.inaxes is not self.ax or ev.xdata is None:
            return
        x, y = float(ev.xdata), float(ev.ydata)
        d1 = ds.n_inputs == 1
        parts = [f"x = {x:.2f}" if d1 else f"x1 = {x:.2f}, x2 = {y:.2f}"]
        if run is not None and run.stage > 0:
            pt = np.array([[x]]) if d1 else np.array([[x, y]])
            v = float(run.predict(pt)[0])
            parts.append(f"P(class 1) = {v:.2f}" if ds.problem == CLS else f"prediction = {v:.2f}")
            if self.hover_hook is not None and not d1:
                self.hover_hook(pt[0])
        near = self._nearest(x, y)
        if near:
            parts.append(near)
        self.hovered.emit("   |   ".join(parts))

    def _nearest(self, x, y) -> str:
        ds = self.dataset
        X, yy = ds.X, ds.y
        d1 = ds.n_inputs == 1
        if d1:
            # scale distance by axis ranges so the hit area is similar in both directions
            lo, hi = y_limits(ds)
            dist = np.hypot((X[:, 0] - x) / 2, (yy - y) / (hi - lo))
        else:
            dist = np.hypot(X[:, 0] - x, X[:, 1] - y) / 2
        i = int(np.argmin(dist))
        if dist[i] > 0.02:
            return ""
        split = "test" if ds.is_test[i] else "train"
        pt = f"{split} point ({X[i, 0]:.2f}" + ("" if d1 else f", {X[i, 1]:.2f}") + ")"
        if ds.problem == CLS:
            txt = f"{pt}: true class {int(yy[i])}"
            if self.ev is not None:
                j = i - len(ds.y_train) if split == "test" else i
                pr = self.ev[split]["pred"][j]
                txt += f", predicted {int(pr)}"
            return txt
        txt = f"{pt}: true y = {yy[i]:.2f}"
        if self.ev is not None:
            j = i - len(ds.y_train) if split == "test" else i
            txt += f", predicted {self.ev[split]['pred'][j]:.2f}"
        return txt

    def _on_click(self, ev):
        if ev.inaxes is self.ax and ev.xdata is not None:
            self.point_clicked.emit(float(ev.xdata), float(ev.ydata))

    # ---- export ----------------------------------------------------------
    def export_png(self, path: str):
        """Paper-style export: box on, small text, white background."""
        self.figure.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
