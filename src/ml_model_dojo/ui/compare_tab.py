"""Compare tab: train several models in parallel on the same data and compare them."""
from __future__ import annotations

import csv
import logging
import math
import multiprocessing as mp
import os
import traceback
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.colors import TwoSlopeNorm
from matplotlib.figure import Figure
from PySide6.QtCore import QSortFilterProxyModel, Qt, QThread, Signal
from PySide6.QtGui import QStandardItem, QStandardItemModel
from PySide6.QtWidgets import (QAbstractItemView, QCheckBox, QComboBox, QFileDialog, QHBoxLayout, QHeaderView, QLabel,
                               QProgressBar, QPushButton, QScrollArea, QTableView, QTabWidget, QVBoxLayout, QWidget)

from .. import metrics as M
from ..data.datasets import Dataset
from ..models import MODEL_ORDER, get_spec
from ..models.base import CLS
from ..seeding import SEEDS
from ..workers.compare_runner import job_key, make_job, run_job, warm_up
from .dialogs import IMAGE_SUFFIXES, fix_suffix, safe_save
from .plots import draw_data, marker_size, reg_norm, y_limits
from .theme import BLUE, CLS_CMAP, MODEL_COLORS, ORANGE, RED, REG_CMAP

log = logging.getLogger("msp")
MAX_MODELS = 8


class CompareWorker(QThread):
    progress = Signal(int, int)
    result = Signal(str, object)
    job_failed = Signal(str, str)
    failed = Signal(str)

    def __init__(self, pool: ProcessPoolExecutor, pending: list[tuple[str, dict]]):
        super().__init__()
        self.pool, self.pending = pool, pending
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def run(self):
        try:
            futs = {self.pool.submit(run_job, job): key for key, job in self.pending}
            waiting, done, total = set(futs), 0, len(futs)
            while waiting and not self._cancel:
                finished, waiting = wait(waiting, timeout=0.2, return_when=FIRST_COMPLETED)
                for f in finished:
                    try:
                        self.result.emit(futs[f], f.result())
                    except Exception:
                        self.job_failed.emit(futs[f], traceback.format_exc())
                    done += 1
                    self.progress.emit(done, total)
        except Exception:
            self.failed.emit(traceback.format_exc())


# ---- drawing helpers -----------------------------------------------------
def draw_panel(ax, ds: Dataset, res: dict, title: str, xlabel: str):
    ax.set_xlim(-1, 1)
    d1 = ds.n_inputs == 1
    g, p = res["surface"]
    if d1:
        ax.set_ylim(*y_limits(ds))
        if ds.extrapolation:
            from ..data.datasets import EXTRAP_TEST, EXTRAP_TRAIN
            from .theme import TEST_REGION, TRAIN_REGION
            ax.axvspan(-1, EXTRAP_TRAIN[1], color=TRAIN_REGION, zorder=0)
            ax.axvspan(EXTRAP_TEST[0], 1, color=TEST_REGION, zorder=0)
        gg = np.linspace(-1, 1, 200)[:, None]
        ax.plot(gg[:, 0], ds.info.truth(gg), ls=":", color="#777", lw=1, zorder=2)
        ax.plot(g, p, color=BLUE, lw=2, zorder=3)
        norm = None
    else:
        ax.set_ylim(-1, 1)
        ax.set_aspect("equal", adjustable="box")
        n = len(g)
        Z = np.asarray(p).reshape(n, n)
        if ds.problem == CLS:
            thr = float(np.clip(res.get("threshold", 0.5), 0.02, 0.98))
            ax.imshow(Z, extent=(-1, 1, -1, 1), origin="lower", cmap=CLS_CMAP,
                      norm=TwoSlopeNorm(vcenter=thr, vmin=0, vmax=1), interpolation="bilinear", zorder=1)
            if Z.min() < thr < Z.max():
                ax.contour(g, g, Z, levels=[thr], colors="black", linewidths=1.2, zorder=2)
            norm = None
        else:
            norm = reg_norm(ds)
            ax.imshow(Z, extent=(-1, 1, -1, 1), origin="lower", cmap=REG_CMAP, norm=norm, interpolation="bilinear", zorder=1)
    draw_data(ax, ds, norm=norm, size=max(5, marker_size(len(ds.y)) * 0.55))
    if ds.problem == CLS:
        bad = res["pred_test"] != ds.y_test
        if bad.any():
            ax.scatter(ds.X_test[bad, 0], ds.X_test[bad, 1], s=marker_size(len(ds.y)) * 2.5, facecolors="none",
                       edgecolors=RED, linewidths=1.0, zorder=7)
    ax.set_title(title, fontsize=8)
    ax.set_xlabel(xlabel, fontsize=7)
    ax.tick_params(labelsize=6)


def size_text(model_id: str, facts: dict) -> tuple[str, float]:
    """A readable model size and a number to sort by."""
    f = facts
    if model_id in ("linear", "mlp"):
        n = f.get("Parameters", 0)
        return f"{n} parameters", n
    if model_id == "knn":
        n = f.get("Stored training points", 0)
        return f"{n} stored points", n
    if model_id == "tree":
        n = f.get("Leaves", 0)
        return f"{n} leaves", n
    if model_id in ("forest", "boosting"):
        return f"{f.get('Trees', 0)} trees, {f.get('Leaves (total)', 0)} leaves", f.get("Leaves (total)", 0)
    if model_id == "svm":
        n = f.get("Support vectors")
        return (f"{n} support vectors", n) if n is not None else ("approximation (300 features)", 300)
    return "", 0


class CompareTab(QWidget):
    busy = Signal(str)
    idle = Signal()
    failed = Signal(str)
    jump_to = Signal(str)

    def __init__(self, model_tabs: dict, parent=None):
        super().__init__(parent)
        self.model_tabs = model_tabs
        self.ds: Dataset | None = None
        self.selected: list[str] = ["tree", "knn", "linear"]
        self.results: dict[str, dict] = {}
        self.used_params: dict[str, dict] = {}
        self.used_opts: dict = {}
        self.cache: dict[str, dict] = {}
        self.pool: ProcessPoolExecutor | None = None
        self.worker: CompareWorker | None = None
        self._errors: list[str] = []
        self._pending_keys: list[tuple[str, dict]] = []
        self._jobs_all: list[tuple[str, dict, dict]] = []
        self.panel_axes: list = []
        self.hl_arts: list = []
        self._build()
        for mid, tab in model_tabs.items():
            tab.settings_changed.connect(lambda _m=None: self._update_cards())

    # ---- layout ------------------------------------------------------------
    def _build(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        page = QScrollArea()
        page.setWidgetResizable(True)
        page.setFrameShape(QScrollArea.NoFrame)
        outer.addWidget(page)
        content = QWidget()
        page.setWidget(content)
        v = QVBoxLayout(content)
        self.scenario = QLabel()
        self.scenario.setObjectName("concept")
        self.scenario.setWordWrap(True)
        self.scenario.setTextFormat(Qt.RichText)
        self.scenario.setVisible(False)
        v.addWidget(self.scenario)

        row = QHBoxLayout()
        row.addWidget(QLabel("Models:"))
        self.chips: dict[str, QPushButton] = {}
        for mid in MODEL_ORDER:
            b = QPushButton(get_spec(mid).display_name)
            b.setCheckable(True)
            b.setChecked(mid in self.selected)
            b.setToolTip(f"Include {get_spec(mid).display_name} in the comparison. Each model type can be added once, up to {MAX_MODELS} models. "
                         "Its settings are read live from its own tab.")
            b.toggled.connect(lambda on, m=mid: self._toggle(m, on))
            row.addWidget(b)
            self.chips[mid] = b
        row.addStretch()
        v.addLayout(row)

        row = QHBoxLayout()
        self.cv = QCheckBox("Cross-validation on the training set (k = 5)")
        self.cv.setToolTip("Adds the mean and spread of 5-fold cross-validation. The held-out test set is never used for it.")
        self.multiseed = QCheckBox("Repeat across all five seeds")
        self.multiseed.setToolTip("Re-runs every model on seeds 1 to 5 and shows the mean and spread of the test score.")
        self.btn = QPushButton("Compare")
        self.btn.setObjectName("primary")
        self.btn.setToolTip("Train all selected models in parallel on the same data and the same split.")
        self.btn.clicked.connect(self.run_compare)
        self.progress = QProgressBar()
        self.progress.setMaximumWidth(220)
        self.progress.setVisible(False)
        self.status = QLabel("")
        for w in (self.cv, self.multiseed, self.btn, self.progress, self.status):
            row.addWidget(w)
        row.addStretch()
        v.addLayout(row)

        self.cards_area = QScrollArea()
        self.cards_area.setWidgetResizable(True)
        self.cards_area.setFixedHeight(80)
        self.cards_host = QWidget()
        self.cards_layout = QHBoxLayout(self.cards_host)
        self.cards_area.setWidget(self.cards_host)
        v.addWidget(self.cards_area)

        self.tabs = QTabWidget()
        # panels
        pw = QWidget()
        pv = QVBoxLayout(pw)
        self.panel_scroll = QScrollArea()
        self.panel_scroll.setWidgetResizable(False)
        self.empty = QLabel("Choose models and press Compare.")
        self.empty.setObjectName("empty")
        self.panel_scroll.setWidget(self.empty)
        pv.addWidget(self.panel_scroll, 1)
        self.point_label = QLabel("Click a point in any panel to highlight it in every panel.")
        self.point_label.setWordWrap(True)
        pv.addWidget(self.point_label)
        self.tabs.addTab(pw, "Panels")
        # metrics
        mw = QWidget()
        mv = QVBoxLayout(mw)
        r = QHBoxLayout()
        r.addWidget(QLabel("Metric for the chart:"))
        self.metric_box = QComboBox()
        self.metric_box.setToolTip("Choose which metric the bar chart shows. Test bars are dark and train bars are light.")
        self.metric_box.currentIndexChanged.connect(lambda _: self._render_metrics())
        r.addWidget(self.metric_box)
        r.addStretch()
        mv.addLayout(r)
        self.fig_bar = Figure(figsize=(8, 2.8), dpi=100)
        self.canvas_bar = FigureCanvasQTAgg(self.fig_bar)
        self.canvas_bar.setMinimumHeight(260)
        mv.addWidget(self.canvas_bar)
        self.table = self._make_table()
        self.table.setMinimumHeight(150)
        mv.addWidget(self.table, 1)
        self.tabs.addTab(mw, "Metrics")
        # curves
        cw = QWidget()
        cvl = QVBoxLayout(cw)
        self.fig_curves = Figure(figsize=(8, 3.2), dpi=100)
        self.canvas_curves = FigureCanvasQTAgg(self.fig_curves)
        cvl.addWidget(self.canvas_curves, 1)
        cnote = QLabel("Loss versus the number of trees for random forest and gradient boosting. Solid lines are test loss and dashed lines are training loss. "
                       "A rising test curve after its minimum is overfitting from too many boosting rounds.")
        cnote.setWordWrap(True)
        cvl.addWidget(cnote)
        self.tabs.addTab(cw, "Tree curves")
        # trade-offs
        tw = QWidget()
        tv = QVBoxLayout(tw)
        self.trade = self._make_table()
        tv.addWidget(self.trade, 1)
        note = QLabel("Timing is machine dependent. Interpretability is a general guideline for the model type, not a measurement.")
        note.setWordWrap(True)
        tv.addWidget(note)
        self.tabs.addTab(tw, "Trade-offs")
        self.tabs.setMinimumHeight(500)
        v.addWidget(self.tabs, 1)

        row = QHBoxLayout()
        self.btn_png = QPushButton("Export panels PNG")
        self.btn_png.setToolTip("Save the panel grid as a PNG image.")
        self.btn_png.clicked.connect(self.export_png)
        self.btn_csv = QPushButton("Export results CSV")
        self.btn_csv.setToolTip("Save the results table as a CSV file.")
        self.btn_csv.clicked.connect(self.export_csv)
        row.addStretch()
        row.addWidget(self.btn_png)
        row.addWidget(self.btn_csv)
        v.addLayout(row)
        self._update_cards()

    def _make_table(self) -> QTableView:
        t = QTableView()
        t.setSortingEnabled(True)
        t.setEditTriggers(QAbstractItemView.NoEditTriggers)
        t.setSelectionBehavior(QAbstractItemView.SelectRows)
        t.setAlternatingRowColors(True)
        t.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        t.horizontalHeader().setStretchLastSection(True)
        return t

    # ---- public api ------------------------------------------------------------
    def selection(self) -> list[str]:
        return [m for m in MODEL_ORDER if m in self.selected]

    def set_selection(self, ids):
        ids = [m for m in ids if m in self.chips][:MAX_MODELS]
        for m, b in self.chips.items():
            b.blockSignals(True)
            b.setChecked(m in ids)
            b.blockSignals(False)
        self.selected = ids
        self._update_cards()

    def options(self) -> dict:
        return {"cv": self.cv.isChecked(), "multiseed": self.multiseed.isChecked(), "metric": self.metric_box.currentData() or ""}

    def set_options(self, d: dict):
        self.cv.setChecked(bool(d.get("cv", False)))
        self.multiseed.setChecked(bool(d.get("multiseed", False)))
        self._pending_metric = d.get("metric") or None

    def set_dataset(self, ds: Dataset):
        self.ds = ds
        self.results = {}
        self.used_params = {}
        self._pending_metric = None
        self.metric_box.blockSignals(True)
        self.metric_box.clear()
        keys = M.CLS_KEYS if ds.problem == CLS else M.REG_KEYS
        for k in keys:
            self.metric_box.addItem(M.LABELS[k], k)
        self.metric_box.setCurrentIndex(max(0, self.metric_box.findData(M.main_score_key(ds.problem))))
        self.metric_box.blockSignals(False)
        self._clear_views("Data changed, so results were cleared. Press Compare to train the selected models again.")
        self._update_cards()

    def show_scenario(self, sc: dict):
        self.scenario.setText(f"<b>{sc['title']}</b><br>{sc['blurb']}<br><b>What to look for:</b> {sc['look_for']}")
        self.scenario.setVisible(True)

    # ---- selection and cards -----------------------------------------------------
    def _toggle(self, mid: str, on: bool):
        if on and len(self.selected) >= MAX_MODELS:
            self.chips[mid].blockSignals(True)
            self.chips[mid].setChecked(False)
            self.chips[mid].blockSignals(False)
            return
        if on and mid not in self.selected:
            self.selected.append(mid)
        if not on and mid in self.selected:
            self.selected.remove(mid)
        self._update_cards()

    def _is_stale(self, mid: str) -> str:
        if not self.results:
            return ""
        if mid not in self.used_params:
            return "Not run"
        return "Stale" if self.used_params[mid] != self.model_tabs[mid].current_params() else ""

    def _update_cards(self):
        while self.cards_layout.count():
            it = self.cards_layout.takeAt(0)
            w = it.widget()
            if w is not None:
                w.hide()              # hide now, otherwise it keeps painting until it is deleted
                w.setParent(None)
                w.deleteLater()
        problem = self.ds.problem if self.ds else CLS
        for mid in self.selection():
            spec = get_spec(mid)
            tab = self.model_tabs[mid]
            card = QWidget()
            cv = QVBoxLayout(card)
            cv.setContentsMargins(6, 2, 6, 2)
            top = QHBoxLayout()
            name = QLabel(f"<b>{spec.display_name}</b>")
            top.addWidget(name)
            edit = QPushButton("Edit")
            edit.setObjectName("link")
            edit.setToolTip(f"Open the {spec.display_name} tab to change its settings.")
            edit.clicked.connect(lambda _=False, m=mid: self.jump_to.emit(m))
            top.addWidget(edit)
            badge_text = self._is_stale(mid)
            if badge_text:
                b = QLabel(badge_text)
                b.setObjectName("stale")
                b.setToolTip("The settings changed since the last comparison run. Press Compare to update.")
                top.addWidget(b)
            top.addStretch()
            cv.addLayout(top)
            summ = QLabel(spec.summary(tab.current_params(problem), problem))
            summ.setStyleSheet("color: #444;")
            cv.addWidget(summ)
            card.setObjectName("card")
            self.cards_layout.addWidget(card)
        self.cards_layout.addStretch()

    # ---- running -------------------------------------------------------------------
    def _get_pool(self) -> ProcessPoolExecutor:
        if self.pool is None:
            n = min(6, max(2, (os.cpu_count() or 4) - 1))
            self.pool = ProcessPoolExecutor(n, mp_context=mp.get_context("spawn"))
            for _ in range(n):
                self.pool.submit(warm_up)
        return self.pool

    def showEvent(self, ev):
        super().showEvent(ev)
        try:
            self._get_pool()          # start warming up the worker processes
        except Exception:
            log.exception("Could not start the worker pool")

    def _kill_pool(self):
        pool, self.pool = self.pool, None
        if pool is None:
            return
        try:
            for p in list(getattr(pool, "_processes", {}).values()):
                p.terminate()
            pool.shutdown(wait=False, cancel_futures=True)
        except Exception:
            log.exception("Error while stopping the worker pool")

    def run_compare(self):
        if self.ds is None or self.worker is not None:
            return
        if not self.selected:
            self.status.setText("Add at least one model.")
            return
        opts = self.options()
        cfg, seed = self.ds.config, self.ds.seed
        self._jobs_all, self.used_params, self.used_opts = [], {}, opts
        for mid in self.selection():
            params = self.model_tabs[mid].current_params()
            self.used_params[mid] = params
            main = make_job(cfg, seed, mid, params, True, opts["cv"])
            self._jobs_all.append((job_key(main), main, {"mid": mid, "main": True}))
            if opts["multiseed"]:
                for s in SEEDS:
                    if s != seed:
                        j = make_job(cfg, s, mid, params, False, False)
                        self._jobs_all.append((job_key(j), j, {"mid": mid, "main": False}))
        pending = [(k, j) for k, j, _ in self._jobs_all if k not in self.cache]
        self._errors = []
        if not pending:
            self.status.setText("All results were cached, no retraining needed.")
            self._collect()
            return
        try:
            pool = self._get_pool()
        except Exception:
            self.failed.emit(traceback.format_exc())
            return
        self.worker = CompareWorker(pool, pending)
        self.worker.progress.connect(self._on_progress)
        self.worker.result.connect(lambda k, r: self.cache.__setitem__(k, r))
        self.worker.job_failed.connect(lambda k, tb: self._errors.append(tb))
        self.worker.failed.connect(lambda tb: self._errors.append(tb))
        self.worker.finished.connect(self._on_finished)
        self.progress.setRange(0, len(pending))
        self.progress.setValue(0)
        self.progress.setVisible(True)
        self.btn.setEnabled(False)
        self.status.setText(f"Training {len(pending)} job(s) in parallel ({len(self.cache)} cached)...")
        self.busy.emit("Comparing models...")
        self.worker.start()

    def _on_progress(self, done: int, total: int):
        self.progress.setValue(done)
        self.status.setText(f"{done} of {total} jobs finished")
        self.busy.emit(f"Comparing models: {done} of {total} jobs finished")

    def _on_finished(self):
        cancelled = self.worker is not None and self.worker._cancel
        self.worker = None
        self.progress.setVisible(False)
        self.btn.setEnabled(True)
        self.idle.emit()
        if cancelled:
            self.status.setText("Cancelled.")
            return
        if self._errors:
            self.failed.emit(self._errors[0])
        self.status.setText("Done." if not self._errors else "Done with errors (see message).")
        self._collect()

    def cancel(self):
        if self.worker is not None:
            self.worker.cancel()
            self._kill_pool()
            self.worker.wait(2000)

    def shutdown(self):
        self.cancel()
        self._kill_pool()

    # ---- collecting results ------------------------------------------------------------
    def _collect(self):
        res: dict[str, dict] = {}
        for key, job, meta in self._jobs_all:
            r = self.cache.get(key)
            if r is None:
                continue
            e = res.setdefault(meta["mid"], {"main": None, "others": []})
            if meta["main"]:
                e["main"] = r
            else:
                e["others"].append(r)
        self.results = {m: e for m, e in res.items() if e["main"] is not None}
        self._update_cards()
        self._render_all()

    def _clear_views(self, text: str):
        self.panel_axes, self.hl_arts = [], []
        self.empty = QLabel(text)
        self.empty.setObjectName("empty")
        self.empty.setWordWrap(True)
        self.panel_scroll.setWidget(self.empty)
        self.fig_bar.clear()
        self.canvas_bar.draw_idle()
        self.fig_curves.clear()
        self.canvas_curves.draw_idle()
        for t in (self.table, self.trade):
            t.setModel(QStandardItemModel())
        self.point_label.setText("")

    # ---- rendering -----------------------------------------------------------------------
    def _render_all(self):
        if not self.results:
            self._clear_views("No results.")
            return
        self._render_panels()
        self._render_metrics()
        self._render_curves()
        self._render_tradeoffs()

    def _main_key(self) -> str:
        return M.main_score_key(self.ds.problem)

    def _seed_stats(self, mid: str, key: str):
        e = self.results[mid]
        vals = [e["main"]["test"][key]] + [o["test"][key] for o in e["others"]]
        if len(vals) < 2:
            return None
        return float(np.mean(vals)), float(np.std(vals)), len(vals)

    def _render_panels(self):
        ds = self.ds
        mids = [m for m in self.selection() if m in self.results]
        n = len(mids)
        cols = 2 if n <= 4 else 3 if n <= 6 else 4
        rows = math.ceil(n / cols)
        fig = Figure(figsize=(cols * 3.3, rows * 3.6), dpi=100)
        axes = fig.subplots(rows, cols, squeeze=False)
        self.panel_axes, self.hl_arts = [], []
        mk = self._main_key()
        self._panel_models = []
        for i, mid in enumerate(mids):
            ax = axes[i // cols][i % cols]
            r = self.results[mid]["main"]
            xl = f"train {r['train'][mk]:.2f} | test {r['test'][mk]:.2f} | {r['train_time'] * 1000:.0f} ms"
            if r["cv"]:
                xl += f"\ncv {r['cv']['mean']:.2f} +/- {r['cv']['std']:.2f}"
            draw_panel(ax, ds, r, get_spec(mid).display_name, xl)
            hl = ax.scatter([], [], s=170, facecolors="none", edgecolors="black", linewidths=2.2, zorder=10)
            self.panel_axes.append(ax)
            self.hl_arts.append(hl)
            self._panel_models.append(mid)
        for j in range(n, rows * cols):
            axes[j // cols][j % cols].axis("off")
        fig.tight_layout(pad=1.0)
        if ds.problem != CLS and ds.n_inputs == 2:
            from matplotlib.cm import ScalarMappable
            cb = fig.colorbar(ScalarMappable(norm=reg_norm(ds), cmap=REG_CMAP), ax=axes.ravel().tolist(), fraction=0.02, pad=0.02)
            cb.set_label("y (true or predicted)", fontsize=7)
            cb.ax.tick_params(labelsize=7)
        self.fig_panels = fig
        canvas = FigureCanvasQTAgg(fig)
        canvas.setFixedSize(int(cols * 3.3 * 100), int(rows * 3.6 * 100))
        canvas.mpl_connect("button_press_event", self._on_panel_click)
        self.panel_scroll.setWidget(canvas)
        self.point_label.setText("Click a point in any panel to highlight it in every panel.")

    def _on_panel_click(self, ev):
        if ev.inaxes not in self.panel_axes or ev.xdata is None:
            return
        ds = self.ds
        X, y = ds.X, ds.y
        if ds.n_inputs == 1:
            lo, hi = y_limits(ds)
            dist = np.hypot((X[:, 0] - ev.xdata) / 2, (y - ev.ydata) / (hi - lo))
        else:
            dist = np.hypot(X[:, 0] - ev.xdata, X[:, 1] - ev.ydata) / 2
        i = int(np.argmin(dist))
        if dist[i] > 0.05:
            return
        pt = np.array([[X[i, 0], y[i] if ds.n_inputs == 1 else X[i, 1]]])
        for art in self.hl_arts:
            art.set_offsets(pt)
        n_tr = len(ds.y_train)
        parts = []
        for mid in self._panel_models:
            r = self.results[mid]["main"]
            pr = (r["pred_train"][i] if i < n_tr else r["pred_test"][i - n_tr])
            parts.append(f"{get_spec(mid).display_name}: {int(pr) if ds.problem == CLS else round(float(pr), 2)}")
        truth = f"class {int(y[i])}" if ds.problem == CLS else f"y = {y[i]:.2f}"
        self.point_label.setText(f"Point {i} ({'test' if i >= n_tr else 'train'}), true {truth}. Predictions: " + "; ".join(parts))
        self.fig_panels.canvas.draw_idle()

    def _render_metrics(self):
        if not self.results:
            return
        key = self.metric_box.currentData() or self._main_key()
        pm = getattr(self, "_pending_metric", None)
        if pm and self.metric_box.findData(pm) >= 0:
            self._pending_metric = None
            self.metric_box.blockSignals(True)
            self.metric_box.setCurrentIndex(self.metric_box.findData(pm))
            self.metric_box.blockSignals(False)
            key = pm
        mids = [m for m in self.selection() if m in self.results]
        fig = self.fig_bar
        fig.clear()
        ax = fig.add_subplot(111)
        x = np.arange(len(mids))
        tr = [self.results[m]["main"]["train"][key] for m in mids]
        te = [self.results[m]["main"]["test"][key] for m in mids]
        stats = [self._seed_stats(m, key) for m in mids]
        te_h = [s[0] if s else t for s, t in zip(stats, te)]
        err = [s[1] if s else 0 for s in stats]
        w = 0.38
        ax.bar(x - w / 2, tr, w, color="#9cc7e4", label="Train", edgecolor="#555", linewidth=0.5)
        bars = ax.bar(x + w / 2, te_h, w, color=BLUE, label="Test (mean of seeds)" if any(stats) else "Test",
                      yerr=err if any(stats) else None, capsize=3, edgecolor="#222", linewidth=0.6)
        lower = key in M.LOWER_IS_BETTER
        best = int(np.argmin(te_h) if lower else np.argmax(te_h))
        bars[best].set_edgecolor(ORANGE)
        bars[best].set_linewidth(3)
        ax.annotate("best", (x[best] + w / 2, te_h[best]), textcoords="offset points", xytext=(0, 6 + (err[best] * 0 if True else 0)),
                    ha="center", fontsize=7, color="#a56a00")
        if key == self._main_key() and any(self.results[m]["main"]["cv"] for m in mids):
            cvm = [self.results[m]["main"]["cv"] for m in mids]
            xs = [xi for xi, c in zip(x, cvm) if c]
            ax.errorbar(xs, [c["mean"] for c in cvm if c], yerr=[c["std"] for c in cvm if c], fmt="D", color="black",
                        ms=4, capsize=3, label="Cross-validation (train only)")
        ax.set_xticks(x)
        ax.set_xticklabels([get_spec(m).display_name for m in mids], fontsize=8)
        ax.set_ylabel(M.LABELS[key])
        ax.set_title(f"{M.LABELS[key]} ({'lower' if lower else 'higher'} is better)", fontsize=9)
        ax.legend(frameon=False, fontsize=7, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.12))
        fig.subplots_adjust(left=0.08, right=0.99, top=0.88, bottom=0.3)
        self.canvas_bar.draw_idle()
        self._fill_metric_table(mids)

    def _render_curves(self):
        fig = self.fig_curves
        fig.clear()
        ax = fig.add_subplot(111)
        drawn = False
        for m in self.selection():
            r = self.results.get(m, {}).get("main")
            if not r or not r.get("curves"):
                continue
            c = r["curves"]
            col = MODEL_COLORS[MODEL_ORDER.index(m)]
            x = np.arange(1, len(c["test_loss"]) + 1)
            ax.plot(x, c["test_loss"], color=col, lw=2, label=f"{get_spec(m).display_name} (test)")
            ax.plot(x, c["train_loss"], color=col, lw=1.2, ls="--", label=f"{get_spec(m).display_name} (train)")
            b = int(np.argmin(c["test_loss"]))
            ax.plot([x[b]], [c["test_loss"][b]], marker="v", color="black", ms=6)
            drawn = True
        if not drawn:
            ax.text(0.5, 0.5, "Add random forest or gradient boosting to see these curves.", ha="center", va="center",
                    transform=ax.transAxes, color="#666")
            ax.set_xticks([]); ax.set_yticks([])
        else:
            ax.set_xlabel("Number of trees")
            ax.set_ylabel("Loss (log loss or MSE)")
            ax.legend(frameon=False, fontsize=7)
        fig.subplots_adjust(left=0.09, right=0.98, top=0.95, bottom=0.14)
        self.canvas_curves.draw_idle()

    def _fill_metric_table(self, mids):
        keys = M.CLS_KEYS if self.ds.problem == CLS else M.REG_KEYS
        mk = self._main_key()
        headers = ["Model", f"Train {M.LABELS[mk]}", "Gap"] + [f"Test {M.LABELS[k]}" for k in keys] + ["CV mean +/- std", "Seeds mean +/- std"]
        rows = []
        for m in mids:
            r = self.results[m]["main"]
            cv = r["cv"]
            st = self._seed_stats(m, mk)
            row = [(get_spec(m).display_name, get_spec(m).display_name),
                   (f"{r['train'][mk]:.3f}", r["train"][mk]),
                   (f"{r['train'][mk] - r['test'][mk]:+.3f}", r["train"][mk] - r["test"][mk])]
            row += [(f"{r['test'][k]:.3f}", r["test"][k]) for k in keys]
            row.append((f"{cv['mean']:.3f} +/- {cv['std']:.3f}", cv["mean"]) if cv else ("not run", float("nan")))
            row.append((f"{st[0]:.3f} +/- {st[1]:.3f} (n={st[2]})", st[0]) if st else ("not run", float("nan")))
            rows.append(row)
        self._fill_table(self.table, headers, rows)

    def _render_tradeoffs(self):
        mids = [m for m in self.selection() if m in self.results]
        mk = self._main_key()
        headers = [f"Test {M.LABELS[mk]}", "CV mean +/- std", "Seeds mean +/- std", "Training time (ms)", "Prediction time (ms)",
                   "Model size", "Interpretability*"]
        headers = ["Model"] + headers
        order = {"High": 3, "Medium": 2, "Low": 1}
        rows = []
        for m in mids:
            r = self.results[m]["main"]
            cv, st = r["cv"], self._seed_stats(m, mk)
            txt, num = size_text(m, r["facts"])
            interp = get_spec(m).interpretability
            rows.append([
                (get_spec(m).display_name, get_spec(m).display_name),
                (f"{r['test'][mk]:.3f}", r["test"][mk]),
                (f"{cv['mean']:.3f} +/- {cv['std']:.3f}", cv["mean"]) if cv else ("not run", float("nan")),
                (f"{st[0]:.3f} +/- {st[1]:.3f}", st[0]) if st else ("not run", float("nan")),
                (f"{r['train_time'] * 1000:.0f}", r["train_time"]),
                (f"{r['predict_time'] * 1000:.2f}", r["predict_time"]),
                (txt, float(num)),
                (interp, float(order[interp])),
            ])
        self._fill_table(self.trade, headers, rows)

    def _fill_table(self, view: QTableView, headers, rows):
        model = QStandardItemModel(len(rows), len(headers))
        model.setHorizontalHeaderLabels(headers)
        for i, row in enumerate(rows):
            for j, (text, sort) in enumerate(row):
                it = QStandardItem(text)
                it.setData(sort if not (isinstance(sort, float) and math.isnan(sort)) else -1e18, Qt.UserRole)
                if j:
                    it.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                model.setItem(i, j, it)
        proxy = QSortFilterProxyModel(view)
        proxy.setSourceModel(model)
        proxy.setSortRole(Qt.UserRole)
        view.setModel(proxy)
        view._src_model = model          # keep a reference alive
        view.sortByColumn(0, Qt.AscendingOrder)

    # ---- export --------------------------------------------------------------------------
    def export_png(self):
        if not getattr(self, "panel_axes", None):
            self.status.setText("Nothing to export yet.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export panels", "comparison_panels.png", "PNG image (*.png)")
        if path:
            path = fix_suffix(path, ".png", IMAGE_SUFFIXES)
            if safe_save(self, path, lambda p: self.fig_panels.savefig(p, dpi=250, bbox_inches="tight", facecolor="white")):
                self.status.setText(f"Saved {path}")

    def results_rows(self) -> tuple[list[str], list[list]]:
        keys = M.CLS_KEYS if self.ds.problem == CLS else M.REG_KEYS
        mk = self._main_key()
        head = ["model", "settings"] + [f"train_{k}" for k in keys] + [f"test_{k}" for k in keys] + [
            "cv_mean", "cv_std", "seeds_mean", "seeds_std", "train_time_ms", "predict_time_ms", "model_size", "interpretability_guideline"]
        rows = []
        for m in self.selection():
            if m not in self.results:
                continue
            r = self.results[m]["main"]
            st = self._seed_stats(m, mk)
            cv = r["cv"]
            rows.append([get_spec(m).display_name, get_spec(m).summary(self.used_params[m], self.ds.problem)]
                        + [round(r["train"][k], 6) for k in keys] + [round(r["test"][k], 6) for k in keys]
                        + [round(cv["mean"], 6) if cv else "", round(cv["std"], 6) if cv else "",
                           round(st[0], 6) if st else "", round(st[1], 6) if st else "",
                           round(r["train_time"] * 1000, 3), round(r["predict_time"] * 1000, 3),
                           size_text(m, r["facts"])[0], get_spec(m).interpretability])
        return head, rows

    def export_csv(self):
        if not self.results:
            self.status.setText("Nothing to export yet.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export results", "comparison_results.csv", "CSV (*.csv)")
        if path:
            path = fix_suffix(path, ".csv")
            head, rows = self.results_rows()

            def write(p):
                with open(p, "w", newline="", encoding="utf-8") as f:
                    w = csv.writer(f)
                    w.writerow(head)
                    w.writerows(rows)
            if safe_save(self, path, write):
                self.status.setText(f"Saved {path}")
