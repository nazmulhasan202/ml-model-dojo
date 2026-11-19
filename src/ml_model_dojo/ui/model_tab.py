"""One model tab: hyperparameters (left), main plot (center), metrics (right)."""
from __future__ import annotations

import logging

from PySide6.QtCore import QTimer, Qt, Signal
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (QCheckBox, QFileDialog, QGroupBox, QHBoxLayout, QLabel, QPushButton, QScrollArea,
                               QSlider, QSplitter, QTabWidget, QVBoxLayout, QWidget)

from ..data.datasets import Dataset
from ..models.base import CLS, ModelSpec
from ..models.evaluate import evaluate
from ..workers.qt_workers import FitTask, pool
from .details import make_details
from .dialogs import IMAGE_SUFFIXES, fix_suffix, safe_save
from .metrics_panel import MetricsPanel
from .param_panel import ParamPanel
from .plots import BoundaryPlot

log = logging.getLogger("msp")
TICK_MS = 66
EMPTY_TEXT = "Choose a dataset and press Train"


class ModelTab(QWidget):
    settings_changed = Signal(str)        # model id
    busy = Signal(str)
    idle = Signal()
    message = Signal(str)
    failed = Signal(str)

    def __init__(self, spec: ModelSpec, parent=None):
        super().__init__(parent)
        self.spec = spec
        self.data: Dataset | None = None
        self.run = None
        self.saved: dict[str, dict] = {}       # problem -> params
        self._schema_key = None
        self._token = 0
        self._after_fit = None                 # "play", "step" or "end"
        self._playing = False
        self._autofit = False                  # live refit after the first Fit of a non-iterative model
        self._acc = 0.0
        self._ticks = 0
        self.timer = QTimer(self)
        self.timer.setInterval(TICK_MS)
        self.timer.timeout.connect(self._tick)
        # Non-iterative models draw a coarse grid first, then refine once the sliders stop moving.
        self.refine = QTimer(self)
        self.refine.setSingleShot(True)
        self.refine.setInterval(350)
        self.refine.timeout.connect(lambda: self._refresh(fine=True))
        self._build()

    # ---- layout ----------------------------------------------------------
    def _build(self):
        root = QHBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)
        split = QSplitter(Qt.Horizontal)
        root.addWidget(split)

        # left column
        left = QWidget()
        lv = QVBoxLayout(left)
        gb = QGroupBox("Hyperparameters")
        gv = QVBoxLayout(gb)
        self.params = ParamPanel()
        self.params.changed.connect(self._on_params)
        gv.addWidget(self.params)
        reset_defaults = QPushButton("Reset to defaults")
        reset_defaults.setToolTip("Restore this model's default hyperparameters.")
        reset_defaults.clicked.connect(self.reset_defaults)
        gv.addWidget(reset_defaults)
        lv.addWidget(gb)
        concept = QLabel(self.spec.concept)
        concept.setObjectName("concept")
        concept.setWordWrap(True)
        concept.setAccessibleName("Concept card")
        lv.addWidget(concept)
        lv.addStretch()
        sa = QScrollArea()
        sa.setWidgetResizable(True)
        sa.setWidget(left)
        sa.setMinimumWidth(310)
        sa.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        split.addWidget(sa)

        # center column
        center = QWidget()
        cv = QVBoxLayout(center)
        cv.setContentsMargins(0, 0, 0, 0)
        self.plot = BoundaryPlot()
        self.plot.hovered.connect(self.message)
        if self.spec.id == "knn":
            self.plot.hover_hook = self._knn_hover
        cv.addWidget(self.plot, 1)
        self.state_label = QLabel(EMPTY_TEXT)
        self.state_label.setObjectName("empty")
        self.state_label.setAlignment(Qt.AlignCenter)
        cv.addWidget(self.state_label)
        bar = QHBoxLayout()
        bar.setSpacing(10)
        self.btn_play = QPushButton("Train")
        self.btn_play.setToolTip("Train the model and animate it. Press again to pause.")
        self.btn_step = QPushButton("Step")
        self.btn_step.setToolTip("Advance one iteration, split, tree or epoch.")
        self.btn_reset = QPushButton("Reset")
        self.btn_reset.setToolTip("Go back to the start of the animation.")
        self.btn_fit = QPushButton("Train")
        self.btn_play.setObjectName("primary")
        self.btn_fit.setObjectName("primary")
        self.btn_fit.setToolTip("Train the model now. After the first training, hyperparameter changes retrain it automatically.")
        for b, fn in ((self.btn_play, self.toggle_play), (self.btn_step, self.step_once),
                      (self.btn_reset, self.reset_run), (self.btn_fit, self.fit_now)):
            b.clicked.connect(fn)
            bar.addWidget(b)
        bar.addSpacing(14)
        self.speed_label = QLabel("Speed")
        bar.addWidget(self.speed_label)
        self.speed = QSlider(Qt.Horizontal)
        self.speed.setRange(1, 10)
        self.speed.setValue(5)
        self.speed.setMaximumWidth(120)
        self.speed.setToolTip("How fast the animation runs.")
        self.speed.setAccessibleName("Animation speed")
        bar.addWidget(self.speed)
        bar.addStretch()
        cv.addLayout(bar)
        tog = QHBoxLayout()
        self.toggles = {}
        for key, text, tip in (("train", "Train points", "Show or hide the filled training points."),
                               ("test", "Test points", "Show or hide the hollow test points."),
                               ("errors", "Errors", "Red rings mark misclassified test points. For regression the test residuals are drawn."),
                               ("truth", "True function", "Show the noise-free function that generated the data (a dotted line, or dashed contour lines for two inputs).")):
            cb = QCheckBox(text)
            cb.setChecked(True)
            cb.setToolTip(tip)
            cb.toggled.connect(lambda on, k=key: self.plot.set_flag(k, on))
            tog.addWidget(cb)
            self.toggles[key] = cb
        export = QPushButton("Export PNG")
        export.setToolTip("Save the main plot as a paper-style PNG image.")
        export.clicked.connect(self.export_png)
        tog.addStretch()
        tog.addWidget(export)
        cv.addLayout(tog)
        split.addWidget(center)

        # right column
        self.metrics = MetricsPanel()
        self.details = make_details(self.spec.id)
        tabs = QTabWidget()
        tabs.addTab(self.metrics, "Metrics")
        tabs.addTab(self.details, "Details")
        tabs.setMinimumWidth(300)
        split.addWidget(tabs)
        split.setStretchFactor(0, 0)
        split.setStretchFactor(1, 1)
        split.setStretchFactor(2, 0)
        split.setSizes([320, 520, 360])
        for key, fn in (("F5", self.toggle_play), ("F6", self.step_once), ("F7", self.reset_run)):
            sc = QShortcut(QKeySequence(key), self)
            sc.setContext(Qt.WidgetWithChildrenShortcut)
            sc.activated.connect(fn)
        self.btn_play.setToolTip("Train the model and animate it. Press again to pause. Shortcut: F5.")
        self.btn_step.setToolTip("Advance one iteration, split, tree or epoch. Shortcut: F6.")
        self.btn_reset.setToolTip("Go back to the start of the animation. Shortcut: F7.")
        self._set_button_state()

    # ---- data and params -------------------------------------------------
    @property
    def problem(self) -> str:
        return self.data.problem if self.data else CLS

    def current_params(self, problem: str | None = None) -> dict:
        problem = problem or self.problem
        return self.spec.clean_params(self.saved.get(problem, {}), problem)

    def set_dataset(self, data: Dataset):
        self._cancel_run()
        self.data = data
        key = (data.problem, data.n_inputs)
        if key != self._schema_key:
            self._schema_key = key
            schema = self.spec.param_schema(data.problem)
            self.params.set_schema(schema, self.current_params(), data.n_inputs)
            self.metrics.set_problem(data.problem)
        self._fix_feature_params()
        self.plot.set_dataset(data)
        self.toggles["truth"].setVisible(data.problem != CLS)
        self.toggles["errors"].setText("Errors" if data.problem == CLS else "Residuals")
        self._autofit = False
        self.run = None
        self.metrics.clear(EMPTY_TEXT + ". Models were cleared because the data changed.")
        self.details.clear()
        self._show_state(EMPTY_TEXT + " (models were cleared, retrain)")
        self._set_button_state()

    def _fix_feature_params(self):
        """The network feature toggles depend on the number of inputs."""
        if self.spec.id == "mlp":
            p = self.current_params()
            self.saved[self.problem] = p

    def apply_params(self, by_problem: dict):
        """Load saved params (from a scenario or file) for every problem type."""
        for problem, params in (by_problem or {}).items():
            self.saved[problem] = self.spec.clean_params(params, problem)
        if self.data is not None:
            self.params.set_values(self.current_params())
            self._invalidate("Settings loaded.")
            self.settings_changed.emit(self.spec.id)

    def all_params(self) -> dict:
        return {p: self.current_params(p) for p in ("classification", "regression") if p in self.spec.supports}

    def reset_defaults(self):
        self.saved[self.problem] = self.spec.default_params(self.problem)
        self.params.set_values(self.saved[self.problem])
        self._on_params(self.params.values())

    def _on_params(self, values: dict):
        self.saved[self.problem] = self.spec.clean_params(values, self.problem)
        self.settings_changed.emit(self.spec.id)
        if self.data is None:
            return
        if not self.spec.iterative(self.problem) and self._autofit:
            self.fit_now()
        else:
            self._invalidate("Settings changed. Press Train to train again." if self.spec.iterative(self.problem)
                             else "Settings changed. Press Train.")

    def _invalidate(self, text: str):
        self._cancel_run()
        self.run = None
        self.plot.update_run(None)
        self.metrics.clear(text)
        self.details.clear()
        self._show_state(text)
        self._set_button_state()

    # ---- run control -----------------------------------------------------
    def _cancel_run(self):
        self.refine.stop()
        self._token += 1
        self._after_fit = None
        self._pause()
        self.idle.emit()

    def cancel(self):
        if self._after_fit is not None or self._playing:
            self._cancel_run()
            self._show_state("Cancelled.")
            self._set_button_state()

    def _start_fit(self, then: str):
        if self.data is None:
            return
        self._token += 1
        self._after_fit = then
        run = self.spec.start(self.current_params(), self.problem, self.data_seed, self.data)
        task = FitTask(self._token, run)
        task.signals.done.connect(self._fit_done, Qt.QueuedConnection)
        task.signals.failed.connect(self._fit_failed, Qt.QueuedConnection)
        self._task = task
        self.busy.emit(f"Fitting {self.spec.display_name}...")
        self._show_state("Fitting...")
        self._set_button_state()
        pool().start(task)

    @property
    def data_seed(self) -> int:
        return self.data.seed

    def _fit_done(self, token, run):
        if token != self._token:
            return
        self.idle.emit()
        self.run = run
        action, self._after_fit = self._after_fit, None
        if action == "end" or not self.spec.iterative(self.problem):
            run.go_to_end()
            self._autofit = True
            self._refresh(fine=False)
            self.refine.start()
        elif action == "step":
            run.step()
            self._refresh(fine=True)
        elif action == "play":
            self._playing = True
            self.timer.start()
            self._refresh(fine=False)
        self._set_button_state()

    def _fit_failed(self, token, tb):
        if token != self._token:
            return
        self.idle.emit()
        self._after_fit = None
        self.run = None
        self._show_state("Training failed.")
        self._set_button_state()
        self.failed.emit(tb)

    def fit_now(self):
        self._start_fit("end")

    def toggle_play(self):
        if self._playing:
            self._pause()
            self._refresh(fine=True)
            return
        if self._after_fit is not None:
            return
        if self.run is None:
            self._start_fit("play")
            return
        if self.run.finished:
            if self.run.replayable:
                self.run.reset()
            else:
                self.run = None
                self._start_fit("play")
                return
        self._playing = True
        self.timer.start()
        self._set_button_state()

    def _pause(self):
        self._playing = False
        self.timer.stop()
        self._set_button_state()

    def step_once(self):
        if self._after_fit is not None:
            return
        self._pause()
        if self.run is None or (self.run.finished and not self.run.replayable):
            self._start_fit("step")
            return
        if self.run.finished:
            self.run.reset()
        self.run.step()
        self._refresh(fine=True)

    def reset_run(self):
        self._cancel_run()
        if self.run is not None and self.run.replayable:
            self.run.reset()
            self._refresh(fine=True)
        else:
            self.run = None
            self.plot.update_run(None)
            self.metrics.clear("Reset. Press Train to train.")
            self.details.clear()
            self._show_state("Reset. Press Train to train.")
        self._set_button_state()

    def _rate(self) -> float:
        n = self.run.n_stages
        return self.speed.value() * max(1.0, n / 40.0)

    def _tick(self):
        r = self.run
        if r is None or not self._playing:
            self.timer.stop()
            return
        self._acc += self._rate() * TICK_MS / 1000.0
        n = int(self._acc)
        if n < 1:
            return
        self._acc -= n
        more = True
        try:
            for _ in range(n):
                more = r.step()
                if not more:
                    break
        except Exception:
            import traceback
            self._pause()
            self.failed.emit(traceback.format_exc())
            return
        self._ticks += 1
        if more:
            self._refresh(fine=False, full=(self._ticks % 3 == 0))
        else:
            self._pause()
            self._refresh(fine=True)

    # ---- drawing ---------------------------------------------------------
    def _refresh(self, fine: bool, full: bool = True):
        r = self.run
        if r is None:
            return
        if r.stage == 0:
            self.plot.update_run(None)
            self.metrics.clear("Stage 0. Press Train or Step.")
            self._show_state("Press Train or Step")
            return
        ev = evaluate(r)
        self.plot.update_run(r, ev, fine=fine)
        self.metrics.update_run(r, ev, self.data, full=full or fine)
        if fine:
            self.details.refresh(r, self.data)
        n = r.n_stages
        self._show_state(f"Step {r.stage} of {n}" if n > 1 else "Fitted")
        self._set_button_state()

    def _show_state(self, text: str):
        self.state_label.setText(text)

    def _set_button_state(self):
        it = self.spec.iterative(self.problem)
        fitting = self._after_fit is not None
        self.btn_play.setVisible(it)
        self.btn_step.setVisible(it)
        self.btn_reset.setVisible(it)
        self.speed.setVisible(it)
        self.speed_label.setVisible(it)
        self.btn_fit.setVisible(not it)
        self.btn_play.setText("Pause" if self._playing else "Train")
        for b in (self.btn_play, self.btn_step, self.btn_fit):
            b.setEnabled(not fitting)

    def _knn_hover(self, pt):
        if self.run is not None and self.run.stage > 0:
            self.plot.set_neighbors(pt, self.run.neighbors(pt))

    def export_png(self):
        path, _ = QFileDialog.getSaveFileName(self, "Export PNG", f"{self.spec.id}.png", "PNG image (*.png)")
        if path:
            path = fix_suffix(path, ".png", IMAGE_SUFFIXES)
            if safe_save(self, path, self.plot.export_png):
                self.message.emit(f"Saved {path}")
