"""Main window: header, shared data panel, model tabs, Compare tab, status bar."""
from __future__ import annotations

import json
import logging
import time
import traceback
from pathlib import Path

from PySide6.QtCore import QSettings, QTimer
from PySide6.QtWidgets import (QFileDialog, QHBoxLayout, QLabel, QMainWindow, QMenu, QPushButton,
                               QTabWidget, QToolButton, QVBoxLayout, QWidget)

from ..data.datasets import DataConfig, Dataset, generate
from ..models import MODEL_ORDER, get_spec
from ..scenarios.definitions import SCENARIOS
from .compare_tab import CompareTab
from .data_panel import DataPanel
from .dialogs import HelpDialog, TourController, fix_suffix, safe_save, show_error
from .model_tab import ModelTab

log = logging.getLogger("msp")
STATE_VERSION = 1


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ML Model Dojo")
        self.setMinimumSize(980, 660)
        self.resize(1360, 840)
        self.dataset: Dataset | None = None
        self._busy: dict[str, str] = {}
        self._t0 = 0.0
        self.settings = QSettings("MLModelDojo", "MLModelDojo")

        central = QWidget()
        self.setCentralWidget(central)
        v = QVBoxLayout(central)
        v.setContentsMargins(6, 4, 6, 4)
        v.addLayout(self._build_header())
        self.data_panel = DataPanel()
        v.addWidget(self.data_panel)
        self.tabs = QTabWidget()
        self.model_tabs: dict[str, ModelTab] = {}
        for mid in MODEL_ORDER:
            tab = ModelTab(get_spec(mid))
            self.model_tabs[mid] = tab
            self.tabs.addTab(tab, tab.spec.display_name)
            tab.message.connect(self._hover)
            tab.failed.connect(self.report_error)
            tab.busy.connect(lambda text, k=mid: self._set_busy(k, text))
            tab.idle.connect(lambda k=mid: self._set_busy(k, None))
        self.compare = CompareTab(self.model_tabs, self)
        self.tabs.addTab(self.compare, "Compare")
        self.compare.busy.connect(lambda text: self._set_busy("compare", text))
        self.compare.idle.connect(lambda: self._set_busy("compare", None))
        self.compare.failed.connect(self.report_error)
        self.compare.jump_to.connect(self.show_model_tab)
        v.addWidget(self.tabs, 1)
        self._build_status_bar()
        self.data_panel.changed.connect(self.regenerate)

        self._restore()
        if self.dataset is None:
            self.regenerate(self.data_panel.config(), self.data_panel.seed_value())

    # ---- header and status ----------------------------------------------
    def _build_header(self):
        h = QHBoxLayout()
        box = QVBoxLayout()
        box.setSpacing(0)
        title = QLabel("ML Model Dojo")
        title.setObjectName("title")
        sub = QLabel("An interactive lab for choosing machine learning models")
        sub.setObjectName("subtitle")
        box.addWidget(title)
        box.addWidget(sub)
        h.addLayout(box)
        h.addStretch()
        self.btn_scen = QToolButton()
        self.btn_scen.setText("Scenarios")
        self.btn_scen.setToolTip("Load a guided lesson: it sets the data and models and opens the Compare tab.")
        self.btn_scen.setPopupMode(QToolButton.InstantPopup)
        menu = QMenu(self.btn_scen)
        for i, sc in enumerate(SCENARIOS):
            act = menu.addAction(f"{i + 1}. {sc['title']}")
            act.triggered.connect(lambda _=False, s=sc: self.load_scenario(s))
        self.btn_scen.setMenu(menu)
        h.addWidget(self.btn_scen)
        self.btn_save = QPushButton("Save")
        self.btn_save.setToolTip("Save the data settings, hyperparameters and comparison choices to a JSON file.")
        self.btn_save.clicked.connect(self.save_state)
        self.btn_load = QPushButton("Load")
        self.btn_load.setToolTip("Load settings from a JSON file saved earlier.")
        self.btn_load.clicked.connect(self.load_state)
        self.btn_help = QPushButton("Help")
        self.btn_help.setToolTip("Open the help window, the glossary and the guided tour.")
        self.btn_help.clicked.connect(self.show_help)
        for b in (self.btn_save, self.btn_load, self.btn_help):
            h.addWidget(b)
        return h

    def _build_status_bar(self):
        sb = self.statusBar()
        self.status_msg = QLabel("Ready")
        self.hover = QLabel("")
        self.elapsed = QLabel("")
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setToolTip("Stop the current long-running task.")
        self.cancel_btn.setVisible(False)
        self.cancel_btn.clicked.connect(self.cancel_all)
        sb.addWidget(self.status_msg)
        sb.addWidget(self.hover, 1)
        sb.addPermanentWidget(self.elapsed)
        sb.addPermanentWidget(self.cancel_btn)
        self.clock = QTimer(self)
        self.clock.setInterval(200)
        self.clock.timeout.connect(self._tick_clock)

    def _hover(self, text: str):
        self.hover.setText(text)

    def _set_busy(self, key: str, text: str | None):
        if text is None:
            self._busy.pop(key, None)
        else:
            if not self._busy:
                self._t0 = time.perf_counter()
            self._busy[key] = text
        if self._busy:
            self.status_msg.setText(next(iter(self._busy.values())))
            self.cancel_btn.setVisible(True)
            self.clock.start()
        else:
            self.status_msg.setText("Ready")
            self.cancel_btn.setVisible(False)
            self.clock.stop()
            self.elapsed.setText("")

    def _tick_clock(self):
        self.elapsed.setText(f"Elapsed {time.perf_counter() - self._t0:.1f} s")

    def cancel_all(self):
        for t in self.model_tabs.values():
            t.cancel()
        self.compare.cancel()
        self._busy.clear()
        self._set_busy("none", None)

    def report_error(self, tb: str):
        log.error("Reported error:\n%s", tb)
        last = tb.strip().splitlines()[-1] if tb.strip() else "Unknown error"
        show_error(self, last, tb)

    # ---- data ------------------------------------------------------------
    def regenerate(self, cfg: DataConfig, seed: int):
        try:
            ds = generate(cfg, seed)
        except Exception:
            self.report_error(traceback.format_exc())
            return
        self.dataset = ds
        self.data_panel.show_thumbnail(ds)
        for tab in self.model_tabs.values():
            tab.set_dataset(ds)
        self.compare.set_dataset(ds)
        self.status_msg.setText("Ready")
        self.hover.setText("Data regenerated. All trained models were cleared, so retrain them.")

    def show_model_tab(self, model_id: str):
        self.tabs.setCurrentWidget(self.model_tabs[model_id])

    # ---- save, load, scenarios ------------------------------------------
    def state_dict(self) -> dict:
        cfg = self.data_panel.config()
        return {"version": STATE_VERSION, "data": {**cfg.to_dict(), "seed": self.data_panel.seed_value()},
                "models": {mid: t.all_params() for mid, t in self.model_tabs.items()},
                "compare": {"selected": self.compare.selection(), **self.compare.options()}}

    def apply_state(self, st: dict, regenerate: bool = True):
        d = dict(st.get("data", {}))
        seed = int(d.pop("seed", 1))
        cfg = DataConfig.from_dict(d)
        for mid, by_problem in st.get("models", {}).items():
            if mid in self.model_tabs:
                self.model_tabs[mid].saved.update({p: get_spec(mid).clean_params(v, p) for p, v in by_problem.items()
                                                   if p in get_spec(mid).supports})
        self.data_panel.set_state(cfg, seed, emit=False)
        # Force the schemas to rebuild so that the sliders show the loaded values.
        for t in self.model_tabs.values():
            t._schema_key = None
        self.regenerate(cfg, seed)
        cmp_state = st.get("compare", {})
        self.compare.set_selection(cmp_state.get("selected", []))
        self.compare.set_options({k: v for k, v in cmp_state.items() if k != "selected"})

    def save_state(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save settings", "playground_settings.json", "JSON (*.json)")
        if path:
            path = fix_suffix(path, ".json")
            text = json.dumps(self.state_dict(), indent=2)
            if safe_save(self, path, lambda p: Path(p).write_text(text, encoding="utf-8")):
                self.hover.setText(f"Saved {path}")

    def load_state(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load settings", "", "JSON (*.json)")
        if not path:
            return
        try:
            self.apply_state(json.loads(Path(path).read_text(encoding="utf-8")))
            self.hover.setText(f"Loaded {path}")
        except Exception:
            self.report_error(traceback.format_exc())

    def load_scenario(self, sc: dict):
        cfg = DataConfig.from_dict(sc["data"])
        st = {"version": STATE_VERSION, "data": {**cfg.to_dict(), "seed": sc["seed"]}, "models": {},
              "compare": {"selected": sc["models"], **sc.get("compare", {})}}
        for mid, params in sc.get("params", {}).items():
            st["models"][mid] = {cfg.problem: params}
        self.apply_state(st)
        self.tabs.setCurrentWidget(self.compare)
        self.compare.show_scenario(sc)
        self.compare.run_compare()

    # ---- help ------------------------------------------------------------
    def show_help(self):
        HelpDialog(self, self.start_tour).exec()

    def start_tour(self):
        cur = lambda: self.tabs.currentWidget()
        steps = [
            (lambda: self.data_panel, "Data panel", "Choose the problem, dataset, size, noise and seed here. Changing the data clears all trained models."),
            (lambda: self.tabs.tabBar(), "Model tabs", "Each tab is one model family. The Compare tab runs several models on the same data."),
            (lambda: getattr(cur(), "params", None), "Hyperparameters", "Sliders and menus change the model. Hover over any control for a one-sentence explanation."),
            (lambda: getattr(cur(), "plot", None), "Main plot", "Filled markers are training points, hollow markers are test points, and red rings are test errors. The color shows the prediction."),
            (lambda: getattr(cur(), "metrics", None), "Metrics", "Train and test scores are always side by side. A large gap means overfitting."),
            (lambda: self.compare, "Compare tab", "Add models, run cross-validation or repeat across seeds, and read the trade-off table."),
            (lambda: self.btn_scen, "Scenarios", "Guided lessons load a ready-made comparison with a note about what to look for."),
        ]
        self._tour = TourController(self, steps)
        self._tour.start()

    # ---- persistence -------------------------------------------------------
    def _restore(self):
        try:
            geo = self.settings.value("geometry")
            if geo:
                self.restoreGeometry(geo)
            raw = self.settings.value("state")
            if raw:
                self.apply_state(json.loads(raw))
        except Exception:
            log.exception("Could not restore settings")

    def closeEvent(self, ev):
        try:
            self.cancel_all()
            self.compare.shutdown()
            self.settings.setValue("geometry", self.saveGeometry())
            self.settings.setValue("state", json.dumps(self.state_dict()))
        except Exception:
            log.exception("Could not save settings")
        super().closeEvent(ev)
