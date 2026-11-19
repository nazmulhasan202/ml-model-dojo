"""GUI-driven sweeps: click every control on every tab, run action sequences, and exercise every dialog.

The ``win`` fixture fails a test if anything is reported to the user as an error, and pytest-qt fails a
test if an exception escapes from a Qt slot.
"""
import json
from types import SimpleNamespace

import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QFileDialog, QPushButton

from ml_model_dojo.data.datasets import CLASSIFICATION, REGRESSION, DataConfig
from ml_model_dojo.models import MODEL_ORDER
from ml_model_dojo.scenarios.definitions import SCENARIOS

pytestmark = pytest.mark.slow

DATASETS = [("classification", "moons"), ("regression", "sine"), ("regression", "inter")]


# --------------------------------------------------------------------------- helpers
def set_data(win, problem, ds, n=100, seed=1, **kw):
    win.data_panel.set_state(DataConfig(problem, ds, n, **kw), seed)


def settle(qtbot, tab, timeout=90000):
    qtbot.waitUntil(lambda: tab._after_fit is None, timeout=timeout)


def train_quick(qtbot, win, tab):
    """One step for iterative models, a fit for the others, then wait until the plot shows the model."""
    win.tabs.setCurrentWidget(tab)
    if tab.spec.iterative(tab.problem):
        tab.step_once()
    else:
        tab.fit_now()
    qtbot.waitUntil(lambda: tab._after_fit is None and tab.run is not None and tab.run.stage >= 1, timeout=90000)
    qtbot.wait(30)
    assert win._errors == []


def buttons(widget, text):
    return [b for b in widget.findChildren(QPushButton) if b.text() == text]


def probes(info):
    """A list of callables that put one control through its extremes."""
    s, w = info["spec"], info["widget"]
    if s.kind in ("int", "float", "log"):
        lo, hi = w.minimum(), w.maximum()
        return [lambda v=v: w.setValue(v) for v in (lo, hi, (lo + hi) // 2, lo)]
    if s.kind == "choice":
        return [lambda i=i: w.setCurrentIndex(i) for i in range(w.count())]
    if s.kind == "bool":
        return [lambda: w.setChecked(not w.isChecked()), lambda: w.setChecked(not w.isChecked())]
    if s.kind == "layers":
        def many(text, times):
            def f():
                for _ in range(times):
                    for b in buttons(w, text):
                        b.click()
            return f
        return [many("+ Layer", 6), many("+", 20), many("-", 20), many("- Layer", 6), many("+ Layer", 2)]
    if s.kind == "flags":
        boxes = info["boxes"]
        def only(keys):
            def f():
                for k, cb in boxes.items():
                    cb.setChecked(k in keys)
            return f
        keys = list(boxes)
        return [only(keys), only([]), only(keys[:1]), only(keys[-1:]), only(keys[:2])]
    raise AssertionError(s.kind)


# --------------------------------------------------------------------------- every control on every tab
@pytest.mark.parametrize("model_id", MODEL_ORDER)
@pytest.mark.parametrize("problem,ds", DATASETS)
def test_every_control_on_every_tab(qtbot, win, model_id, problem, ds):
    set_data(win, problem, ds, 80)
    tab = win.model_tabs[model_id]
    win.tabs.setCurrentWidget(tab)
    train_quick(qtbot, win, tab)
    for key, info in list(tab.params.rows.items()):
        for probe in probes(info):
            probe()
            qtbot.wait(190)               # ParamPanel debounces for 150 ms
            assert win._errors == [], (model_id, key)
            train_quick(qtbot, win, tab)
            assert tab.run is not None and tab.run.stage >= 1, (model_id, key)
        tab.window().grab()               # painting must not fail either
    tab.reset_defaults()
    qtbot.wait(250)
    defaults = tab.spec.default_params(tab.problem)
    shown = tab.params.values()
    for k, v in defaults.items():
        if k == "features":
            continue
        assert shown[k] == pytest.approx(v, rel=0.02, abs=0.02) if isinstance(v, float) else shown[k] == v, (model_id, k)
    train_quick(qtbot, win, tab)


# --------------------------------------------------------------------------- action sequences
@pytest.mark.parametrize("model_id", MODEL_ORDER)
@pytest.mark.parametrize("problem,ds", DATASETS)
def test_action_sequences(qtbot, win, monkeypatch, tmp_path, model_id, problem, ds):
    set_data(win, problem, ds, 80)
    tab = win.model_tabs[model_id]
    win.tabs.setCurrentWidget(tab)
    # keep iterative runs short
    for key, val in (("n_estimators", 20), ("max_epochs", 20), ("epochs_per_step", 5)):
        if key in tab.params.rows and tab.params.rows[key]["spec"].kind == "int":
            tab.params.rows[key]["widget"].setValue(val)
    qtbot.wait(250)

    # pause and step and reset before anything is trained
    tab.reset_run(); tab.step_once(); settle(qtbot, tab); tab.reset_run()
    tab.toggle_play(); settle(qtbot, tab)
    qtbot.wait(250)
    tab.toggle_play() if tab._playing else None      # pause mid-way
    tab.step_once(); tab.step_once()
    tab.reset_run()
    tab.toggle_play(); settle(qtbot, tab)
    qtbot.waitUntil(lambda: tab.run is not None and tab.run.finished and not tab._playing, timeout=120000)
    tab.toggle_play(); settle(qtbot, tab)            # replay after completion
    qtbot.waitUntil(lambda: tab.run is not None and tab.run.finished and not tab._playing, timeout=120000)

    # hover and click on the plot, with and without a model
    for x, y in ((0.0, 0.0), (-1.0, 1.0), (0.97, -0.95)):
        tab.plot._on_move(SimpleNamespace(inaxes=tab.plot.ax, xdata=x, ydata=y))
        tab.plot._on_click(SimpleNamespace(inaxes=tab.plot.ax, xdata=x, ydata=y))
    tab.plot._on_move(SimpleNamespace(inaxes=None, xdata=None, ydata=None))
    for flag in ("train", "test", "errors", "truth"):
        tab.plot.set_flag(flag, False)
        tab.plot.set_flag(flag, True)
    # speed slider extremes
    tab.speed.setValue(1); tab.speed.setValue(10)
    # export
    out = tmp_path / f"{model_id}.png"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", staticmethod(lambda *a, **k: (str(out), "")))
    tab.export_png()
    assert out.exists() and out.stat().st_size > 1000
    # the metrics panel: switch the curve mode and open the glossary
    tab.metrics.mode.setCurrentIndex(1); tab.metrics.mode.setCurrentIndex(0)
    # change data and settings while a run is in flight
    tab.reset_run()
    tab.toggle_play()
    qtbot.wait(120)
    set_data(win, "classification" if problem == "regression" else "regression",
             "xor" if problem == "regression" else "lin", 60, seed=2)
    qtbot.wait(300)
    tab.toggle_play()
    qtbot.wait(150)
    tab.cancel()
    tab.step_once(); settle(qtbot, tab)
    assert win._errors == []


def test_changing_settings_and_data_while_training(qtbot, win):
    set_data(win, "classification", "spirals", 300)
    tab = win.model_tabs["mlp"]
    win.tabs.setCurrentWidget(tab)
    tab.params.rows["max_epochs"]["widget"].setValue(300)
    qtbot.wait(250)
    tab.toggle_play()
    qtbot.wait(300)
    tab.params.rows["lr"]["widget"].setValue(10)      # a different learning rate mid-run
    qtbot.wait(300)
    tab.toggle_play()
    qtbot.wait(300)
    set_data(win, "classification", "rings", 150, seed=3)
    qtbot.wait(300)
    tab.toggle_play()
    qtbot.wait(300)
    win.cancel_all()
    qtbot.wait(300)
    assert win._errors == []


# --------------------------------------------------------------------------- the data panel
def test_data_panel_every_dataset_trains_on_every_tab(qtbot, win):
    for problem, reg in (("classification", CLASSIFICATION), ("regression", REGRESSION)):
        for ds in reg:
            set_data(win, problem, ds, 40)
            for mid in MODEL_ORDER:
                train_quick(qtbot, win, win.model_tabs[mid])


def test_data_panel_widgets_extremes(qtbot, win):
    dp = win.data_panel
    for problem_idx in range(dp.problem.count()):
        dp.problem.setCurrentIndex(problem_idx)
        for ds_idx in range(dp.dataset.count()):
            dp.dataset.setCurrentIndex(ds_idx)
            for slider in (dp.noise, dp.label_noise, dp.balance, dp.split):
                slider.slider.setValue(slider.slider.minimum())
                dp._emit()
                slider.slider.setValue(slider.slider.maximum())
                dp._emit()
            for i in range(dp.seed.count()):
                dp.seed.setCurrentIndex(i)
                dp._emit()
            for i in range(dp.mode.count()):
                dp.mode.setCurrentIndex(i)
                dp._emit()
            for i in range(dp.size_box.count() - 1):
                if dp.size_box.itemData(i) >= 3000 and ds_idx % 6:
                    continue
                dp.size_box.setCurrentIndex(i)
                dp._emit()
        for tab in win.model_tabs.values():
            assert tab.data is win.dataset
    dp.size_box.setCurrentIndex(dp.size_box.count() - 1)        # custom size
    for v in (dp.custom.minimum(), 777, dp.custom.maximum()):
        dp.custom.setValue(v)
        dp._emit()
        assert len(win.dataset.y) == v
    dp.regen.click()
    assert win._errors == []


@pytest.mark.parametrize("problem,ds", [("classification", "spirals"), ("regression", "inter"), ("regression", "sine")])
def test_very_large_data_on_every_tab(qtbot, win, problem, ds):
    set_data(win, problem, ds, 15000)
    for mid in MODEL_ORDER:
        tab = win.model_tabs[mid]
        for key in ("n_estimators", "max_epochs"):
            if key in tab.params.rows:
                tab.params.rows[key]["widget"].setValue(10 if key == "max_epochs" else 20)
        qtbot.wait(250)
        train_quick(qtbot, win, tab)


# --------------------------------------------------------------------------- compare
@pytest.mark.parametrize("problem,ds", DATASETS)
def test_compare_all_models_everything_on(qtbot, win, monkeypatch, tmp_path, problem, ds):
    set_data(win, problem, ds, 120)
    c = win.compare
    for mid in ("forest", "boosting"):
        win.model_tabs[mid].params.rows["n_estimators"]["widget"].setValue(30)
    win.model_tabs["mlp"].params.rows["max_epochs"]["widget"].setValue(30)
    qtbot.wait(300)
    c.set_selection(MODEL_ORDER)
    c.cv.setChecked(True)
    c.multiseed.setChecked(True)
    c.run_compare()
    qtbot.waitUntil(lambda: c.worker is None and len(c.results) == len(MODEL_ORDER), timeout=600000)
    for i in range(c.metric_box.count()):
        c.metric_box.setCurrentIndex(i)
    for tab_index in range(c.tabs.count()):
        c.tabs.setCurrentIndex(tab_index)
        win.grab()
    # linked highlighting: click near several data points in a panel
    ds_ = win.dataset
    for i in (0, len(ds_.y) // 2, len(ds_.y) - 1):
        x, y = ds_.X[i, 0], (ds_.y[i] if ds_.n_inputs == 1 else ds_.X[i, 1])
        c._on_panel_click(SimpleNamespace(inaxes=c.panel_axes[0], xdata=float(x), ydata=float(y)))
    c._on_panel_click(SimpleNamespace(inaxes=None, xdata=None, ydata=None))
    # exports
    png, csv_path = tmp_path / "p.png", tmp_path / "r.csv"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", staticmethod(lambda *a, **k: (str(png), "")))
    c.export_png()
    monkeypatch.setattr(QFileDialog, "getSaveFileName", staticmethod(lambda *a, **k: (str(csv_path), "")))
    c.export_csv()
    assert png.exists() and csv_path.exists()
    assert len(csv_path.read_text().strip().splitlines()) == 1 + len(MODEL_ORDER)
    # tables sort without error
    for view in (c.table, c.trade):
        for col in range(view.model().columnCount()):
            view.sortByColumn(col, Qt.DescendingOrder)
    # the stale logic and cached rerun
    win.model_tabs["tree"].params.rows["max_depth"]["widget"].setValue(2)
    qtbot.wait(300)
    assert c._is_stale("tree") == "Stale"
    c.run_compare()
    qtbot.waitUntil(lambda: c.worker is None, timeout=300000)
    assert c._is_stale("tree") == ""


@pytest.mark.parametrize("n_models", [1, 2, 3, 5])
def test_compare_with_subsets_and_edit_links(qtbot, win, n_models):
    set_data(win, "classification", "xor", 100)
    c = win.compare
    ids = ["linear", "knn", "tree", "svm", "forest"][:n_models]
    c.set_selection(ids)
    c.run_compare()
    qtbot.waitUntil(lambda: c.worker is None and len(c.results) == n_models, timeout=300000)
    for b in buttons(c, "Edit"):
        b.click()
    # empty selection is handled
    c.set_selection([])
    c.run_compare()
    assert c.worker is None


def test_compare_cancel_and_close_while_running(qtbot, win):
    set_data(win, "classification", "spirals", 600)
    c = win.compare
    win.model_tabs["mlp"].params.rows["max_epochs"]["widget"].setValue(500)
    qtbot.wait(250)
    c.set_selection(MODEL_ORDER)
    c.cv.setChecked(True)
    c.run_compare()
    qtbot.wait(500)
    win.cancel_all()
    qtbot.waitUntil(lambda: c.worker is None, timeout=60000)
    # the pool is rebuilt and a new run works
    c.set_selection(["tree", "knn"])
    c.cv.setChecked(False)
    c.run_compare()
    qtbot.waitUntil(lambda: c.worker is None and len(c.results) == 2, timeout=300000)
    # closing the window while a run is in flight
    c.set_selection(MODEL_ORDER)
    c.cv.setChecked(True)
    c.run_compare()
    qtbot.wait(300)
    win.close()


# --------------------------------------------------------------------------- scenarios, files, dialogs
@pytest.mark.parametrize("sc", SCENARIOS, ids=[s["id"] for s in SCENARIOS])
def test_every_scenario_and_its_followups(qtbot, win, sc):
    win.load_scenario(sc)
    c = win.compare
    qtbot.waitUntil(lambda: c.worker is None and len(c.results) == len(sc["models"]), timeout=600000)
    for fu in sc.get("followups", []):
        if "data" in fu:
            cfg = win.data_panel.config().with_(**fu["data"])
            win.data_panel.set_state(cfg, win.data_panel.seed_value())
        for mid, params in fu.get("params", {}).items():
            tab = win.model_tabs[mid]
            tab.apply_params({sc["data"]["problem"]: {**tab.current_params(sc["data"]["problem"]), **params}})
        qtbot.wait(300)
        c.run_compare()
        qtbot.waitUntil(lambda: c.worker is None and len(c.results) == len(sc["models"]), timeout=600000)
    assert c.scenario.isVisible()


def test_save_and_load_round_trip_with_dialogs(qtbot, win, monkeypatch, tmp_path):
    path = tmp_path / "state.json"
    set_data(win, "regression", "inter", 200, noise=0.4)
    win.model_tabs["mlp"].apply_params({"regression": {"layers": [3, 5, 7], "features": ["x1x2"], "max_epochs": 50}})
    win.compare.set_selection(["mlp", "tree"])
    win.compare.cv.setChecked(True)
    monkeypatch.setattr(QFileDialog, "getSaveFileName", staticmethod(lambda *a, **k: (str(path), "")))
    win.save_state()
    saved = json.loads(path.read_text())
    set_data(win, "classification", "xor", 150)
    monkeypatch.setattr(QFileDialog, "getOpenFileName", staticmethod(lambda *a, **k: (str(path), "")))
    win.load_state()
    assert win.dataset.config.dataset == "inter" and win.dataset.config.noise == pytest.approx(0.4)
    assert win.model_tabs["mlp"].params.values()["layers"] == [3, 5, 7]
    assert win.compare.selection() == ["tree", "mlp"] and win.compare.cv.isChecked()
    assert win.state_dict()["models"]["mlp"]["regression"]["layers"] == saved["models"]["mlp"]["regression"]["layers"]
    train_quick(qtbot, win, win.model_tabs["mlp"])


@pytest.mark.parametrize("payload", [
    "not json at all", "[]", "{}", '{"data": {"dataset": "does_not_exist"}}',
    '{"data": {"problem": "regression", "dataset": "inter", "n_samples": 99999999, "noise": 7, "seed": 77}}',
    '{"models": {"mlp": {"regression": {"layers": [], "features": [], "max_epochs": -5}}, "nonsense": {}}}',
    '{"models": {"knn": {"classification": {"k": 1000, "metric": "weird"}}}, "compare": {"selected": ["x", "tree"]}}',
])
def test_loading_damaged_or_odd_files_never_crashes(qtbot, win, monkeypatch, tmp_path, payload):
    path = tmp_path / "bad.json"
    path.write_text(payload)
    monkeypatch.setattr(QFileDialog, "getOpenFileName", staticmethod(lambda *a, **k: (str(path), "")))
    win.load_state()
    win._errors.clear()                 # an error message is acceptable, an unhandled exception is not
    # and the application still works afterwards
    set_data(win, "classification", "moons", 80)
    train_quick(qtbot, win, win.model_tabs["tree"])


def test_help_glossary_tour_and_scenario_menu(qtbot, win):
    from ml_model_dojo.ui.dialogs import GlossaryDialog, HelpDialog
    g = GlossaryDialog(win)
    qtbot.addWidget(g)
    g.show()
    h = HelpDialog(win, lambda: None)
    qtbot.addWidget(h)
    h.show()
    for i in range(win.tabs.count()):
        win.tabs.setCurrentIndex(i)
        win.start_tour()
        t = win._tour
        for _ in range(len(t.steps) + 1):
            t.go(+1)
        for _ in range(len(t.steps) + 1):
            t.go(-1)
        t.finish()
    for act in win.btn_scen.menu().actions():
        assert act.text()
    for tab in win.model_tabs.values():
        tab.metrics.findChildren(QPushButton)[-1].text()
    assert win._errors == []


@pytest.mark.parametrize("size", [(1024, 768), (980, 660), (1920, 1080)])
def test_window_sizes_render_every_tab(qtbot, win, size):
    win.resize(*size)
    set_data(win, "classification", "moons", 100)
    for tab in win.model_tabs.values():
        train_quick(qtbot, win, tab)
    for i in range(win.tabs.count()):
        win.tabs.setCurrentIndex(i)
        pix = win.grab()
        assert pix.width() > 0
    win.close()
