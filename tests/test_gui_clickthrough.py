"""Click-through: find every control in the window by itself and operate it with real mouse events.

The other GUI sweeps call model-tab methods directly. This file does what a person does: it walks the widget
tree of every tab, clicks every button and checkbox, steps every combo box through all its entries, moves every
slider and spin box to its extremes, switches every sub-tab, and then lists any control that was never touched.
Blocking dialogs are replaced by non-modal ones and file dialogs return temporary files.
"""
import json

import pytest
import shiboken6
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QAbstractButton, QAbstractSlider, QAbstractSpinBox, QComboBox, QDialog, QFileDialog,
                               QMessageBox, QTabBar, QToolButton, QWidget)

from ml_model_dojo.data.datasets import DataConfig
from ml_model_dojo.models import MODEL_ORDER

pytestmark = pytest.mark.slow

DATASETS = [("classification", "moons"), ("regression", "sine"), ("regression", "inter")]
KINDS = (QAbstractButton, QComboBox, QAbstractSlider, QAbstractSpinBox, QTabBar)


# --------------------------------------------------------------------------- helpers
@pytest.fixture
def no_blocking(win, monkeypatch, tmp_path):
    """Dialogs must never block the test, and file dialogs must return real files."""
    out, state = tmp_path / "out.dat", tmp_path / "state.json"       # an odd extension on purpose: the app must cope
    state.write_text(json.dumps(win.state_dict()))
    monkeypatch.setattr(QDialog, "exec", lambda self: (self.show(), 0)[1])
    monkeypatch.setattr(QMessageBox, "exec", lambda self: (self.show(), 0)[1])
    monkeypatch.setattr(QFileDialog, "getSaveFileName", staticmethod(lambda *a, **k: (str(out), "")))
    monkeypatch.setattr(QFileDialog, "getOpenFileName", staticmethod(lambda *a, **k: (str(state), "")))
    return SimpleNs(out=out, state=state)


class SimpleNs:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def set_data(win, problem, ds, n=80):
    win.data_panel.set_state(DataConfig(problem, ds, n), 1)


def describe(win, w):
    text = w.text() if isinstance(w, QAbstractButton) else ""
    return (win.tabs.currentIndex(), type(w).__name__, text, w.toolTip(), w.accessibleName(), w.objectName())


def controls(win, only_usable=True):
    out = []
    for w in win.findChildren(QWidget):
        if not isinstance(w, KINDS) or isinstance(w, QToolButton):
            continue
        if only_usable and not (w.isVisibleTo(win) and w.isEnabled()):
            continue
        out.append(w)
    return out


def operate(qtbot, win, w):
    dp = win.data_panel
    if isinstance(w, QAbstractButton):
        qtbot.mouseClick(w, Qt.LeftButton)
        if shiboken6.isValid(w) and w.isCheckable():
            qtbot.mouseClick(w, Qt.LeftButton)          # and back again
    elif isinstance(w, QTabBar):
        cur = w.currentIndex()
        for i in range(w.count()):
            w.setCurrentIndex(i)
        w.setCurrentIndex(cur)
    elif isinstance(w, QComboBox):
        cur = w.currentIndex()
        for i in range(w.count()):
            data = w.itemData(i)
            if w is dp.size_box and isinstance(data, int) and data >= 3000:
                continue                                  # huge datasets are covered by the very large data test
            w.setCurrentIndex(i)
        w.setCurrentIndex(cur)
    elif isinstance(w, (QAbstractSlider, QAbstractSpinBox)):
        if w is dp.custom:
            return
        cur = w.value()
        for v in (w.minimum(), w.maximum(), (w.minimum() + w.maximum()) // 2, cur):
            w.setValue(v)
    qtbot.wait(25)


def close_stray_windows(win):
    for d in win.findChildren(QDialog):
        if shiboken6.isValid(d) and d.isVisible():
            d.close()
    tour = getattr(win, "_tour", None)
    if tour is not None and shiboken6.isValid(tour.dlg) and (tour.dlg.isVisible() or tour.frame.isVisible()):
        tour.finish()


def settle_all(qtbot, win, timeout=240000):
    qtbot.waitUntil(lambda: all(t._after_fit is None for t in win.model_tabs.values()) and win.compare.worker is None,
                    timeout=timeout)


def shrink(win):
    """Keep iterative runs short, as a user would not wait for a thousand epochs in a test."""
    for tab in win.model_tabs.values():
        for key, val in (("n_estimators", 20), ("max_epochs", 20), ("epochs_per_step", 5)):
            if key in tab.params.rows and tab.params.rows[key]["spec"].kind == "int":
                tab.params.rows[key]["widget"].setValue(val)


def touch(qtbot, win, w, touched):
    d = describe(win, w)
    touched.add(d)
    operate(qtbot, win, w)
    assert win._errors == [], d


def exercise_current_tab(qtbot, win, touched, passes=2):
    for _ in range(passes):
        done = set()
        for _guard in range(500):
            # Collect again after every action: a click can rebuild a panel (Reset to defaults, + Layer) or open a dialog.
            items = [w for w in controls(win) if describe(win, w) not in done]
            if not items:
                break
            items.sort(key=lambda w: isinstance(w, QAbstractButton))   # settings first, buttons last
            done.add(describe(win, items[0]))
            touch(qtbot, win, items[0], touched)
        win.cancel_all()
        close_stray_windows(win)
        qtbot.wait(50)
        settle_all(qtbot, win)
        assert win._errors == []
    # Some controls are disabled in some states (Step after a run has finished). Go back to the start, then touch the rest.
    for b in [w for w in controls(win) if isinstance(w, QAbstractButton) and w.text() == "Reset"]:
        qtbot.mouseClick(b, Qt.LeftButton)
    for _round in range(5):
        rest = [w for w in controls(win) if describe(win, w) not in touched]
        if not rest:
            break
        touch(qtbot, win, rest[0], touched)
    # Controls on a sub-tab that is not showing are only reachable by visiting that sub-tab.
    for bar in [w for w in controls(win) if isinstance(w, QTabBar)]:
        if not shiboken6.isValid(bar):
            continue
        cur = bar.currentIndex()
        for i in range(bar.count()):
            bar.setCurrentIndex(i)
            qtbot.wait(30)
            for _round in range(60):
                rest = [w for w in controls(win) if not isinstance(w, QTabBar) and describe(win, w) not in touched]
                if not rest:
                    break
                touch(qtbot, win, rest[0], touched)
        if shiboken6.isValid(bar):
            bar.setCurrentIndex(cur)
    win.cancel_all()
    close_stray_windows(win)
    settle_all(qtbot, win)


# --------------------------------------------------------------------------- the click-through
@pytest.mark.parametrize("problem,ds", DATASETS)
def test_click_every_control_on_every_tab(qtbot, win, no_blocking, problem, ds):
    touched = set()
    set_data(win, problem, ds)
    shrink(win)
    qtbot.wait(300)
    for i in range(win.tabs.count()):
        win.tabs.setCurrentIndex(i)
        set_data(win, problem, ds)
        qtbot.wait(100)
        if win.tabs.widget(i) is win.compare:
            qtbot.mouseClick(win.compare.btn, Qt.LeftButton)           # a real compare run, so Edit and Export exist
            qtbot.waitUntil(lambda: win.compare.worker is None and len(win.compare.results) > 0, timeout=400000)
        exercise_current_tab(qtbot, win, touched)
        win.grab()

    # every control that exists in the application, whether or not it was visible, must have been touched
    # unless it is hidden by design.
    missing = set()
    for i in range(win.tabs.count()):
        win.tabs.setCurrentIndex(i)
        for w in controls(win, only_usable=False):
            d = describe(win, w)
            if d not in touched:
                missing.add(d + (w.isVisibleTo(win), w.isEnabled()))
    allowed = [
        lambda d: d[1] == "QPushButton" and d[2] in ("Cancel", "Edit"),     # only shown while busy / after a run
    ]
    unexpected = sorted(d for d in missing if not any(rule(d) for rule in allowed))
    # hidden by design: the alternate Train button (iterative or not), and parameter rows that do not apply
    unexpected = [d for d in unexpected if d[-2] is True]
    assert unexpected == [], "controls that were visible but never touched:\n" + "\n".join(map(str, unexpected))
    assert len(touched) > 150, len(touched)


# --------------------------------------------------------------------------- shortcuts and the menu
def test_keyboard_shortcuts_on_every_tab(qtbot, win):
    set_data(win, "classification", "moons")
    shrink(win)
    qtbot.wait(300)
    for mid in MODEL_ORDER:
        tab = win.model_tabs[mid]
        win.tabs.setCurrentWidget(tab)
        tab.btn_step.setFocus()
        qtbot.keyClick(tab.btn_step, Qt.Key_F6)
        qtbot.waitUntil(lambda: tab._after_fit is None and tab.run is not None and tab.run.stage >= 1, timeout=90000)
        qtbot.keyClick(tab.btn_step, Qt.Key_F5)                      # play or pause
        qtbot.wait(150)
        qtbot.keyClick(tab.btn_step, Qt.Key_F7)                      # reset
        qtbot.wait(100)
        win.cancel_all()
    settle_all(qtbot, win)
    assert win._errors == []


def test_saving_and_exporting_survive_odd_names_and_bad_folders(qtbot, win, monkeypatch, tmp_path):
    """A mistyped extension, a missing folder or an unwritable path gives a plain message, never a traceback."""
    from ml_model_dojo.ui import dialogs
    shown = []
    monkeypatch.setattr(dialogs, "show_error", lambda parent, message, details="": shown.append(message))
    set_data(win, "classification", "moons")
    tab = win.model_tabs["tree"]
    win.tabs.setCurrentWidget(tab)
    tab.step_once() if tab.spec.iterative(tab.problem) else tab.fit_now()
    qtbot.waitUntil(lambda: tab._after_fit is None and tab.run is not None and tab.run.stage >= 1, timeout=90000)
    c = win.compare
    c.set_selection(["tree", "knn"])
    c.run_compare()
    qtbot.waitUntil(lambda: c.worker is None and len(c.results) == 2, timeout=300000)

    def answer(path):
        monkeypatch.setattr(QFileDialog, "getSaveFileName", staticmethod(lambda *a, **k: (str(path), "")))

    a_file = tmp_path / "blocker"
    a_file.write_text("x")
    for exporter, name, expected in (
        (tab.export_png, "plain", "plain.png"), (tab.export_png, "odd.xyz", "odd.xyz.png"), (tab.export_png, "UP.PNG", "UP.PNG"),
        (c.export_png, "panels", "panels.png"), (c.export_csv, "table", "table.csv"), (c.export_csv, "t.dat", "t.dat"),
        (win.save_state, "state", "state.json"),
    ):
        answer(tmp_path / name)
        exporter()
        assert (tmp_path / expected).exists() and (tmp_path / expected).stat().st_size > 20, (name, expected)
    assert shown == []

    bad = [tmp_path / "no_such_folder" / "x.png", a_file / "inside.png"]           # missing folder, parent is a file
    for exporter in (tab.export_png, c.export_png, c.export_csv, win.save_state):
        for path in bad:
            before = len(shown)
            answer(path)
            exporter()
            assert len(shown) == before + 1, (exporter.__name__, path)
    assert win._errors == []


def test_forest_tree_toggle_without_a_trained_model(qtbot, win):
    """Regression: ticking 'Show the first six individual trees' crashed when no forest was trained."""
    set_data(win, "classification", "moons")
    tab = win.model_tabs["forest"]
    win.tabs.setCurrentWidget(tab)
    toggle = tab.details.toggle
    toggle.setChecked(True)                      # before any training
    toggle.setChecked(False)
    tab.params.rows["n_estimators"]["widget"].setValue(10)
    qtbot.wait(250)
    tab.step_once()
    qtbot.waitUntil(lambda: tab._after_fit is None and tab.run is not None and tab.run.stage >= 1, timeout=90000)
    toggle.setChecked(True)                      # after training
    set_data(win, "classification", "xor")       # the data change clears the run
    qtbot.wait(300)
    toggle.setChecked(False)
    toggle.setChecked(True)                      # after the run was cleared
    assert win._errors == []


def test_every_scenario_menu_entry_opens_the_compare_tab(qtbot, win, no_blocking):
    menu = win.btn_scen.menu()
    assert len(menu.actions()) >= 11
    for act in menu.actions():
        win.tabs.setCurrentIndex(0)
        act.trigger()
        assert win.tabs.currentWidget() is win.compare, act.text()
        assert win.compare.scenario.isVisible(), act.text()
        qtbot.wait(150)
        win.cancel_all()
        settle_all(qtbot, win)
    assert win._errors == []
