import pytest

pytest.importorskip("PySide6")

from ml_model_dojo.data.datasets import DataConfig
from ml_model_dojo.models import MODEL_ORDER, get_spec
from ml_model_dojo.scenarios.definitions import SCENARIOS
from ml_model_dojo.ui.theme import apply_theme


@pytest.fixture(scope="session", autouse=True)
def _theme(qapp):
    apply_theme(qapp)


@pytest.fixture
def win(qtbot):
    from ml_model_dojo.ui.main_window import MainWindow
    w = MainWindow()
    qtbot.addWidget(w)
    w.show()
    yield w
    w.compare.shutdown()


def test_decision_tree_on_xor_shows_metrics(qtbot, win):
    win.data_panel.set_state(DataConfig("classification", "xor", 150), 1)
    assert win.dataset.config.dataset == "xor"
    tab = win.model_tabs["tree"]
    win.tabs.setCurrentWidget(tab)
    tab.toggle_play()
    qtbot.waitUntil(lambda: tab.run is not None and tab.run.finished, timeout=30000)
    item = tab.metrics.table.item(0, 1)           # test accuracy
    assert item is not None and item.text() not in ("-", "")
    assert float(item.text()) > 0.8
    assert "Overfitting?" in tab.metrics.badge.text()


def test_changing_data_clears_models(qtbot, win):
    tab = win.model_tabs["knn"]
    tab.fit_now()
    qtbot.waitUntil(lambda: tab.run is not None and tab.run.finished, timeout=20000)
    win.data_panel.set_state(DataConfig("classification", "moons", 150), 2)
    assert tab.run is None


def test_param_change_only_affects_that_model(qtbot, win):
    knn, tree = win.model_tabs["knn"], win.model_tabs["tree"]
    tree.fit_now() if not get_spec("tree").iterative("classification") else tree.step_once()
    qtbot.waitUntil(lambda: tree.run is not None, timeout=20000)
    before = tree.run
    knn.params.rows["k"]["widget"].setValue(9)
    qtbot.wait(400)
    assert tree.run is before
    assert knn.current_params()["k"] == 9


def test_compare_runs_two_models(qtbot, win):
    win.data_panel.set_state(DataConfig("classification", "moons", 150), 1)
    c = win.compare
    c.set_selection(["tree", "knn"])
    c.cv.setChecked(True)
    c.run_compare()
    qtbot.waitUntil(lambda: c.worker is None and len(c.results) == 2, timeout=180000)
    for mid in ("tree", "knn"):
        r = c.results[mid]["main"]
        assert r["cv"]["mean"] > 0.5 and r["test"]["accuracy"] > 0.5
    assert c.table.model().rowCount() == 2 and c.trade.model().rowCount() == 2
    head, rows = c.results_rows()
    assert len(rows) == 2 and len(rows[0]) == len(head)
    # nothing changed: a second run is served from the cache and the badge stays hidden
    n_cache = len(c.cache)
    c.run_compare()
    qtbot.waitUntil(lambda: c.worker is None, timeout=30000)
    assert len(c.cache) == n_cache
    # changing a setting in a model tab marks that model as stale
    win.model_tabs["knn"].params.rows["k"]["widget"].setValue(11)
    qtbot.wait(400)
    assert c._is_stale("knn") == "Stale" and c._is_stale("tree") == ""


def test_state_round_trip(qtbot, win):
    win.data_panel.set_state(DataConfig("regression", "inter", 300, noise=0.2), 3)
    win.model_tabs["linear"].params.rows["degree"]["widget"].setValue(2)
    qtbot.wait(400)
    win.compare.set_selection(["linear", "forest"])
    st = win.state_dict()
    win.data_panel.set_state(DataConfig("classification", "xor", 150), 1)
    win.apply_state(st)
    assert win.dataset.config.dataset == "inter" and win.dataset.seed == 3
    assert win.model_tabs["linear"].current_params("regression")["degree"] == 2
    assert win.model_tabs["linear"].params.values()["degree"] == 2
    assert win.compare.selection() == ["linear", "forest"]


def test_scenario_loads_and_runs(qtbot, win):
    sc = next(s for s in SCENARIOS if s["id"] == "xor")
    win.load_scenario(sc)
    assert win.tabs.currentWidget() is win.compare
    qtbot.waitUntil(lambda: win.compare.worker is None and len(win.compare.results) == len(sc["models"]), timeout=180000)
    assert win.compare.results["linear"]["main"]["test"]["accuracy"] < 0.65


def test_every_control_has_tooltip_and_concepts_are_short(win):
    for mid in MODEL_ORDER:
        spec = get_spec(mid)
        assert len(spec.concept.split()) <= 60, mid
        for problem in ("classification", "regression"):
            for p in spec.param_schema(problem):
                assert p.tooltip.strip(), (mid, p.key)


@pytest.mark.parametrize("problem,ds", [("classification", "moons"), ("regression", "lin"), ("regression", "inter")])
def test_all_tabs_fit(qtbot, win, problem, ds):
    win.data_panel.set_state(DataConfig(problem, ds, 150), 1)
    for mid in MODEL_ORDER:
        tab = win.model_tabs[mid]
        if mid == "mlp":
            tab.params.rows["max_epochs"]["widget"].setValue(10)
        win.tabs.setCurrentWidget(tab)
        tab.step_once() if tab.spec.iterative(problem) else tab.fit_now()
        qtbot.waitUntil(lambda: tab.run is not None and tab.run.stage >= 1, timeout=30000)
        assert tab.plot.run is tab.run
