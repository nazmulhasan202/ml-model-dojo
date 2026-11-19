import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import pytest


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: exhaustive sweeps (deselect with -m 'not slow')")


@pytest.fixture(autouse=True)
def isolated_qsettings(tmp_path, monkeypatch):
    """Keep tests from reading or writing the real application settings (the Windows registry).

    QSettings(org, app) ignores QSettings.setDefaultFormat on Windows, so the window module is given a
    factory that always returns an ini file inside the test's temporary folder.
    """
    from PySide6.QtCore import QSettings
    import ml_model_dojo.ui.main_window as mw

    monkeypatch.setattr(mw, "QSettings", lambda *a, **k: QSettings(str(tmp_path / "settings.ini"), QSettings.IniFormat))
    yield


@pytest.fixture(scope="session", autouse=True)
def _theme(qapp):
    from ml_model_dojo.ui.theme import apply_theme
    apply_theme(qapp)


@pytest.fixture
def win(qtbot, monkeypatch):
    """A main window whose error dialog is replaced by a list, so tests fail on any reported error."""
    from ml_model_dojo.ui.main_window import MainWindow

    errors: list[str] = []
    monkeypatch.setattr(MainWindow, "report_error", lambda self, tb: errors.append(tb))
    w = MainWindow()
    w._errors = errors
    qtbot.addWidget(w)
    w.show()
    yield w
    w.compare.shutdown()
    assert errors == [], "errors were reported to the user:\n" + "\n---\n".join(errors)
