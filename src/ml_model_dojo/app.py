"""Application setup: logging, error handling, theme and the main window."""
from __future__ import annotations

import logging
import multiprocessing
import os
import sys
import traceback
from pathlib import Path

os.environ.setdefault("QT_API", "pyside6")
# Fixed thread counts keep results identical between runs.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

log = logging.getLogger("msp")


def logs_dir() -> Path:
    root = Path(__file__).resolve().parents[2] / "logs"
    for candidate in (root, Path.home() / ".ml_model_dojo" / "logs"):
        try:
            candidate.mkdir(parents=True, exist_ok=True)
            (candidate / ".w").write_text("ok")
            return candidate
        except OSError:
            continue
    return Path.cwd()


def setup_logging() -> Path:
    path = logs_dir() / "app.log"
    logging.basicConfig(filename=str(path), level=logging.INFO, encoding="utf-8",
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    return path


def install_excepthook(get_window):
    def hook(exc_type, exc, tb):
        text = "".join(traceback.format_exception(exc_type, exc, tb))
        log.error("Unhandled exception:\n%s", text)
        try:
            from .ui.dialogs import show_error
            show_error(get_window(), f"{exc_type.__name__}: {exc}", text)
        except Exception:
            print(text, file=sys.stderr)
    sys.excepthook = hook


def main(argv=None) -> int:
    multiprocessing.freeze_support()
    setup_logging()
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QApplication
    QApplication.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    app = QApplication(argv if argv is not None else sys.argv)
    from .ui.theme import apply_theme
    apply_theme(app)
    from .ui.main_window import MainWindow
    holder = {}
    install_excepthook(lambda: holder.get("w"))
    win = MainWindow()
    holder["w"] = win
    win.show()
    return app.exec()
