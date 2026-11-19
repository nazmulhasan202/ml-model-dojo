"""Dialogs: metric glossary, help with guided tour, error reporting."""
from __future__ import annotations

import logging
from pathlib import Path

from PySide6.QtCore import QObject, QRect, Qt
from PySide6.QtWidgets import (QDialog, QDialogButtonBox, QFrame, QHBoxLayout, QLabel, QMessageBox, QPushButton,
                               QTabWidget, QTextBrowser, QVBoxLayout, QWidget)

from .. import metrics as M

log = logging.getLogger(__name__)

TERMS = {
    "Overfitting": "The model fits noise in the training data, so training scores look great while test scores fall behind.",
    "Underfitting": "The model is too simple to capture the pattern, so both training and test scores are poor.",
    "Train and test split": "Data are split once. The model learns from the training part only, and the test part checks how well it generalizes.",
    "Cross-validation": "The training data are split into k folds. Each fold takes a turn as a validation set, which gives a more stable estimate and a spread.",
    "Seed": "A number that fixes the random choices, so the same seed always gives the same data, split and model.",
    "Bagging": "Train many models on random resamples of the rows and average them. Random forests do this.",
    "Boosting": "Train models one after another, each fixing the errors of the previous ones. XGBoost does this.",
    "Decision threshold": "The probability above which a point is called class 1. Lowering it finds more positives but adds false alarms.",
    "Majority baseline": "The accuracy of always predicting the most common class. Any useful model must beat it.",
}


def glossary_html() -> str:
    parts = ["<h3>Metrics</h3><ul>"]
    for k in M.CLS_KEYS + M.REG_KEYS:
        parts.append(f"<li><b>{M.LABELS[k]}:</b> {M.GLOSSARY[k]}</li>")
    parts.append("</ul><h3>Terms</h3><ul>")
    for k, v in TERMS.items():
        parts.append(f"<li><b>{k}:</b> {v}</li>")
    parts.append("</ul>")
    return "".join(parts)


class GlossaryDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Metric glossary")
        self.resize(560, 520)
        v = QVBoxLayout(self)
        tb = QTextBrowser()
        tb.setHtml(glossary_html())
        v.addWidget(tb)
        bb = QDialogButtonBox(QDialogButtonBox.Close)
        bb.rejected.connect(self.reject)
        v.addWidget(bb)


class HelpDialog(QDialog):
    def __init__(self, parent, start_tour):
        super().__init__(parent)
        self.setWindowTitle("Help")
        self.resize(620, 540)
        v = QVBoxLayout(self)
        tabs = QTabWidget()
        about = QTextBrowser()
        about.setHtml(
            "<h3>ML Model Dojo</h3>"
            "<p>Pick a dataset at the top, then open a model tab and press <b>Train</b>. Compare models on the same data in the "
            "<b>Compare</b> tab. Use <b>Scenarios</b> for guided lessons.</p>"
            "<ul><li>Filled markers are training points. Hollow markers are test points. Red rings are test errors.</li>"
            "<li>Changing the data clears all trained models. Changing a hyperparameter only affects that model.</li>"
            "<li>Hover over the plot to read predictions in the status bar.</li>"
            "<li>All data are synthetic and seeded, so results are repeatable.</li></ul>")
        tabs.addTab(about, "Overview")
        gl = QTextBrowser()
        gl.setHtml(glossary_html())
        tabs.addTab(gl, "Glossary")
        v.addWidget(tabs)
        row = QHBoxLayout()
        tour = QPushButton("Start guided tour")
        tour.clicked.connect(lambda: (self.accept(), start_tour()))
        close = QPushButton("Close")
        close.clicked.connect(self.reject)
        row.addWidget(tour)
        row.addStretch()
        row.addWidget(close)
        v.addLayout(row)


class TourController(QObject):
    """A sequence of highlighted steps. ``steps`` is a list of (widget getter, title, text)."""

    def __init__(self, window: QWidget, steps):
        super().__init__(window)
        self.window, self.steps, self.i = window, steps, 0
        self.frame = QFrame(window)
        self.frame.setStyleSheet("border: 3px solid #D55E00; background: transparent;")
        self.frame.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.dlg = QDialog(window, Qt.Tool | Qt.WindowTitleHint)
        self.dlg.setWindowTitle("Guided tour")
        v = QVBoxLayout(self.dlg)
        self.title = QLabel()
        self.title.setStyleSheet("font-weight: bold; font-size: 11pt;")
        self.text = QLabel()
        self.text.setWordWrap(True)
        self.text.setMinimumWidth(320)
        v.addWidget(self.title)
        v.addWidget(self.text)
        row = QHBoxLayout()
        self.back, self.next = QPushButton("Back"), QPushButton("Next")
        done = QPushButton("Close")
        self.back.clicked.connect(lambda: self.go(-1))
        self.next.clicked.connect(lambda: self.go(+1))
        done.clicked.connect(self.finish)
        row.addWidget(self.back); row.addWidget(self.next); row.addStretch(); row.addWidget(done)
        v.addLayout(row)
        self.dlg.finished.connect(lambda _: self.frame.hide())

    def start(self):
        self.i = 0
        self.dlg.show()
        self.show_step()

    def go(self, d):
        self.i = max(0, min(len(self.steps) - 1, self.i + d))
        self.show_step()

    def show_step(self):
        getter, title, text = self.steps[self.i]
        self.title.setText(f"{self.i + 1} of {len(self.steps)}: {title}")
        self.text.setText(text)
        self.back.setEnabled(self.i > 0)
        self.next.setEnabled(self.i < len(self.steps) - 1)
        w = getter()
        if w is not None and w.isVisible():
            top_left = w.mapTo(self.window, w.rect().topLeft())
            self.frame.setGeometry(QRect(top_left, w.size()).adjusted(-3, -3, 3, 3))
            self.frame.show()
            self.frame.raise_()
        else:
            self.frame.hide()

    def finish(self):
        self.dlg.close()
        self.frame.hide()


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".pdf", ".svg", ".tif", ".tiff", ".webp"}


def fix_suffix(path: str, default: str, allowed: set[str] | None = None) -> str:
    """Add a default extension when the typed name has none, or one the writer cannot produce."""
    suffix = Path(path).suffix.lower()
    if allowed is None:
        return path if suffix else path + default
    return path if suffix in allowed else path + default


def safe_save(parent, path: str, writer) -> bool:
    """Call writer(path). A locked, read-only or missing file gives a plain message, not a traceback."""
    try:
        writer(path)
        return True
    except (OSError, ValueError) as e:
        log.warning("Could not save %s: %s", path, e)
        show_error(parent, "The file could not be saved. Check that the folder exists, that you may write to it, "
                           "and that the file is not open in another program such as Excel.", f"{type(e).__name__}: {e}")
        return False


def show_error(parent, message: str, details: str = ""):
    box = QMessageBox(parent)
    box.setIcon(QMessageBox.Warning)
    box.setWindowTitle("Something went wrong")
    box.setText(message)
    box.setInformativeText("The full details were written to logs/app.log. The application is still running.")
    if details:
        box.setDetailedText(details)
    box.exec()
