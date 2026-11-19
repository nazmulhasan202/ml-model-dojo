"""Right column of a model tab: metrics, gap indicator, confusion matrix, curves and facts."""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QAbstractItemView, QComboBox, QHeaderView, QHBoxLayout, QLabel, QPushButton, QScrollArea,
                               QStackedWidget, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget)

from .. import metrics as M
from ..models.base import CLS
from .dialogs import GlossaryDialog
from .side_plots import ConfusionPlot, CurvePlot, PredActualPlot

OVERFIT_TIP = ("The training score is much higher than the test score (gap above 10 points). "
               "The model has memorized the training data instead of learning a pattern that carries over to new data.")


class MetricsPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.problem = CLS
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        inner = QWidget()
        scroll.setWidget(inner)
        outer.addWidget(scroll)
        v = QVBoxLayout(inner)

        self.header = QLabel()
        self.header.setWordWrap(True)
        self.header.setObjectName("empty")
        v.addWidget(self.header)

        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["Train", "Test", "Gap"])
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionMode(QAbstractItemView.NoSelection)
        self.table.setFocusPolicy(Qt.NoFocus)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.verticalHeader().setMaximumWidth(120)
        self.table.setAccessibleName("Train and test metrics")
        self.table.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.table.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        v.addWidget(self.table)

        row = QHBoxLayout()
        self.badge = QLabel("Overfitting? no data")
        self.badge.setToolTip(OVERFIT_TIP)
        self.baseline = QLabel()
        self.baseline.setWordWrap(True)
        row.addWidget(self.badge)
        row.addStretch()
        v.addLayout(row)
        v.addWidget(self.baseline)

        self.stack = QStackedWidget()
        self.confusion = ConfusionPlot()
        self.pred_actual = PredActualPlot()
        self.stack.addWidget(self.confusion)
        self.stack.addWidget(self.pred_actual)
        v.addWidget(self.stack)

        crow = QHBoxLayout()
        crow.addWidget(QLabel("Training curve:"))
        self.mode = QComboBox()
        self.mode.addItem("Loss", "loss")
        self.mode.addItem("Score", "score")
        self.mode.setToolTip("Show the loss or the score (accuracy or R squared) for training and test data.")
        crow.addWidget(self.mode)
        crow.addStretch()
        v.addLayout(crow)
        self.curve = CurvePlot()
        v.addWidget(self.curve)
        self.mode.currentIndexChanged.connect(lambda _: self.curve.update_run(self._run, self.mode.currentData()))

        self.notice = QLabel()
        self.notice.setWordWrap(True)
        self.notice.setStyleSheet("color: #8a4b00;")
        v.addWidget(self.notice)
        self.facts = QLabel()
        self.facts.setWordWrap(True)
        self.facts.setTextFormat(Qt.RichText)
        v.addWidget(self.facts)
        gl = QPushButton("Metric glossary")
        gl.setToolTip("Open a short explanation of every metric, with a tiny worked example.")
        gl.clicked.connect(lambda: GlossaryDialog(self).exec())
        v.addWidget(gl)
        v.addStretch()
        self._run = None
        self.set_problem(CLS)

    # ---- state -------------------------------------------------------------
    def set_problem(self, problem: str):
        self.problem = problem
        keys = M.CLS_KEYS if problem == CLS else M.REG_KEYS
        self.table.setRowCount(len(keys))
        self.keys = keys
        for i, k in enumerate(keys):
            it = QTableWidgetItem(M.LABELS[k])
            it.setToolTip(M.GLOSSARY[k])
            self.table.setVerticalHeaderItem(i, it)
        self.stack.setCurrentIndex(0 if problem == CLS else 1)
        self.clear("Choose a dataset and press Train.")

    def _fit_table_height(self):
        self.table.resizeRowsToContents()
        self.table.setFixedHeight(self.table.horizontalHeader().height() + sum(
            self.table.rowHeight(r) for r in range(self.table.rowCount())) + 8)

    def clear(self, message: str):
        self._run = None
        self.header.setText(message)
        for r in range(self.table.rowCount()):
            for c in range(3):
                self.table.setItem(r, c, QTableWidgetItem("-"))
        self._fit_table_height()
        self.badge.setText("Overfitting? no model yet")
        self.badge.setObjectName("badge_ok")
        self.badge.setStyleSheet("")
        self.baseline.setText("")
        self.notice.setText("")
        self.facts.setText("")
        self.confusion.update_cm(None)
        self.pred_actual.update_run(None, None)
        self.curve.update_run(None)

    def update_run(self, run, ev, data, full: bool = True):
        self._run = run
        n = run.n_stages
        label = f"Step {run.stage} of {n}" if n > 1 else "Fitted"
        self.header.setText(label + (" (training...)" if run.stage < n else " (done)"))
        tr, te = ev["train"], ev["test"]
        for r, k in enumerate(self.keys):
            gap = tr[k] - te[k]
            for c, v in enumerate((tr[k], te[k], gap)):
                it = QTableWidgetItem(f"{v:.3f}" if v == v else "n/a")
                it.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                if k == M.main_score_key(self.problem):
                    f = it.font()
                    f.setBold(True)
                    it.setFont(f)
                self.table.setItem(r, c, it)
        self._fit_table_height()
        flag = M.is_overfit(tr, te, self.problem)
        g = M.gap(tr, te, self.problem)
        self.badge.setText(f"Overfitting? {'YES' if flag else 'no'} (gap {g:+.2f})")
        self.badge.setObjectName("badge_warn" if flag else "badge_ok")
        self.badge.setStyleSheet("")  # force the stylesheet to re-evaluate the object name
        self.badge.style().unpolish(self.badge)
        self.badge.style().polish(self.badge)
        if self.problem == CLS:
            self.baseline.setText(f"Majority class baseline: accuracy {te['baseline']:.3f} on the test set. "
                                  "A useful model must beat this, and accuracy alone can hide a model that ignores the rare class.")
            if full:
                self.confusion.update_cm(te["confusion"])
        elif full:
            self.pred_actual.update_run(data, ev)
        if full:
            self.curve.update_run(run, self.mode.currentData())
        self.notice.setText(run.notice)
        self._show_facts(run, full)

    def _show_facts(self, run, full):
        rows = [f"<b>{k}:</b> {v}" for k, v in run.facts().items()]
        rows.append(f"<b>Training time:</b> {run.train_time * 1000:.0f} ms <i>(machine dependent)</i>")
        if full and run.stage > 0:
            rows.append(f"<b>Prediction time (test set):</b> {run.predict_time(run.data.X_test, 1) * 1000:.1f} ms <i>(machine dependent)</i>")
        self.facts.setText("<br>".join(rows))
