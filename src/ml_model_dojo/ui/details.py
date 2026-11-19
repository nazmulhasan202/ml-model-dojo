"""Model-specific details shown in the second tab of the right column."""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QAbstractItemView, QCheckBox, QLabel, QScrollArea, QTableWidget, QTableWidgetItem,
                               QVBoxLayout, QWidget)

from ..models.mlp import feature_options
from .side_plots import NetworkDiagram, SmallMultiples, TreeDiagram


def _label(text="", wrap=True):
    lab = QLabel(text)
    lab.setWordWrap(wrap)
    lab.setTextInteractionFlags(Qt.TextSelectableByMouse)
    return lab


class Details(QWidget):
    """Base class. Subclasses build widgets and implement ``refresh``."""

    def __init__(self):
        super().__init__()
        self.v = QVBoxLayout(self)
        self.hint = _label()
        self.v.addWidget(self.hint)

    def clear(self):
        self.hint.setText("Train the model to see details.")

    def refresh(self, run, data):
        pass


class LinearDetails(Details):
    def __init__(self):
        super().__init__()
        self.eq = _label()
        self.table = QTableWidget(0, 2)
        self.table.setHorizontalHeaderLabels(["Term", "Coefficient"])
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.note = _label()
        for w in (self.eq, self.table, self.note):
            self.v.addWidget(w)

    def clear(self):
        super().clear()
        self.eq.setText("")
        self.table.setRowCount(0)
        self.note.setText("")

    def refresh(self, run, data):
        ex = run.extra()
        self.hint.setText("Learned coefficients")
        self.eq.setText(f"<b>{ex['equation']}</b>" if ex["equation"] else "")
        rows = ex["coef_rows"]
        self.table.setRowCount(len(rows))
        for i, (n, c) in enumerate(rows):
            self.table.setItem(i, 0, QTableWidgetItem(n))
            it = QTableWidgetItem(f"{c:.4f}")
            it.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self.table.setItem(i, 1, it)
        self.note.setText(ex["coef_note"])


class KnnDetails(Details):
    def clear(self):
        self.hint.setText("Hover over the plot to draw lines to the k nearest training points of that location. "
                          "Train the model first.")

    def refresh(self, run, data):
        self.clear()


class TreeDetails(Details):
    def __init__(self):
        super().__init__()
        self.diagram = TreeDiagram()
        sa = QScrollArea()
        sa.setWidgetResizable(True)
        sa.setWidget(self.diagram)
        self.v.addWidget(sa, 1)

    def clear(self):
        self.hint.setText("Tree diagram (display capped at depth 4)")
        self.diagram.update_run(None)

    def refresh(self, run, data):
        self.hint.setText("Tree diagram (display capped at depth 4)")
        self.diagram.update_run(run)


class ForestDetails(Details):
    def __init__(self):
        super().__init__()
        self.toggle = QCheckBox("Show the first six individual trees")
        self.toggle.setToolTip("Each small plot is one tree on its own. The ensemble is the average of many such trees.")
        self.multi = SmallMultiples()
        self.oob = _label()
        self.v.addWidget(self.toggle)
        self.v.addWidget(self.multi)
        self.v.addWidget(self.oob)
        self.v.addStretch()
        self.multi.setVisible(False)
        self.toggle.toggled.connect(self.multi.setVisible)
        self.toggle.toggled.connect(self._on_toggle)
        self._run = self._data = None

    def _on_toggle(self, on: bool):
        # There is nothing to draw until a forest has been trained (and again after the data change clears it).
        if on and self._run is not None and self._data is not None:
            self.refresh(self._run, self._data)

    def clear(self):
        self.hint.setText("Out-of-bag score appears after training.")
        self.oob.setText("")
        self._run = self._data = None

    def refresh(self, run, data):
        self._run, self._data = run, data
        oob = run.extra()["oob"]
        self.hint.setText("Random forest details")
        self.oob.setText("Out-of-bag score: " + (f"{oob:.3f}" if oob is not None else "off (bootstrap is off)"))
        if self.toggle.isChecked():
            self.multi.update_run(run, data)


class BoostDetails(Details):
    def clear(self):
        self.hint.setText("The training curve shows the test loss. The marker is the number of trees with the lowest test loss. "
                          "Picking it from test data leaks information, so a real project would use a validation set.")

    def refresh(self, run, data):
        self.clear()
        self.hint.setText(self.hint.text() + f"\n\nLowest test loss at {run.extra()['best_iter']} trees.")


class SvmDetails(Details):
    def clear(self):
        self.hint.setText("Circled points are the support vectors, the training points that define the boundary. "
                          "For a linear kernel the dashed lines show the margin.")

    def refresh(self, run, data):
        self.clear()


class MlpDetails(Details):
    def __init__(self):
        super().__init__()
        self.net = NetworkDiagram()
        self.v.addWidget(self.net)
        self.v.addStretch()

    def clear(self):
        self.hint.setText("Network diagram. After training, orange lines are positive weights and blue lines are negative.")
        self.net.update_run(["x1", "x2"], [8, 8])

    def refresh(self, run, data):
        self.hint.setText("Network diagram. Orange lines are positive weights and blue lines are negative. Thickness shows size.")
        ex = run.extra()
        names = dict(feature_options(data.n_inputs))
        self.net.update_run([names[f] for f in ex["features"]], ex["layers"], ex["coefs"])


DETAILS = {"linear": LinearDetails, "knn": KnnDetails, "tree": TreeDetails, "forest": ForestDetails,
           "boosting": BoostDetails, "svm": SvmDetails, "mlp": MlpDetails}


def make_details(spec_id: str) -> Details:
    d = DETAILS[spec_id]()
    d.clear()
    return d
