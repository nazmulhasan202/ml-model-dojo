"""Hyperparameter panel generated from a list of ParamSpec objects."""
from __future__ import annotations

import math

from PySide6.QtCore import QSignalBlocker, Qt, QTimer, Signal
from PySide6.QtWidgets import (QCheckBox, QComboBox, QFormLayout, QGridLayout, QHBoxLayout, QLabel, QPushButton, QSlider,
                               QVBoxLayout, QWidget)

from ..models.base import ParamSpec
from ..models.mlp import feature_options

SLIDER_STEPS = 1000
DEBOUNCE_MS = 150


class LayersEditor(QWidget):
    """Hidden layers editor: 1 to 5 layers, 1 to 16 neurons each."""
    changed = Signal()
    MAX_LAYERS, MAX_NEURONS = 5, 16

    def __init__(self, layers, parent=None):
        super().__init__(parent)
        self.layers = [int(n) for n in layers]
        self.box = QVBoxLayout(self)
        self.box.setContentsMargins(0, 0, 0, 0)
        self._rebuild()

    def value(self) -> list[int]:
        return list(self.layers)

    def set_value(self, layers):
        self.layers = [int(n) for n in layers]
        self._rebuild()

    def _rebuild(self):
        while self.box.count():
            item = self.box.takeAt(0)
            if item.widget():
                item.widget().hide()
                item.widget().setParent(None)
                item.widget().deleteLater()
            elif item.layout():
                self._clear(item.layout())
        for i, n in enumerate(self.layers):
            row = QHBoxLayout()
            row.addWidget(QLabel(f"Layer {i + 1}"))
            minus, plus = QPushButton("-"), QPushButton("+")
            for b, txt in ((minus, f"Remove a neuron from layer {i + 1}"), (plus, f"Add a neuron to layer {i + 1}")):
                b.setFixedWidth(32)
                b.setObjectName("mini")
                b.setToolTip(txt)
                b.setAccessibleName(txt)
            lab = QLabel(f"{n}")
            lab.setMinimumWidth(24)
            lab.setAlignment(Qt.AlignCenter)
            lab.setToolTip(f"Neurons in layer {i + 1}")
            minus.clicked.connect(lambda _=False, i=i: self._bump(i, -1))
            plus.clicked.connect(lambda _=False, i=i: self._bump(i, +1))
            row.addWidget(minus); row.addWidget(lab); row.addWidget(plus); row.addStretch()
            holder = QWidget()
            holder.setLayout(row)
            self.box.addWidget(holder)
        row = QHBoxLayout()
        add, rem = QPushButton("+ Layer"), QPushButton("- Layer")
        add.setMinimumWidth(80)
        rem.setMinimumWidth(80)
        add.setToolTip("Add a hidden layer (up to 5).")
        rem.setToolTip("Remove the last hidden layer (at least 1 stays).")
        add.setEnabled(len(self.layers) < self.MAX_LAYERS)
        rem.setEnabled(len(self.layers) > 1)
        add.clicked.connect(self._add)
        rem.clicked.connect(self._remove)
        row.addWidget(add); row.addWidget(rem)
        holder = QWidget()
        holder.setLayout(row)
        self.box.addWidget(holder)

    @staticmethod
    def _clear(layout):
        while layout.count():
            it = layout.takeAt(0)
            if it.widget():
                it.widget().deleteLater()

    def _bump(self, i, d):
        self.layers[i] = int(min(self.MAX_NEURONS, max(1, self.layers[i] + d)))
        self._rebuild()
        self.changed.emit()

    def _add(self):
        if len(self.layers) < self.MAX_LAYERS:
            self.layers.append(self.layers[-1])
            self._rebuild()
            self.changed.emit()

    def _remove(self):
        if len(self.layers) > 1:
            self.layers.pop()
            self._rebuild()
            self.changed.emit()


class ParamPanel(QWidget):
    changed = Signal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.form = QFormLayout(self)
        self.form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        self.form.setLabelAlignment(Qt.AlignLeft)
        self.form.setContentsMargins(4, 4, 4, 4)
        self.rows: dict[str, dict] = {}
        self.timer = QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.setInterval(DEBOUNCE_MS)
        self.timer.timeout.connect(lambda: self.changed.emit(self.values()))

    # ---- building --------------------------------------------------------
    def set_schema(self, schema: list[ParamSpec], values: dict, n_inputs: int = 2):
        while self.form.rowCount():
            self.form.removeRow(0)
        self.rows = {}
        for spec in schema:
            self._add_row(spec, values.get(spec.key, spec.default), n_inputs)
        self._update_visibility()

    def _add_row(self, spec: ParamSpec, value, n_inputs: int):
        label = QLabel(spec.label)
        label.setToolTip(spec.tooltip)
        label.setWordWrap(True)
        info: dict = {"spec": spec, "label": label}
        if spec.kind in ("int", "float", "log"):
            slider = QSlider(Qt.Horizontal)
            slider.setFocusPolicy(Qt.StrongFocus)
            val = QLabel()
            val.setMinimumWidth(62)
            if spec.kind == "int":
                slider.setRange(int(spec.lo), int(spec.hi))
            else:
                slider.setRange(0, SLIDER_STEPS)
            info.update(widget=slider, valuelabel=val)
            wrap = QWidget()
            h = QHBoxLayout(wrap)
            h.setContentsMargins(0, 0, 0, 0)
            h.addWidget(slider, 1)
            h.addWidget(val)
            info["field"] = wrap
            slider.valueChanged.connect(lambda _=0, k=spec.key: self._on_slider(k))
            self._set_slider(info, value)
        elif spec.kind == "choice":
            combo = QComboBox()
            for v, text in spec.choices:
                combo.addItem(text, v)
            idx = max(0, combo.findData(value))
            combo.setCurrentIndex(idx)
            combo.currentIndexChanged.connect(lambda _=0: self._on_change())
            info.update(widget=combo, field=combo)
        elif spec.kind == "bool":
            cb = QCheckBox()
            cb.setChecked(bool(value))
            cb.toggled.connect(lambda _=False: self._on_change())
            info.update(widget=cb, field=cb)
        elif spec.kind == "layers":
            ed = LayersEditor(value)
            ed.changed.connect(self._on_change)
            info.update(widget=ed, field=ed)
        elif spec.kind == "flags":
            wrap = QWidget()
            v = QGridLayout(wrap)
            v.setContentsMargins(0, 0, 0, 0)
            boxes = {}
            options = feature_options(n_inputs)
            chosen = [x for x in value if x in dict(options)] or [options[0][0]]
            for n_, (key, text) in enumerate(options):
                cb = QCheckBox(text)
                cb.setChecked(key in chosen)
                cb.toggled.connect(lambda _=False: self._on_change())
                v.addWidget(cb, n_, 0)
                boxes[key] = cb
            info.update(widget=wrap, field=wrap, boxes=boxes)
        else:
            raise ValueError(spec.kind)
        info["field"].setToolTip(spec.tooltip)
        if hasattr(info["widget"], "setAccessibleName"):
            info["widget"].setAccessibleName(spec.label)
        self.form.addRow(label, info["field"])
        self.rows[spec.key] = info

    # ---- slider mapping --------------------------------------------------
    def _slider_value(self, info) -> float:
        s, sl = info["spec"], info["widget"]
        if s.kind == "int":
            return sl.value()
        t = sl.value() / SLIDER_STEPS
        if s.kind == "float":
            return s.lo + (s.hi - s.lo) * t
        return s.lo * (s.hi / s.lo) ** t

    def _set_slider(self, info, value):
        s, sl = info["spec"], info["widget"]
        with QSignalBlocker(sl):
            if s.kind == "int":
                sl.setValue(int(value))
            elif s.kind == "float":
                sl.setValue(int(round((float(value) - s.lo) / (s.hi - s.lo) * SLIDER_STEPS)))
            else:
                sl.setValue(int(round(math.log(float(value) / s.lo) / math.log(s.hi / s.lo) * SLIDER_STEPS)))
        self._show_slider_text(info)

    def _show_slider_text(self, info):
        s, v = info["spec"], self._slider_value(info)
        if s.kind == "int":
            txt = s.zero_label if (s.zero_label and int(v) == int(s.lo)) else str(int(v))
        elif s.kind == "float":
            txt = f"{v:.{s.decimals}f}"
        else:
            txt = f"{v:.3g}"
        info["valuelabel"].setText(txt)

    def _on_slider(self, key):
        self._show_slider_text(self.rows[key])
        self._on_change()

    # ---- values ----------------------------------------------------------
    def values(self) -> dict:
        out = {}
        for key, info in self.rows.items():
            s, w = info["spec"], info["widget"]
            if s.kind in ("int", "float", "log"):
                v = self._slider_value(info)
                out[key] = int(v) if s.kind == "int" else round(float(v), 6)
            elif s.kind == "choice":
                out[key] = w.currentData()
            elif s.kind == "bool":
                out[key] = w.isChecked()
            elif s.kind == "layers":
                out[key] = w.value()
            elif s.kind == "flags":
                out[key] = [k for k, cb in info["boxes"].items() if cb.isChecked()]
        return out

    def set_values(self, values: dict):
        for key, info in self.rows.items():
            if key not in values:
                continue
            s, w, v = info["spec"], info["widget"], values[key]
            if s.kind in ("int", "float", "log"):
                self._set_slider(info, v)
            elif s.kind == "choice":
                with QSignalBlocker(w):
                    w.setCurrentIndex(max(0, w.findData(v)))
            elif s.kind == "bool":
                with QSignalBlocker(w):
                    w.setChecked(bool(v))
            elif s.kind == "layers":
                w.set_value(v)
            elif s.kind == "flags":
                for k, cb in info["boxes"].items():
                    with QSignalBlocker(cb):
                        cb.setChecked(k in v)
        self._update_visibility()

    def _on_change(self):
        self._update_visibility()
        self.timer.start()

    def _update_visibility(self):
        vals = self.values()
        for info in self.rows.values():
            s = info["spec"]
            show = s.show_if(vals) if s.show_if else True
            info["label"].setVisible(show)
            info["field"].setVisible(show)
