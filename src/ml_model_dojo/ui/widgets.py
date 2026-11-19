"""Small reusable widgets."""
from __future__ import annotations

from PySide6.QtCore import QSignalBlocker, Qt, Signal
from PySide6.QtWidgets import QHBoxLayout, QLabel, QSlider, QWidget


class LabeledSlider(QWidget):
    """A slider that maps integer steps to a float range and shows the value."""
    valueChanged = Signal(float)

    def __init__(self, lo: float, hi: float, value: float, steps: int = 100, fmt: str = "{:.2f}", tooltip: str = "",
                 name: str = ""):
        super().__init__()
        self.lo, self.hi, self.fmt = lo, hi, fmt
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, steps)
        self.slider.setFocusPolicy(Qt.StrongFocus)
        self.label = QLabel()
        self.label.setMinimumWidth(48)
        h = QHBoxLayout(self)
        h.setContentsMargins(0, 0, 0, 0)
        h.addWidget(self.slider, 1)
        h.addWidget(self.label)
        self.setToolTip(tooltip)
        self.slider.setToolTip(tooltip)
        self.slider.setAccessibleName(name)
        self.set_value(value)
        self.slider.valueChanged.connect(self._changed)

    def value(self) -> float:
        return self.lo + (self.hi - self.lo) * self.slider.value() / self.slider.maximum()

    def set_value(self, v: float):
        with QSignalBlocker(self.slider):
            self.slider.setValue(int(round((v - self.lo) / (self.hi - self.lo) * self.slider.maximum())))
        self.label.setText(self.fmt.format(self.value()))

    def _changed(self, _):
        self.label.setText(self.fmt.format(self.value()))
        self.valueChanged.emit(self.value())
