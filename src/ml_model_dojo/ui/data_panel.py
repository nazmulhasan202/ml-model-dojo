"""The persistent data panel above the model tabs."""
from __future__ import annotations

from PySide6.QtCore import QSignalBlocker, Qt, QTimer, Signal
from PySide6.QtWidgets import (QComboBox, QGridLayout, QGroupBox, QHBoxLayout, QLabel, QPushButton, QSpinBox,
                               QWidget)

from ..data.datasets import (MAX_SAMPLES, MIN_SAMPLES, SIZE_PRESETS, DataConfig, dataset_infos, default_dataset,
                             get_info)
from ..seeding import SEEDS
from .widgets import LabeledSlider
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from .plots import draw_data, reg_norm, y_limits

CUSTOM = "Custom..."


class Thumbnail(FigureCanvasQTAgg):
    def __init__(self):
        fig = Figure(figsize=(1.7, 1.45), dpi=100)
        super().__init__(fig)
        self.ax = fig.add_subplot(111)
        self.setFixedSize(170, 145)
        self.setToolTip("Thumbnail of the current data. Filled markers are training points and hollow markers are test points.")

    def show_data(self, ds):
        ax = self.ax
        ax.clear()
        norm = reg_norm(ds) if ds.problem != "classification" and ds.n_inputs == 2 else None
        draw_data(ax, ds, norm=norm, size=6)
        ax.set_xlim(-1, 1)
        ax.set_ylim(*(y_limits(ds) if ds.n_inputs == 1 else (-1, 1)))
        ax.set_xticks([]); ax.set_yticks([])
        self.figure.subplots_adjust(0.04, 0.04, 0.96, 0.96)
        self.draw_idle()


class DataPanel(QGroupBox):
    changed = Signal(object, int)          # DataConfig, seed

    def __init__(self, parent=None):
        super().__init__("Data (shared by every tab)", parent)
        self.timer = QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.setInterval(250)
        self.timer.timeout.connect(self._emit)
        g = QGridLayout(self)
        g.setVerticalSpacing(4)

        self.problem = QComboBox()
        self.problem.addItem("Classification", "classification")
        self.problem.addItem("Regression", "regression")
        self.problem.setToolTip("Classification predicts a class. Regression predicts a number.")
        self.dataset = QComboBox()
        self.dataset.setMinimumWidth(260)
        self.dataset.setToolTip("Choose the shape of the synthetic data.")
        self.seed = QComboBox()
        for s in SEEDS:
            self.seed.addItem(str(s), s)
        self.seed.setMaximumWidth(80)
        self.seed.setToolTip("The seed fixes the data, the split and the model initialization. Try other seeds to see how results vary.")
        self.size_box = QComboBox()
        for name, n in SIZE_PRESETS.items():
            self.size_box.addItem(name, n)
        self.size_box.addItem(CUSTOM, -1)
        self.size_box.setCurrentIndex(1)
        self.size_box.setToolTip("Number of points, including the test set. More data usually shrinks the train and test gap.")
        self.custom = QSpinBox()
        self.custom.setRange(MIN_SAMPLES, MAX_SAMPLES)
        self.custom.setValue(150)
        self.custom.setVisible(False)
        self.custom.setToolTip("Custom sample size between 20 and 20,000.")
        self.noise = LabeledSlider(0, 1, 0.3, 100, "{:.2f}", "How noisy the data are. More noise blurs the pattern.", "Noise")
        self.label_noise = LabeledSlider(0, 0.30, 0.0, 30, "{:.0%}", "Share of labels flipped at random. This is irreducible error.", "Label noise")
        self.balance = LabeledSlider(0.02, 0.5, 0.5, 48, "{:.0%}", "Share of the minority class. 50 percent is balanced, 2 percent is very rare.", "Class balance")
        self.split = LabeledSlider(0.1, 0.5, 0.3, 40, "{:.0%}", "Share of points held out for testing.", "Test split")
        self.mode = QComboBox()
        self.mode.addItem("Random", "random")
        self.mode.addItem("Extrapolation", "extrapolation")
        self.mode.setToolTip("Extrapolation trains on x up to 0.4 and tests on x from 0.6, so models must predict outside their training range.")
        self.regen = QPushButton("Regenerate")
        self.regen.setToolTip("Generate the data again and clear all trained models.")
        self.blurb = QLabel()
        self.blurb.setWordWrap(True)
        self.blurb.setTextFormat(Qt.RichText)
        self.thumb = Thumbnail()

        self.lbl = {}

        def add(row, col, text, w):
            lab = QLabel(text)
            lab.setBuddy(w)
            g.addWidget(lab, row, col * 2)
            g.addWidget(w, row, col * 2 + 1)
            self.lbl[text] = (lab, w)

        add(0, 0, "Problem:", self.problem)
        add(0, 1, "Dataset:", self.dataset)
        add(0, 2, "Seed:", self.seed)
        szw = QWidget()
        h = QHBoxLayout(szw); h.setContentsMargins(0, 0, 0, 0)
        h.addWidget(self.size_box); h.addWidget(self.custom)
        add(1, 0, "Sample size:", szw)
        add(1, 1, "Noise:", self.noise)
        add(1, 2, "Label noise:", self.label_noise)
        add(2, 0, "Test split:", self.split)
        add(2, 1, "Class balance:", self.balance)
        add(2, 2, "Test mode:", self.mode)
        g.addWidget(self.regen, 3, 0)
        g.addWidget(self.blurb, 3, 1, 1, 5)
        g.addWidget(self.thumb, 0, 6, 4, 1)
        g.setColumnStretch(3, 1)
        g.setColumnStretch(5, 1)

        self._fill_datasets("classification", default_dataset("classification"))
        self.problem.currentIndexChanged.connect(self._problem_changed)
        self.dataset.currentIndexChanged.connect(self._dataset_changed)
        self.seed.currentIndexChanged.connect(self._schedule)
        self.size_box.currentIndexChanged.connect(self._size_changed)
        self.custom.valueChanged.connect(self._schedule)
        for s in (self.noise, self.label_noise, self.balance, self.split):
            s.valueChanged.connect(self._schedule)
        self.mode.currentIndexChanged.connect(self._schedule)
        self.regen.clicked.connect(self._emit)
        self._update_visibility()

    # ---- helpers ---------------------------------------------------------
    def _fill_datasets(self, problem: str, select: str):
        with QSignalBlocker(self.dataset):
            self.dataset.clear()
            for info in dataset_infos(problem):
                self.dataset.addItem(f"{info.name}  [{info.tag}]", info.id)
                self.dataset.setItemData(self.dataset.count() - 1, info.description, Qt.ToolTipRole)
            self.dataset.setCurrentIndex(max(0, self.dataset.findData(select)))

    def _problem_changed(self):
        p = self.problem.currentData()
        self._fill_datasets(p, default_dataset(p))
        self._update_visibility()
        self._schedule()

    def _dataset_changed(self):
        self._update_visibility()
        self._schedule()

    def _size_changed(self):
        self.custom.setVisible(self.size_box.currentData() == -1)
        self._schedule()

    def _update_visibility(self):
        cls = self.problem.currentData() == "classification"
        info = get_info(self.problem.currentData(), self.dataset.currentData())
        for key, show in (("Label noise:", cls), ("Class balance:", cls), ("Test mode:", (not cls) and info.n_inputs == 1)):
            lab, w = self.lbl[key]
            lab.setVisible(show)
            w.setVisible(show)
        self.blurb.setText(f"<b>{info.tag}.</b> {info.description} <i>What this teaches:</i> {info.teaches}")

    def _schedule(self, *_):
        self.timer.start()

    def _emit(self):
        self._update_visibility()
        self.changed.emit(self.config(), self.seed_value())

    # ---- state -----------------------------------------------------------
    def seed_value(self) -> int:
        return int(self.seed.currentData())

    def config(self) -> DataConfig:
        n = self.size_box.currentData()
        if n == -1:
            n = self.custom.value()
        cls = self.problem.currentData() == "classification"
        info = get_info(self.problem.currentData(), self.dataset.currentData())
        return DataConfig(
            problem=self.problem.currentData(), dataset=self.dataset.currentData(), n_samples=int(n),
            noise=round(self.noise.value(), 4), label_noise=round(self.label_noise.value(), 4) if cls else 0.0,
            minority_share=round(self.balance.value(), 4) if cls else 0.5, test_frac=round(self.split.value(), 4),
            test_mode=self.mode.currentData() if (not cls and info.n_inputs == 1) else "random")

    def set_state(self, cfg: DataConfig, seed: int, emit: bool = True):
        self.timer.stop()
        for w in (self.problem, self.dataset, self.seed, self.size_box, self.custom, self.mode):
            w.blockSignals(True)
        self.problem.setCurrentIndex(max(0, self.problem.findData(cfg.problem)))
        self._fill_datasets(cfg.problem, cfg.dataset)
        self.seed.setCurrentIndex(max(0, self.seed.findData(seed)))
        idx = self.size_box.findData(cfg.n_samples)
        if idx >= 0:
            self.size_box.setCurrentIndex(idx)
        else:
            self.size_box.setCurrentIndex(self.size_box.findData(-1))
            self.custom.setValue(cfg.n_samples)
        self.custom.setVisible(self.size_box.currentData() == -1)
        self.mode.setCurrentIndex(max(0, self.mode.findData(cfg.test_mode)))
        for w in (self.problem, self.dataset, self.seed, self.size_box, self.custom, self.mode):
            w.blockSignals(False)
        self.noise.set_value(cfg.noise)
        self.label_noise.set_value(cfg.label_noise)
        self.balance.set_value(cfg.minority_share)
        self.split.set_value(cfg.test_frac)
        self._update_visibility()
        if emit:
            self._emit()

    def show_thumbnail(self, ds):
        self.thumb.show_data(ds)
