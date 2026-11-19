"""Light theme for Qt and matplotlib, with a color-blind-safe palette."""
from __future__ import annotations

import os

os.environ.setdefault("QT_API", "pyside6")

import matplotlib
from matplotlib.colors import LinearSegmentedColormap
from PySide6.QtGui import QColor, QFont, QPalette
from PySide6.QtWidgets import QApplication

BLUE = "#0072B2"       # class 0
ORANGE = "#E69F00"     # class 1
RED = "#D62728"        # errors
INK = "#1a1a1a"
GRID = "#d0d0d0"
TRAIN_REGION = "#eaf1f8"
TEST_REGION = "#fdf0d9"
MODEL_COLORS = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#D55E00", "#F0E442", "#000000"]

CLS_CMAP = LinearSegmentedColormap.from_list("blue_orange", [BLUE, "#f7f7f7", ORANGE])
REG_CMAP = "viridis"

ACCENT, ACCENT_DARK, ACCENT_SOFT = "#0072B2", "#005a8d", "#e8f2f9"
BG, CARD, LINE, MUTED = "#f3f5f9", "#ffffff", "#dbe1ea", "#5b6777"

QSS = f"""
QWidget {{ font-size: 10pt; color: {INK}; }}
QMainWindow, QDialog {{ background: {BG}; }}
QToolTip {{ background: #1f2933; color: #ffffff; border: none; padding: 5px 8px; border-radius: 4px; }}

QPushButton, QToolButton {{ padding: 6px 14px; border: 1px solid #cdd5e0; border-radius: 8px; background: {CARD}; }}
QPushButton:hover, QToolButton:hover {{ background: #eef3f9; border-color: #b8c4d4; }}
QPushButton:pressed, QToolButton:pressed {{ background: #e1eaf4; }}
QPushButton:focus, QToolButton:focus {{ border: 2px solid {ACCENT}; padding: 5px 13px; }}
QPushButton:disabled {{ color: #9aa5b1; background: #f1f3f7; }}
QPushButton:checked {{ background: {ACCENT_SOFT}; border: 1px solid {ACCENT}; color: {ACCENT_DARK}; font-weight: 600; }}
QPushButton#primary {{ background: {ACCENT}; color: #ffffff; border: none; font-weight: 600; padding: 7px 22px; }}
QPushButton#primary:hover {{ background: {ACCENT_DARK}; }}
QPushButton#primary:disabled {{ background: #9cc3dc; color: #ffffff; }}
QPushButton#primary:focus {{ border: 2px solid #0a2a40; padding: 5px 20px; }}
QPushButton#mini {{ padding: 2px 0; min-width: 0; }}
QPushButton#mini:focus {{ padding: 1px 0; }}
QTabWidget QGroupBox::title {{ background: {CARD}; }}
QPushButton#link {{ border: none; background: transparent; color: {ACCENT}; padding: 0 4px; text-decoration: underline; }}

QComboBox, QSpinBox {{ border: 1px solid #cdd5e0; border-radius: 6px; padding: 4px 8px; background: {CARD}; min-height: 20px; }}
QComboBox:hover, QSpinBox:hover {{ border-color: #9fb2c8; }}
QComboBox:focus, QSpinBox:focus {{ border: 2px solid {ACCENT}; }}
QComboBox QAbstractItemView {{ background: {CARD}; border: 1px solid {LINE}; selection-background-color: {ACCENT_SOFT}; selection-color: {INK}; }}

QGroupBox {{ background: {CARD}; border: 1px solid {LINE}; border-radius: 12px; margin-top: 14px; padding: 12px 10px 8px 10px; font-weight: 600; }}
QGroupBox::title {{ subcontrol-origin: margin; left: 14px; padding: 0 6px; color: {ACCENT_DARK}; background: {BG}; }}
QGroupBox QLabel, QGroupBox QCheckBox {{ font-weight: normal; }}

QSlider:horizontal {{ min-height: 24px; }}
QSlider::groove:horizontal {{ height: 4px; background: #d5dce6; border-radius: 2px; }}
QSlider::sub-page:horizontal {{ background: {ACCENT}; border-radius: 2px; }}
QSlider::handle:horizontal {{ background: {CARD}; border: 2px solid {ACCENT}; width: 10px; height: 10px; margin: -6px 0; border-radius: 7px; }}
QSlider::handle:horizontal:hover {{ background: {ACCENT_SOFT}; }}
QSlider::handle:horizontal:focus {{ border: 3px solid {ACCENT_DARK}; }}

QTabWidget::pane {{ border: 1px solid {LINE}; border-radius: 12px; background: {CARD}; top: -1px; }}
QTabBar::tab {{ padding: 9px 16px; margin-right: 2px; border: none; border-bottom: 3px solid transparent; background: transparent; color: {MUTED}; }}
QTabBar::tab:hover {{ color: {INK}; background: #e9eef5; border-top-left-radius: 8px; border-top-right-radius: 8px; }}
QTabBar::tab:selected {{ color: {ACCENT_DARK}; font-weight: 600; border-bottom: 3px solid {ACCENT}; }}
QTabBar::tab:focus {{ background: {ACCENT_SOFT}; }}

QScrollArea {{ border: none; background: transparent; }}
QScrollBar:vertical {{ width: 10px; background: transparent; margin: 2px; }}
QScrollBar::handle:vertical {{ background: #c3ccd8; border-radius: 4px; min-height: 30px; }}
QScrollBar::handle:vertical:hover {{ background: #a6b3c3; }}
QScrollBar:horizontal {{ height: 10px; background: transparent; margin: 2px; }}
QScrollBar::handle:horizontal {{ background: #c3ccd8; border-radius: 4px; min-width: 30px; }}
QScrollBar::add-line, QScrollBar::sub-line {{ width: 0; height: 0; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}

QTableView, QTableWidget {{ background: {CARD}; alternate-background-color: #f6f8fb; border: 1px solid {LINE}; border-radius: 8px; gridline-color: #edf0f5; selection-background-color: {ACCENT_SOFT}; selection-color: {INK}; }}
QHeaderView::section {{ background: #f0f3f8; border: none; border-bottom: 1px solid {LINE}; padding: 5px 8px; font-weight: 600; color: {MUTED}; }}
QTableCornerButton::section {{ background: #f0f3f8; border: none; }}

QProgressBar {{ border: none; background: #dfe5ee; border-radius: 6px; text-align: center; min-height: 12px; }}
QProgressBar::chunk {{ background: {ACCENT}; border-radius: 6px; }}
QSplitter::handle {{ background: transparent; }}
QStatusBar {{ background: {CARD}; border-top: 1px solid {LINE}; color: {MUTED}; }}
QStatusBar::item {{ border: none; }}
QTextBrowser {{ background: {CARD}; border: 1px solid {LINE}; border-radius: 8px; padding: 6px; }}

QLabel#title {{ font-size: 19pt; font-weight: 700; color: #14324a; }}
QLabel#subtitle {{ color: {MUTED}; font-size: 10pt; }}
QLabel#concept {{ background: #eef5fb; border: none; border-left: 4px solid {ACCENT}; border-radius: 6px; padding: 10px 12px; }}
QLabel#badge_ok {{ background: #e3f4ea; border: 1px solid #9ad2ae; border-radius: 10px; padding: 3px 10px; color: #14532d; }}
QLabel#badge_warn {{ background: #fde8e8; border: 1px solid #e59a9a; border-radius: 10px; padding: 3px 10px; font-weight: 600; color: #7f1d1d; }}
QLabel#stale {{ background: #fff1cc; border: 1px solid #e0b440; border-radius: 9px; padding: 1px 8px; color: #6b4a00; }}
QLabel#empty {{ color: {MUTED}; font-size: 11pt; background: transparent; }}
QWidget#card {{ background: {CARD}; border: 1px solid {LINE}; border-radius: 10px; }}
"""


def apply_theme(app: QApplication) -> None:
    app.setStyle("Fusion")
    pal = QPalette()
    white, light, ink = QColor("#ffffff"), QColor("#f3f5f9"), QColor(INK)
    pal.setColor(QPalette.Window, light)
    pal.setColor(QPalette.WindowText, ink)
    pal.setColor(QPalette.Base, white)
    pal.setColor(QPalette.AlternateBase, QColor("#f0f0f0"))
    pal.setColor(QPalette.Text, ink)
    pal.setColor(QPalette.Button, QColor("#f4f4f4"))
    pal.setColor(QPalette.ButtonText, ink)
    pal.setColor(QPalette.ToolTipBase, white)
    pal.setColor(QPalette.ToolTipText, ink)
    pal.setColor(QPalette.Highlight, QColor("#0072B2"))
    pal.setColor(QPalette.HighlightedText, white)
    pal.setColor(QPalette.PlaceholderText, QColor("#666666"))
    app.setPalette(pal)
    app.setFont(QFont("Segoe UI", 10))
    app.setStyleSheet(QSS)
    matplotlib.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
        "axes.edgecolor": "#7b8794", "axes.linewidth": 0.9, "font.family": ["Segoe UI", "DejaVu Sans"], "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": "#3e4c59", "ytick.color": "#3e4c59", "axes.grid": False,
        "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8, "legend.fontsize": 7,
        "axes.spines.top": True, "axes.spines.right": True, "figure.dpi": 100,
    })
