"""Background tasks for the GUI. Heavy fitting never runs on the GUI thread."""
from __future__ import annotations

import logging
import traceback

from PySide6.QtCore import QObject, QRunnable, QThreadPool, Signal

log = logging.getLogger("msp")


class TaskSignals(QObject):
    done = Signal(int, object)
    failed = Signal(int, str)


class FitTask(QRunnable):
    """Runs ``run.fit_initial()`` in a pool thread."""

    def __init__(self, token: int, model_run):
        super().__init__()
        self.token, self.job = token, model_run
        self.signals = TaskSignals()

    def run(self):
        try:
            self.job.fit_initial()
            self.signals.done.emit(self.token, self.job)
        except Exception:  # reported to the user by the receiver
            tb = traceback.format_exc()
            log.error("Fit failed:\n%s", tb)
            self.signals.failed.emit(self.token, tb)


def pool() -> QThreadPool:
    return QThreadPool.globalInstance()
