"""Playing one stretch of a recording, for checking it against the text.

Playback goes through ffplay, which arrives with the ffmpeg this application
already requires. That means every format the application accepts can also be
listened to, with no second decoder and no extra dependency.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from PySide6 import QtCore

#: Padding either side of a segment. A clip cut exactly on the boundary
#: clips the first consonant, which is the sound most often misheard.
PADDING_SECONDS = 0.15


def playback_available() -> bool:
    return shutil.which("ffplay") is not None


class SegmentPlayer(QtCore.QObject):
    """Plays a time range of one file, and stops when asked or when it ends."""

    started = QtCore.Signal()
    stopped = QtCore.Signal()
    failed = QtCore.Signal(str)

    def __init__(self, parent: QtCore.QObject | None = None) -> None:
        super().__init__(parent)
        self.source: Path | None = None
        self._process: QtCore.QProcess | None = None
        self._range: tuple[float, float] = (0.0, 0.0)

    # ---- source ---------------------------------------------------------

    def set_source(self, path: Path | None) -> None:
        self.stop()
        self.source = Path(path) if path else None

    @property
    def ready(self) -> bool:
        return bool(self.source and self.source.exists() and playback_available())

    @property
    def playing(self) -> bool:
        return self._process is not None

    @property
    def current_range(self) -> tuple[float, float]:
        return self._range

    # ---- playing --------------------------------------------------------

    @staticmethod
    def arguments(source: Path, start: float, end: float,
                  padding: float = PADDING_SECONDS) -> list[str]:
        """What ffplay is told: this file, from here, for this long.

        ``-autoexit`` ends the process at the end of the clip, which is what
        signals that playing has finished.
        """
        begin = max(0.0, float(start) - padding)
        length = max(0.25, float(end) + padding - begin)
        return [
            "-nodisp", "-autoexit", "-hide_banner",
            "-loglevel", "error",
            "-ss", f"{begin:.3f}",
            "-t", f"{length:.3f}",
            str(source),
        ]

    def play(self, start: float, end: float) -> bool:
        """Play one stretch. Returns whether it started."""
        self.stop()
        if not playback_available():
            self.failed.emit(
                "ffplay is not installed, so the audio cannot be played. "
                "It comes with ffmpeg: sudo apt install ffmpeg"
            )
            return False
        if not self.source or not self.source.exists():
            self.failed.emit("the audio for this recording is not available")
            return False

        process = QtCore.QProcess(self)
        process.setProgram("ffplay")
        process.setArguments(self.arguments(self.source, start, end))
        process.finished.connect(self._on_finished)
        process.errorOccurred.connect(self._on_error)
        self._process = process
        self._range = (start, end)
        process.start()
        self.started.emit()
        return True

    def stop(self) -> None:
        process, self._process = self._process, None
        if process is not None:
            process.finished.disconnect()
            process.errorOccurred.disconnect()
            process.kill()
            process.waitForFinished(1000)
            self.stopped.emit()

    # ---- process --------------------------------------------------------

    @QtCore.Slot()
    def _on_finished(self) -> None:
        self._process = None
        self.stopped.emit()

    @QtCore.Slot()
    def _on_error(self, error=None) -> None:
        self._process = None
        self.failed.emit("the audio could not be played")
        self.stopped.emit()
