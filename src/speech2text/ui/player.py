"""Playing a recording: one segment, or onwards from a point.

Playback goes through ffplay, which arrives with the ffmpeg this application
already requires. That means every format the application accepts can also be
listened to, with no second decoder and no extra dependency.

ffplay does not report where it has got to, so position is tracked from the
clock: audio advances in real time, multiplied by the chosen speed. That is
accurate enough to keep a transcript in step with what is being heard, and it
costs nothing.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from PySide6 import QtCore

#: Padding either side of a single segment. A clip cut exactly on the
#: boundary loses the first consonant, which is the sound most often misheard.
PADDING_SECONDS = 0.15

#: How often the position is recomputed while playing.
_TICK_MILLISECONDS = 80

#: Speeds offered. ffmpeg's atempo filter takes 0.5 to 2.0 in one pass, and
#: keeps the pitch, so a faster read does not turn into a chipmunk.
SPEEDS = (0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0)
DEFAULT_SPEED = 1.0


def playback_available() -> bool:
    return shutil.which("ffplay") is not None


class SegmentPlayer(QtCore.QObject):
    """Plays part of a file, and says where it has got to while it does."""

    started = QtCore.Signal()
    #: The clip reached its end on its own.
    finished = QtCore.Signal()
    #: Playing has ended, however it ended. Always follows ``finished``.
    stopped = QtCore.Signal()
    #: Seconds from the start of the recording, while playing.
    position_changed = QtCore.Signal(float)
    failed = QtCore.Signal(str)

    def __init__(self, parent: QtCore.QObject | None = None) -> None:
        super().__init__(parent)
        self.source: Path | None = None
        self.speed = DEFAULT_SPEED
        self._process: QtCore.QProcess | None = None
        self._began = QtCore.QElapsedTimer()
        self._from = 0.0
        self._until: float | None = None
        self._ticker = QtCore.QTimer(self)
        self._ticker.setInterval(_TICK_MILLISECONDS)
        self._ticker.timeout.connect(self._tick)

    # ---- source and speed -----------------------------------------------

    def set_source(self, path: Path | None) -> None:
        self.stop()
        self.source = Path(path) if path else None

    def set_speed(self, speed: float) -> None:
        """Change the speed. Takes effect on the next thing played."""
        self.speed = min(max(float(speed), SPEEDS[0]), SPEEDS[-1])

    @property
    def ready(self) -> bool:
        return bool(self.source and self.source.exists() and playback_available())

    @property
    def playing(self) -> bool:
        return self._process is not None

    @property
    def position(self) -> float:
        """Where in the recording playing has got to, in seconds."""
        if not self.playing:
            return self._from
        return self._from + (self._began.elapsed() / 1000.0) * self.speed

    # ---- what ffplay is told --------------------------------------------

    @staticmethod
    def arguments(
        source: Path,
        start: float,
        end: float | None = None,
        *,
        speed: float = DEFAULT_SPEED,
        padding: float = 0.0,
    ) -> list[str]:
        """Play this file from here, optionally stopping at a point.

        ``-autoexit`` ends the process at the end of the clip, which is what
        signals that playing has finished on its own.
        """
        begin = max(0.0, float(start) - padding)
        args = [
            "-nodisp", "-autoexit", "-hide_banner",
            "-loglevel", "error",
            "-ss", f"{begin:.3f}",
        ]
        if end is not None:
            args += ["-t", f"{max(0.25, float(end) + padding - begin):.3f}"]
        if abs(speed - 1.0) > 0.001:
            # atempo changes the tempo without changing the pitch.
            args += ["-af", f"atempo={speed:.3f}"]
        args.append(str(source))
        return args

    # ---- playing --------------------------------------------------------

    def play_segment(self, start: float, end: float) -> bool:
        """Play one stretch and stop at the end of it."""
        return self._start(start, end, padding=PADDING_SECONDS)

    def play_onwards(self, start: float) -> bool:
        """Play from here to the end of the recording, without stopping.

        This is what makes following along possible: one continuous play
        rather than a clip per segment, so there is no gap at every boundary.
        """
        return self._start(start, None, padding=0.0)

    def _start(self, start: float, end: float | None, padding: float) -> bool:
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
        process.setArguments(
            self.arguments(self.source, start, end, speed=self.speed, padding=padding)
        )
        process.finished.connect(self._on_finished)
        process.errorOccurred.connect(self._on_error)
        self._process = process
        self._from = max(0.0, start - padding)
        self._until = end
        self._began.restart()
        process.start()
        self._ticker.start()
        self.started.emit()
        return True

    def stop(self) -> None:
        """Stop playing. Does not report the clip as having finished."""
        self._ticker.stop()
        process, self._process = self._process, None
        if process is not None:
            process.finished.disconnect()
            process.errorOccurred.disconnect()
            process.kill()
            process.waitForFinished(1000)
            self._from = self.position
            self.stopped.emit()

    # ---- while it runs --------------------------------------------------

    @QtCore.Slot()
    def _tick(self) -> None:
        self.position_changed.emit(self.position)

    @QtCore.Slot()
    def _on_finished(self) -> None:
        self._ticker.stop()
        self._process = None
        if self._until is not None:
            self._from = self._until
        self.finished.emit()
        self.stopped.emit()

    @QtCore.Slot()
    def _on_error(self, error=None) -> None:
        self._ticker.stop()
        self._process = None
        self.failed.emit("the audio could not be played")
        self.stopped.emit()
