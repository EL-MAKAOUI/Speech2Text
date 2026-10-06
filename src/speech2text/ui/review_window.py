"""Checking a transcript against the recording, one segment at a time.

The work this window exists for: play a few seconds, read what was written
for it, and either fix it or say it is right. The queue puts the segments the
recognizer was least sure about first, so a two-hour recording becomes a
finite list rather than a re-listen.

Corrections never overwrite the recognition. What was heard stays in
`transcript.raw.txt` and in the artifact; an edit is stored beside it.
"""

from __future__ import annotations

from pathlib import Path

from PySide6 import QtCore, QtGui, QtWidgets

from .. import artifact, media
from ..export import clock
from ..model import Transcript
from .guide_window import REVIEW_SHORTCUTS
from .player import DEFAULT_SPEED, SPEEDS, SegmentPlayer, playback_available

#: Below this a segment is worth a listen; it matches the review queue.
_DOUBTFUL = 0.6


class DecodeWorker(QtCore.QThread):
    """Decodes the recording so any format can be played back."""

    done = QtCore.Signal(object)
    failed = QtCore.Signal(str)

    def __init__(self, source: Path, destination: Path,
                 parent: QtCore.QObject | None = None) -> None:
        super().__init__(parent)
        self._source = source
        self._destination = destination

    def run(self) -> None:  # pragma: no cover - exercised through the window
        try:
            self.done.emit(media.decode_to_wav(self._source, self._destination))
        except media.MediaError as exc:
            self.failed.emit(str(exc))


class ReviewWindow(QtWidgets.QMainWindow):
    """Listen, compare, correct."""

    COLUMNS = ("At", "Sure?", "Text", "")

    saved = QtCore.Signal()

    def __init__(self, directory: str | Path, parent: QtWidgets.QWidget | None = None,
                 audio: Path | None = None) -> None:
        super().__init__(parent)
        self.directory = Path(directory)
        self.transcript: Transcript = artifact.load(self.directory)
        self.player = SegmentPlayer(self)
        self._decoder: DecodeWorker | None = None
        self._workspace: QtCore.QTemporaryDir | None = None
        self._loading = False
        #: True while playing straight through rather than one segment.
        self._following = False
        #: Set while the highlight is moved by playback, so that moving it
        #: does not itself start a new clip.
        self._following_move = False

        self.setWindowTitle(f"Checking {self.transcript.media.name}")
        self.resize(1000, 720)
        self._build()
        self._bind_shortcuts()
        self._fill()
        self._prepare_audio(audio)
        self._restore_position()

    # ---- construction ---------------------------------------------------

    def _build(self) -> None:
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QVBoxLayout(central)

        layout.addLayout(self._build_header())

        splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        splitter.addWidget(self._build_list())
        splitter.addWidget(self._build_editor())
        splitter.setSizes([380, 320])
        layout.addWidget(splitter, 1)

        self.setStatusBar(QtWidgets.QStatusBar())
        self.player.failed.connect(self._on_playback_failed)
        self.player.started.connect(self._on_playing_changed)
        self.player.stopped.connect(self._on_playing_changed)
        self.player.position_changed.connect(self._on_position)
        self.player.finished.connect(self._on_clip_finished)

    def _build_header(self) -> QtWidgets.QHBoxLayout:
        row = QtWidgets.QHBoxLayout()
        self.heading = QtWidgets.QLabel()
        self.heading.setToolTip(
            "The recording being checked, and how much of it still wants a look."
        )
        row.addWidget(self.heading)
        row.addStretch(1)

        self.queue_button = QtWidgets.QPushButton("Next to check")
        self.queue_button.setToolTip(
            "Jump to the next segment the recognizer was unsure about, least "
            "confident first (Ctrl+J)."
        )
        self.queue_button.clicked.connect(self.go_to_next_issue)
        row.addWidget(self.queue_button)
        return row

    def _build_list(self) -> QtWidgets.QWidget:
        box = QtWidgets.QGroupBox("The recording, segment by segment")
        layout = QtWidgets.QVBoxLayout(box)
        self.segments = QtWidgets.QTreeWidget()
        self.segments.setHeaderLabels(list(self.COLUMNS))
        self.segments.setRootIsDecorated(False)
        self.segments.setToolTip(
            "Every stretch of speech found. Choose one to hear it and read "
            "what was written for it. A dot marks the ones worth checking."
        )
        self.segments.setColumnWidth(0, 90)
        self.segments.setColumnWidth(1, 70)
        self.segments.currentItemChanged.connect(self._on_segment_chosen)
        self.segments.itemDoubleClicked.connect(lambda *_: self.play())
        layout.addWidget(self.segments)
        return box

    def _build_editor(self) -> QtWidgets.QWidget:
        box = QtWidgets.QGroupBox("This segment")
        layout = QtWidgets.QVBoxLayout(box)

        controls = QtWidgets.QHBoxLayout()
        self.play_button = QtWidgets.QPushButton("▶ Play")
        self.play_button.setToolTip(
            "Listen to just this segment, so it can be compared with the text "
            "below (Ctrl+Space). Double-clicking a row does the same."
        )
        self.play_button.clicked.connect(self.toggle_play)
        controls.addWidget(self.play_button)

        self.again_button = QtWidgets.QPushButton("⟲ Again")
        self.again_button.setToolTip("Play the same few seconds again (Ctrl+R).")
        self.again_button.clicked.connect(self.play)
        controls.addWidget(self.again_button)

        self.follow_button = QtWidgets.QPushButton("▶▶ Play on")
        self.follow_button.setToolTip(
            "Keep playing from here through the rest of the recording, "
            "highlighting each segment as it is said. Use this to listen "
            "through and stop only where something is wrong (Ctrl+P)."
        )
        self.follow_button.clicked.connect(self.toggle_follow)
        controls.addWidget(self.follow_button)

        controls.addWidget(QtWidgets.QLabel("Speed"))
        self.speed_box = QtWidgets.QComboBox()
        for speed in SPEEDS:
            self.speed_box.addItem(f"{speed:g}×", speed)
        self.speed_box.setCurrentIndex(max(0, self.speed_box.findData(DEFAULT_SPEED)))
        self.speed_box.setToolTip(
            "How fast to play. The pitch stays the same, so a faster read is "
            "still clear; slower helps on a difficult passage."
        )
        self.speed_box.currentIndexChanged.connect(self._on_speed_changed)
        controls.addWidget(self.speed_box)

        self.position_label = QtWidgets.QLabel()
        self.position_label.setToolTip("Where this segment sits in the recording.")
        controls.addWidget(self.position_label)
        controls.addStretch(1)

        self.autoplay_box = QtWidgets.QCheckBox("Play as I move")
        self.autoplay_box.setToolTip(
            "Play each segment automatically when it is selected, so checking "
            "is listen, read, fix, next."
        )
        self.autoplay_box.setChecked(True)
        controls.addWidget(self.autoplay_box)
        layout.addLayout(controls)

        self.heard_label = QtWidgets.QLabel("What was heard")
        self.heard_label.setToolTip(
            "The recognizer's own words for this segment, kept whatever you "
            "type below."
        )
        layout.addWidget(self.heard_label)
        self.heard = QtWidgets.QPlainTextEdit()
        self.heard.setReadOnly(True)
        self.heard.setMaximumHeight(70)
        self.heard.setToolTip(
            "The recognizer's own words. This is never changed — your edit is "
            "kept beside it, so the original can always be recovered."
        )
        layout.addWidget(self.heard)

        layout.addWidget(QtWidgets.QLabel("Your text"))
        self.editor = QtWidgets.QPlainTextEdit()
        self.editor.setMaximumHeight(90)
        self.editor.setToolTip(
            "Fix the text here, then press Save correction (Ctrl+S)."
        )
        self.editor.textChanged.connect(self._on_edited)
        layout.addWidget(self.editor)

        buttons = QtWidgets.QHBoxLayout()
        self.save_button = QtWidgets.QPushButton("Save correction")
        self.save_button.setToolTip(
            "Store what you typed for this segment and move on (Ctrl+S)."
        )
        self.save_button.clicked.connect(self.save_correction)
        buttons.addWidget(self.save_button)

        self.accept_button = QtWidgets.QPushButton("It's correct")
        self.accept_button.setToolTip(
            "The recognizer was unsure but got it right. Clears it from the "
            "queue without changing the text (Ctrl+K)."
        )
        self.accept_button.clicked.connect(self.accept_segment)
        buttons.addWidget(self.accept_button)

        self.revert_button = QtWidgets.QPushButton("Undo my change")
        self.revert_button.setToolTip(
            "Put back exactly what the recognizer heard for this segment."
        )
        self.revert_button.clicked.connect(self.revert_segment)
        buttons.addWidget(self.revert_button)

        buttons.addStretch(1)
        self.previous_button = QtWidgets.QPushButton("◀")
        self.previous_button.setToolTip("Previous segment (Ctrl+Left).")
        self.previous_button.clicked.connect(lambda: self.step(-1))
        buttons.addWidget(self.previous_button)
        self.next_button = QtWidgets.QPushButton("▶")
        self.next_button.setToolTip("Next segment (Ctrl+Right).")
        self.next_button.clicked.connect(lambda: self.step(1))
        buttons.addWidget(self.next_button)
        layout.addLayout(buttons)
        return box

    def _bind_shortcuts(self) -> None:
        for key, action in (
            ("Ctrl+Space", self.toggle_play),
            ("Ctrl+P", self.toggle_follow),
            ("Ctrl+R", self.play),
            ("Ctrl+S", self.save_correction),
            ("Ctrl+K", self.accept_segment),
            ("Ctrl+Right", lambda: self.step(1)),
            ("Ctrl+Left", lambda: self.step(-1)),
            ("Ctrl+J", self.go_to_next_issue),
        ):
            assert key in REVIEW_SHORTCUTS, f"{key} is bound but not in the guide"
            QtGui.QShortcut(QtGui.QKeySequence(key), self, activated=action)

    # ---- filling --------------------------------------------------------

    def _fill(self) -> None:
        self.segments.clear()
        for segment in self.transcript.segments:
            item = QtWidgets.QTreeWidgetItem(["", "", "", ""])
            item.setData(0, QtCore.Qt.UserRole, segment.index)
            self.segments.addTopLevelItem(item)
            self._refresh_row(segment.index)
        self._refresh_heading()

    def _refresh_row(self, index: int) -> None:
        item = self.segments.topLevelItem(index)
        if item is None:
            return
        segment = self.transcript.segments[index]
        corrected = index in self.transcript.corrected_segment_indexes()
        open_issue = any(
            issue.segment_index == index and not issue.resolved
            for issue in self.transcript.issues
        )
        item.setText(0, clock(segment.start))
        item.setText(1, self._confidence_text(segment.confidence))
        item.setText(2, self.transcript.text_of(index).strip() or "(nothing heard)")
        item.setText(3, "edited" if corrected else ("check" if open_issue else ""))
        item.setToolTip(2, f"{clock(segment.start)} to {clock(segment.end)}")

    @staticmethod
    def _confidence_text(confidence: float | None) -> str:
        if confidence is None:
            return "—"
        mark = "●" if confidence < _DOUBTFUL else "○"
        return f"{mark} {confidence * 100:.0f}%"

    def _refresh_heading(self) -> None:
        remaining = len(self.transcript.unresolved_issues())
        corrections = len(self.transcript.corrected_segment_indexes())
        parts = [
            f"<b>{self.transcript.media.name}</b>",
            f"{len(self.transcript.segments)} segments",
            media.format_duration(self.transcript.duration),
        ]
        if remaining:
            parts.append(f"<b>{remaining}</b> to check")
        else:
            parts.append("nothing left to check")
        if corrections:
            parts.append(f"{corrections} corrected")
        self.heading.setText(" · ".join(parts))
        self.queue_button.setEnabled(bool(remaining))

    # ---- audio ----------------------------------------------------------

    def _prepare_audio(self, audio: Path | None) -> None:
        """Find something playable: the kept audio, or the original decoded."""
        if audio and Path(audio).exists():
            self.player.set_source(Path(audio))
            self._set_playable(True)
            return

        kept = self.directory / "audio.wav"
        if kept.exists():
            self.player.set_source(kept)
            self._set_playable(True)
            return

        source = Path(self.transcript.media.path)
        if not source.exists():
            self._set_playable(
                False,
                f"The recording is no longer at {source}, so it cannot be "
                f"played. The text can still be corrected.",
            )
            return
        if not playback_available():
            self._set_playable(
                False,
                "ffplay is not installed, so the audio cannot be played. It "
                "comes with ffmpeg: sudo apt install ffmpeg",
            )
            return

        self._set_playable(False, "Preparing the audio…")
        self._workspace = QtCore.QTemporaryDir()
        self._decoder = DecodeWorker(
            source, Path(self._workspace.path()) / "audio.wav", self
        )
        self._decoder.done.connect(self._on_decoded)
        self._decoder.failed.connect(
            lambda reason: self._set_playable(False, f"The audio could not be read: {reason}")
        )
        self._decoder.start()

    @QtCore.Slot(object)
    def _on_decoded(self, path: Path) -> None:
        self.player.set_source(path)
        self._set_playable(True)
        self.statusBar().showMessage("Ready. Choose a segment to hear it.")

    def _set_playable(self, playable: bool, message: str = "") -> None:
        self.play_button.setEnabled(playable)
        self.again_button.setEnabled(playable)
        self.follow_button.setEnabled(playable)
        self.speed_box.setEnabled(playable)
        self.autoplay_box.setEnabled(playable)
        if message:
            self.statusBar().showMessage(message)

    # ---- moving about ---------------------------------------------------

    @property
    def current_index(self) -> int | None:
        item = self.segments.currentItem()
        if item is None:
            return None
        return item.data(0, QtCore.Qt.UserRole)

    def select(self, index: int, play: bool | None = None) -> None:
        if not 0 <= index < len(self.transcript.segments):
            return
        self.segments.setCurrentItem(self.segments.topLevelItem(index))
        if play or (play is None and self.autoplay_box.isChecked()):
            self.play()

    def step(self, by: int) -> None:
        index = self.current_index
        if index is None:
            self.select(0)
            return
        self.select(index + by)

    def go_to_next_issue(self) -> None:
        """The least confident segment still unresolved, wherever it is."""
        queue = self.transcript.unresolved_issues()
        if not queue:
            self.statusBar().showMessage("Nothing left to check.")
            return
        here = self.current_index
        following = [i for i in queue if here is None or i.segment_index != here]
        self.select((following or queue)[0].segment_index)

    @QtCore.Slot()
    def _on_segment_chosen(self, current=None, previous=None) -> None:
        index = self.current_index
        if index is None:
            return
        segment = self.transcript.segments[index]
        self._loading = True
        self.heard.setPlainText(segment.text)
        self.editor.setPlainText(self.transcript.text_of(index))
        self._loading = False

        self.position_label.setText(
            f"{clock(segment.start)} – {clock(segment.end)} "
            f"({segment.duration:.1f}s)"
        )
        corrected = index in self.transcript.corrected_segment_indexes()
        self.revert_button.setEnabled(corrected)
        self.heard_label.setText(
            "What was heard (kept, and never overwritten)" if corrected
            else "What was heard"
        )
        self.save_button.setEnabled(False)
        artifact.remember_review_position(self.directory, index)
        if self._following_move or self.following:
            # The highlight is following what is being played; starting
            # another clip here would fight with it.
            return
        if self.autoplay_box.isChecked() and self.autoplay_box.isEnabled():
            self.play()

    def _restore_position(self) -> None:
        """Come back to where checking got to, rather than the top."""
        saved = artifact.load_project(self.directory).get("review_position", 0)
        try:
            position = int(saved)
        except (TypeError, ValueError):
            position = 0
        self.select(min(max(0, position), max(0, len(self.transcript.segments) - 1)),
                    play=False)

    # ---- playing --------------------------------------------------------

    def play(self) -> None:
        """Play just this segment and stop at the end of it."""
        index = self.current_index
        if index is None:
            return
        self._following = False
        segment = self.transcript.segments[index]
        self.player.play_segment(segment.start, segment.end)

    def toggle_play(self) -> None:
        if self.player.playing:
            self.player.stop()
        else:
            self.play()

    def play_onwards(self) -> None:
        """Keep playing from here, following the text as it goes.

        One continuous play rather than a clip per segment: a recording with
        nothing wrong in it can be listened through without a gap at every
        boundary, stopping only where something needs fixing.
        """
        index = self.current_index
        if index is None:
            index = 0
            self.select(0, play=False)
        segment = self.transcript.segments[index]
        self._following = True
        if not self.player.play_onwards(segment.start):
            self._following = False

    def toggle_follow(self) -> None:
        if self.player.playing and self._following:
            self.player.stop()
        else:
            self.play_onwards()

    @property
    def following(self) -> bool:
        return self._following and self.player.playing

    def _on_speed_changed(self) -> None:
        speed = self.speed_box.currentData() or DEFAULT_SPEED
        self.player.set_speed(speed)
        if self.player.playing:
            # Take effect now rather than at the next segment.
            resume_from = self.player.position
            following = self._following
            self.player.stop()
            if following:
                self._following = True
                self.player.play_onwards(resume_from)
            else:
                self.play()

    @QtCore.Slot()
    def _on_playing_changed(self) -> None:
        playing = self.player.playing
        self.play_button.setText("■ Stop" if playing and not self._following else "▶ Play")
        self.follow_button.setText(
            "■ Stop" if playing and self._following else "▶▶ Play on"
        )
        if not playing:
            self._following = False

    @QtCore.Slot(float)
    def _on_position(self, position: float) -> None:
        """Keep the highlight on whatever is being said."""
        if not self._following:
            return
        index = self.segment_at(position)
        if index is None or index == self.current_index:
            return
        self._following_move = True
        try:
            self.segments.setCurrentItem(self.segments.topLevelItem(index))
        finally:
            self._following_move = False

    def segment_at(self, position: float) -> int | None:
        """Which segment covers this moment, or the next one still to come."""
        for segment in self.transcript.segments:
            if segment.start <= position < segment.end:
                return segment.index
        upcoming = [s for s in self.transcript.segments if s.start >= position]
        return upcoming[0].index if upcoming else None

    @QtCore.Slot()
    def _on_clip_finished(self) -> None:
        if self._following:
            self._following = False
            self.statusBar().showMessage("Reached the end of the recording.")

    @QtCore.Slot(str)
    def _on_playback_failed(self, message: str) -> None:
        self.statusBar().showMessage(message)

    # ---- editing --------------------------------------------------------

    def _on_edited(self) -> None:
        if self._loading:
            return
        index = self.current_index
        changed = (
            index is not None
            and self.editor.toPlainText() != self.transcript.text_of(index)
        )
        self.save_button.setEnabled(changed)

    def save_correction(self) -> bool:
        index = self.current_index
        if index is None:
            return False
        text = self.editor.toPlainText().strip()
        if text == self.transcript.text_of(index):
            self.statusBar().showMessage("Nothing changed for this segment.")
            return False
        self.transcript.correct(index, text)
        self._persist()
        self._refresh_row(index)
        self._refresh_heading()
        self.save_button.setEnabled(False)
        self.revert_button.setEnabled(True)
        self.statusBar().showMessage(f"Saved. {self._remaining_message()}")
        return True

    def accept_segment(self) -> bool:
        """The recognizer was unsure, but it is right. Clears it, changes nothing."""
        index = self.current_index
        if index is None:
            return False
        self.transcript.resolve_issue(index, accepted=True)
        self._persist()
        self._refresh_row(index)
        self._refresh_heading()
        self.statusBar().showMessage(f"Marked correct. {self._remaining_message()}")
        return True

    def revert_segment(self) -> bool:
        """Put back what was heard, by correcting it back to the recognition."""
        index = self.current_index
        if index is None:
            return False
        if index not in self.transcript.corrected_segment_indexes():
            return False
        original = self.transcript.segments[index].text
        self.transcript.correct(index, original, note="reverted to the recognition")
        self._persist()
        self._loading = True
        self.editor.setPlainText(original)
        self._loading = False
        self._refresh_row(index)
        self._refresh_heading()
        self.statusBar().showMessage("Put back what was heard.")
        return True

    def _remaining_message(self) -> str:
        remaining = len(self.transcript.unresolved_issues())
        return f"{remaining} left to check." if remaining else "Nothing left to check."

    def _persist(self) -> None:
        """Write the bundle, keeping the recognition file exactly as it was."""
        project = artifact.load_project(self.directory)
        artifact.write_bundle(self.transcript, self.directory, project=project or None)
        if self.current_index is not None:
            artifact.remember_review_position(self.directory, self.current_index)
        self.saved.emit()

    # ---- closing --------------------------------------------------------

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        self.player.stop()
        if self._decoder is not None and self._decoder.isRunning():
            self._decoder.wait(2000)
        if self.current_index is not None:
            artifact.remember_review_position(self.directory, self.current_index)
        super().closeEvent(event)
