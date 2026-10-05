"""The Speech2Text window.

Every control carries a tooltip saying what it is for and what happens as a
result. The guide behind **? Guide** is generated from
:mod:`speech2text.ui.guide_window`, and the tests check it against this code,
so what the window says and what it does cannot drift apart.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence

from PySide6 import QtCore, QtGui, QtWidgets

from .. import artifact, export, languages, media, summarize
from ..engines import DEFAULT_ENGINE, EngineError, create, describe_all
from ..engines.whisper import DEFAULT_MODEL, MODEL_SIZES
from ..model import Transcript
from ..pipeline import Cancelled, Progress, Run, TranscribeOptions
from .guide_window import SHORTCUTS, open_guide

DEFAULT_OUTPUT = Path.home() / "Speech2Text"

_BADGE_LOCAL = "LOCAL • PRIVATE"
_BADGE_CLOUD = "CLOUD • AUDIO IS UPLOADED"

_STATUS_WAITING = "waiting"
_STATUS_DONE = "done"
_STATUS_FAILED = "failed"


# ---------------------------------------------------------------------------
# background work
# ---------------------------------------------------------------------------


class TranscribeWorker(QtCore.QThread):
    """Runs the pipeline off the interface thread, one file at a time."""

    advanced = QtCore.Signal(int, object)       # row, Progress
    completed = QtCore.Signal(int, object)      # row, Bundle
    failed = QtCore.Signal(int, str)            # row, message
    finished_all = QtCore.Signal()

    def __init__(self, jobs: list[tuple[int, Path]], run_root: Path,
                 options: TranscribeOptions) -> None:
        super().__init__()
        self._jobs = jobs
        self._run_root = run_root
        self._options = options
        self._current: Run | None = None
        self._stopped = False

    def stop(self) -> None:
        self._stopped = True
        if self._current is not None:
            self._current.stop()

    def run(self) -> None:  # pragma: no cover - exercised through the window
        for position, (row, source) in enumerate(self._jobs, start=1):
            if self._stopped:
                break
            destination = artifact.allocate(self._run_root, source.name, position)
            run = Run(
                source,
                destination,
                self._options,
                on_progress=lambda progress, row=row: self.advanced.emit(row, progress),
            )
            self._current = run
            try:
                self.completed.emit(row, run.execute())
            except Cancelled:
                break
            except (media.MediaError, EngineError) as exc:
                self.failed.emit(row, str(exc))
            except Exception as exc:  # unexpected, but must not kill the window
                self.failed.emit(row, f"unexpected failure: {exc}")
        self._current = None
        self.finished_all.emit()


class SummaryWorker(QtCore.QThread):
    """Runs a language-model task on a finished transcript."""

    progressed = QtCore.Signal(int, int)
    completed = QtCore.Signal(object)
    failed = QtCore.Signal(str)

    def __init__(self, transcript: Transcript, task, provider: str,
                 directory: Path | None) -> None:
        super().__init__()
        self._transcript = transcript
        self._task = task
        self._provider = provider
        self._directory = directory

    def run(self) -> None:  # pragma: no cover - exercised through the window
        try:
            result = summarize.summarize_transcript(
                self._transcript,
                self._task,
                provider=self._provider,
                on_progress=lambda n, total: self.progressed.emit(n, total),
            )
        except summarize.SummaryError as exc:
            self.failed.emit(str(exc))
            return
        except Exception as exc:
            self.failed.emit(f"unexpected failure: {exc}")
            return
        if self._directory is not None:
            try:
                summarize.write_result(
                    result, self._directory, self._transcript.media.name
                )
            except OSError:
                pass
        self.completed.emit(result)


# ---------------------------------------------------------------------------
# the window
# ---------------------------------------------------------------------------


class MainWindow(QtWidgets.QMainWindow):
    """Add recordings, transcribe them, take the text away."""

    COLUMNS = ("Recording", "What it is", "Length", "Status", "Words", "To check")

    def __init__(self, inputs: Sequence[str] = ()) -> None:
        super().__init__()
        self.setWindowTitle("Speech2Text")
        self.resize(1040, 760)
        self.setAcceptDrops(True)

        self._bundles: dict[int, artifact.Bundle] = {}
        self._worker: TranscribeWorker | None = None
        self._summary_worker: SummaryWorker | None = None
        self._summary_text = ""
        self._output = DEFAULT_OUTPUT

        self._build()
        self._bind_shortcuts()
        self._refresh_engine_controls()
        self.add_files([Path(p) for p in inputs])
        self._update_buttons()

    # ---- construction ---------------------------------------------------

    def _build(self) -> None:
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QVBoxLayout(central)
        layout.setSpacing(10)

        layout.addLayout(self._build_header())
        layout.addWidget(self._build_files(), 3)
        layout.addWidget(self._build_settings())
        layout.addLayout(self._build_actions())
        layout.addWidget(self._build_results(), 4)
        self.setStatusBar(QtWidgets.QStatusBar())
        self.statusBar().showMessage("Add a recording to begin.")

    def _build_header(self) -> QtWidgets.QHBoxLayout:
        row = QtWidgets.QHBoxLayout()
        title = QtWidgets.QLabel("<b>Speech2Text</b>")
        subtitle = QtWidgets.QLabel(
            "Any audio or video file, whatever it is called."
        )
        subtitle.setStyleSheet("color: palette(mid);")
        row.addWidget(title)
        row.addWidget(subtitle)
        row.addStretch(1)

        self.badge = QtWidgets.QLabel(_BADGE_LOCAL)
        self.badge.setToolTip(
            "Whether your audio stays on this machine. It says CLOUD only when "
            "a cloud recognizer is selected, which uploads what you transcribe."
        )
        self._style_badge(cloud=False)
        row.addWidget(self.badge)

        guide = QtWidgets.QPushButton("? Guide")
        guide.setToolTip("How to transcribe, choose a language, and get the text out (F1).")
        guide.clicked.connect(lambda: open_guide(self))
        row.addWidget(guide)
        return row

    def _style_badge(self, cloud: bool) -> None:
        self.badge.setText(_BADGE_CLOUD if cloud else _BADGE_LOCAL)
        colour = "#b3261e" if cloud else "#1f6f43"
        self.badge.setStyleSheet(
            f"color: white; background: {colour}; padding: 3px 9px; border-radius: 3px;"
        )

    def _build_files(self) -> QtWidgets.QWidget:
        box = QtWidgets.QGroupBox("Recordings")
        layout = QtWidgets.QVBoxLayout(box)

        self.files = QtWidgets.QTreeWidget()
        self.files.setHeaderLabels(list(self.COLUMNS))
        self.files.setRootIsDecorated(False)
        self.files.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.files.setToolTip(
            "The recordings to transcribe. Drop files here from the file "
            "manager, or press Add files. The extension is ignored — what "
            "matters is what is inside the file."
        )
        self.files.itemSelectionChanged.connect(self._on_selection_changed)
        self.files.setColumnWidth(0, 300)
        layout.addWidget(self.files)

        buttons = QtWidgets.QHBoxLayout()
        self.add_button = QtWidgets.QPushButton("Add files…")
        self.add_button.setToolTip(
            "Choose recordings to transcribe. Several at once is fine (Ctrl+O)."
        )
        self.add_button.clicked.connect(self._choose_files)
        buttons.addWidget(self.add_button)

        self.remove_button = QtWidgets.QPushButton("Remove")
        self.remove_button.setToolTip("Take the selected recordings off the list.")
        self.remove_button.clicked.connect(self._remove_selected)
        buttons.addWidget(self.remove_button)

        self.clear_button = QtWidgets.QPushButton("Clear")
        self.clear_button.setToolTip("Empty the list and start again.")
        self.clear_button.clicked.connect(self._clear_files)
        buttons.addWidget(self.clear_button)
        buttons.addStretch(1)
        layout.addLayout(buttons)
        return box

    def _build_settings(self) -> QtWidgets.QWidget:
        box = QtWidgets.QGroupBox("Settings")
        grid = QtWidgets.QGridLayout(box)

        grid.addWidget(QtWidgets.QLabel("Recognition"), 0, 0)
        self.engine_box = QtWidgets.QComboBox()
        for described in describe_all():
            label = described["title"]
            if not described["available"]:
                label += "  (not ready)"
            self.engine_box.addItem(label, described["name"])
            index = self.engine_box.count() - 1
            self.engine_box.setItemData(
                index,
                described["summary"]
                + ("\n\n" + described["reason"] if described["reason"] else ""),
                QtCore.Qt.ToolTipRole,
            )
        self.engine_box.setCurrentIndex(
            max(0, self.engine_box.findData(DEFAULT_ENGINE))
        )
        self.engine_box.setToolTip(
            "Which recognizer reads the audio. Standard runs on this machine. "
            "Cloud is more accurate and uploads your audio."
        )
        self.engine_box.currentIndexChanged.connect(self._refresh_engine_controls)
        grid.addWidget(self.engine_box, 0, 1)

        grid.addWidget(QtWidgets.QLabel("Model"), 0, 2)
        self.model_box = QtWidgets.QComboBox()
        for size in MODEL_SIZES:
            self.model_box.addItem(size, size)
        self.model_box.setCurrentIndex(max(0, self.model_box.findData(DEFAULT_MODEL)))
        self.model_box.setToolTip(
            "Bigger models are more accurate and slower. Each one downloads "
            "once, the first time it is used, and then works offline."
        )
        grid.addWidget(self.model_box, 0, 3)

        grid.addWidget(QtWidgets.QLabel("Language"), 1, 0)
        self.language_box = QtWidgets.QComboBox()
        self.language_box.addItem("Detect automatically", "")
        for language in languages.choices():
            self.language_box.addItem(language.label, language.code)
        self.language_box.setToolTip(
            "Leave it on Detect automatically unless you know the language. "
            "Detection listens to the opening, so music or a greeting in "
            "another language can mislead it."
        )
        grid.addWidget(self.language_box, 1, 1)

        grid.addWidget(QtWidgets.QLabel("Save into"), 1, 2)
        folder_row = QtWidgets.QHBoxLayout()
        self.output_label = QtWidgets.QLineEdit(str(self._output))
        self.output_label.setReadOnly(True)
        self.output_label.setToolTip(
            "Where transcripts are written. Each recording gets its own folder."
        )
        folder_row.addWidget(self.output_label)
        choose = QtWidgets.QPushButton("Change…")
        choose.setToolTip("Pick a different folder for the transcripts.")
        choose.clicked.connect(self._choose_output)
        folder_row.addWidget(choose)
        container = QtWidgets.QWidget()
        container.setLayout(folder_row)
        grid.addWidget(container, 1, 3)

        self.timestamps_box = QtWidgets.QCheckBox("Include times in the text")
        self.timestamps_box.setToolTip(
            "Put [0:14] in front of each line when copying or saving as text "
            "or Word. Subtitles always carry their own timings."
        )
        grid.addWidget(self.timestamps_box, 2, 1)
        return box

    def _build_actions(self) -> QtWidgets.QHBoxLayout:
        row = QtWidgets.QHBoxLayout()
        self.transcribe_button = QtWidgets.QPushButton("Transcribe")
        self.transcribe_button.setToolTip(
            "Transcribe the selected recordings, or all of them if none is "
            "selected (Ctrl+Return)."
        )
        self.transcribe_button.clicked.connect(self.start)
        row.addWidget(self.transcribe_button)

        self.stop_button = QtWidgets.QPushButton("Stop")
        self.stop_button.setToolTip(
            "Stop after the current step. Finished recordings are kept (Esc)."
        )
        self.stop_button.clicked.connect(self.stop)
        self.stop_button.setEnabled(False)
        row.addWidget(self.stop_button)

        self.progress = QtWidgets.QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setToolTip("How far the current recording has got.")
        row.addWidget(self.progress, 1)
        return row

    def _build_results(self) -> QtWidgets.QWidget:
        tabs = QtWidgets.QTabWidget()
        tabs.addTab(self._build_text_tab(), "Transcript")
        tabs.addTab(self._build_summary_tab(), "After transcribing")
        tabs.setToolTip(
            "Transcript holds the text itself. After transcribing is the "
            "optional step that hands that text to a language model."
        )
        self.tabs = tabs
        return tabs

    def _build_text_tab(self) -> QtWidgets.QWidget:
        page = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(page)

        self.text_view = QtWidgets.QPlainTextEdit()
        self.text_view.setReadOnly(True)
        self.text_view.setPlaceholderText(
            "The text appears here once a recording has been transcribed."
        )
        self.text_view.setToolTip(
            "The transcript of the selected recording. Copy it or save it "
            "with the buttons below."
        )
        layout.addWidget(self.text_view)

        row = QtWidgets.QHBoxLayout()
        self.copy_button = QtWidgets.QPushButton("Copy text")
        self.copy_button.setToolTip(
            "Put the whole transcript on the clipboard, ready to paste (Ctrl+C)."
        )
        self.copy_button.clicked.connect(self.copy_text)
        row.addWidget(self.copy_button)

        self.save_button = QtWidgets.QPushButton("Save as…")
        self.save_button.setToolTip(
            "Write the transcript as a file: Word, plain text, subtitles, "
            "Markdown, CSV or JSON (Ctrl+S)."
        )
        self.save_button.clicked.connect(self.save_as)
        row.addWidget(self.save_button)

        self.folder_button = QtWidgets.QPushButton("Open folder")
        self.folder_button.setToolTip(
            "Open the folder holding this transcript and its other files."
        )
        self.folder_button.clicked.connect(self.open_folder)
        row.addWidget(self.folder_button)
        row.addStretch(1)
        layout.addLayout(row)
        return page

    def _build_summary_tab(self) -> QtWidgets.QWidget:
        page = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(page)

        note = QtWidgets.QLabel(
            "Optional. Hands the finished <b>text</b> to a language model. "
            "Your transcript is already complete without this."
        )
        note.setWordWrap(True)
        note.setStyleSheet("color: palette(mid);")
        layout.addWidget(note)

        controls = QtWidgets.QHBoxLayout()
        controls.addWidget(QtWidgets.QLabel("Task"))
        self.task_box = QtWidgets.QComboBox()
        for task in summarize.TASKS.values():
            self.task_box.addItem(task.title, task.name)
        self.task_box.addItem("My own instruction…", "custom")
        self.task_box.setToolTip(
            "What to do with the transcript: summarize it, pull out the "
            "decisions, tidy it up, or anything you type yourself."
        )
        self.task_box.currentIndexChanged.connect(self._refresh_summary_controls)
        controls.addWidget(self.task_box)

        controls.addWidget(QtWidgets.QLabel("Provider"))
        self.provider_box = QtWidgets.QComboBox()
        for name in sorted(summarize.PROVIDERS):
            self.provider_box.addItem(name, name)
        self.provider_box.setToolTip(
            "Which service to send the text to. It needs an API key; see the "
            "guide for where to get one."
        )
        controls.addWidget(self.provider_box)

        self.summary_button = QtWidgets.QPushButton("Run")
        self.summary_button.setToolTip(
            "Send the selected recording's text and show what comes back."
        )
        self.summary_button.clicked.connect(self.run_summary)
        controls.addWidget(self.summary_button)
        controls.addStretch(1)
        layout.addLayout(controls)

        self.instruction_box = QtWidgets.QLineEdit()
        self.instruction_box.setPlaceholderText(
            "e.g. list every question that was asked, and who asked it"
        )
        self.instruction_box.setToolTip(
            "Your own instruction, used instead of a preset task."
        )
        self.instruction_box.hide()
        layout.addWidget(self.instruction_box)

        self.summary_view = QtWidgets.QPlainTextEdit()
        self.summary_view.setReadOnly(True)
        self.summary_view.setPlaceholderText(
            "The result appears here, and is saved beside the transcript."
        )
        self.summary_view.setToolTip(
            "What the model returned. It is also written as a Markdown file "
            "in the recording's folder, so it is not lost when you close this."
        )
        layout.addWidget(self.summary_view)

        row = QtWidgets.QHBoxLayout()
        self.copy_summary_button = QtWidgets.QPushButton("Copy result")
        self.copy_summary_button.setToolTip("Put the result on the clipboard.")
        self.copy_summary_button.clicked.connect(self.copy_summary)
        row.addWidget(self.copy_summary_button)
        row.addStretch(1)
        layout.addLayout(row)
        return page

    def _bind_shortcuts(self) -> None:
        for key, action in (
            ("Ctrl+O", self._choose_files),
            ("Ctrl+Return", self.start),
            ("Ctrl+C", self.copy_text),
            ("Ctrl+S", self.save_as),
            ("Ctrl+V", self.paste_files),
            ("Esc", self.stop),
            ("F1", lambda: open_guide(self)),
        ):
            assert key in SHORTCUTS, f"{key} is bound but not in the guide"
            QtGui.QShortcut(QtGui.QKeySequence(key), self, activated=action)

    # ---- files ----------------------------------------------------------

    def add_files(self, paths: Sequence[Path]) -> int:
        """Add recordings, describing each one by looking inside it."""
        added = 0
        existing = {
            self.files.topLevelItem(i).data(0, QtCore.Qt.UserRole)
            for i in range(self.files.topLevelItemCount())
        }
        for path in paths:
            resolved = str(Path(path).expanduser().resolve())
            if resolved in existing:
                continue
            item = QtWidgets.QTreeWidgetItem(
                [Path(resolved).name, "", "", _STATUS_WAITING, "", ""]
            )
            item.setData(0, QtCore.Qt.UserRole, resolved)
            try:
                info = media.probe(resolved)
            except media.MediaError as exc:
                item.setText(1, "cannot be read")
                item.setText(3, _STATUS_FAILED)
                item.setToolTip(1, str(exc))
            else:
                item.setText(1, info.kind)
                item.setText(2, media.format_duration(info.duration))
                item.setToolTip(1, info.describe())
            self.files.addTopLevelItem(item)
            existing.add(resolved)
            added += 1
        if added:
            self.statusBar().showMessage(f"Added {added} recording(s).")
        self._update_buttons()
        return added

    def _choose_files(self) -> None:
        chosen, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self,
            "Choose recordings",
            str(Path.home()),
            # Deliberately permissive: the extension is not what decides.
            "Audio and video (*);;All files (*)",
        )
        if chosen:
            self.add_files([Path(p) for p in chosen])

    def paste_files(self) -> None:
        """Take files copied in the file manager, or a path on the clipboard."""
        clipboard = QtWidgets.QApplication.clipboard()
        mime = clipboard.mimeData()
        paths: list[Path] = []
        if mime.hasUrls():
            paths = [Path(url.toLocalFile()) for url in mime.urls() if url.isLocalFile()]
        elif mime.hasText():
            candidate = Path(mime.text().strip().strip('"'))
            if candidate.exists() and candidate.is_file():
                paths = [candidate]
        if paths:
            self.add_files(paths)
        else:
            self.statusBar().showMessage("Nothing on the clipboard that is a file.")

    def _remove_selected(self) -> None:
        for item in self.files.selectedItems():
            index = self.files.indexOfTopLevelItem(item)
            self._bundles.pop(index, None)
            self.files.takeTopLevelItem(index)
        self._reindex_bundles()
        self._update_buttons()

    def _clear_files(self) -> None:
        self.files.clear()
        self._bundles.clear()
        self.text_view.clear()
        self._update_buttons()

    def _reindex_bundles(self) -> None:
        """Keep bundles attached to their rows after a removal."""
        rebuilt: dict[int, artifact.Bundle] = {}
        for row in range(self.files.topLevelItemCount()):
            item = self.files.topLevelItem(row)
            bundle = item.data(0, QtCore.Qt.UserRole + 1)
            if bundle is not None:
                rebuilt[row] = bundle
        self._bundles = rebuilt

    # ---- drag and drop --------------------------------------------------

    def dragEnterEvent(self, event: QtGui.QDragEnterEvent) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QtGui.QDropEvent) -> None:
        paths = [
            Path(url.toLocalFile())
            for url in event.mimeData().urls()
            if url.isLocalFile()
        ]
        if paths:
            self.add_files(paths)
            event.acceptProposedAction()

    # ---- settings -------------------------------------------------------

    def _choose_output(self) -> None:
        chosen = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Where should transcripts go?", str(self._output)
        )
        if chosen:
            self._output = Path(chosen)
            self.output_label.setText(chosen)

    def _refresh_engine_controls(self) -> None:
        engine = self.engine_box.currentData() or DEFAULT_ENGINE
        self.model_box.setEnabled(engine == "whisper")
        self._style_badge(cloud=(engine == "cloud"))
        if engine == "cloud":
            self.statusBar().showMessage(
                "Cloud recognition uploads your audio to a third party."
            )

    def _refresh_summary_controls(self) -> None:
        self.instruction_box.setVisible(self.task_box.currentData() == "custom")

    def selected_language(self) -> list[str]:
        code = self.language_box.currentData()
        return [code] if code else []

    def options(self) -> TranscribeOptions:
        engine = self.engine_box.currentData() or DEFAULT_ENGINE
        engine_options: dict = {}
        if engine == "whisper":
            engine_options["model_size"] = self.model_box.currentData() or DEFAULT_MODEL
        elif engine == "cloud":
            engine_options["fallback"] = create("whisper")
        return TranscribeOptions(
            engine=engine,
            engine_options=engine_options,
            languages=self.selected_language(),
        )

    # ---- running --------------------------------------------------------

    def _pending_rows(self) -> list[tuple[int, Path]]:
        selected = {
            self.files.indexOfTopLevelItem(item) for item in self.files.selectedItems()
        }
        jobs: list[tuple[int, Path]] = []
        for row in range(self.files.topLevelItemCount()):
            if selected and row not in selected:
                continue
            item = self.files.topLevelItem(row)
            if item.text(3) == _STATUS_DONE:
                continue
            jobs.append((row, Path(item.data(0, QtCore.Qt.UserRole))))
        return jobs

    def start(self) -> None:
        if self._worker is not None and self._worker.isRunning():
            return
        jobs = self._pending_rows()
        if not jobs:
            self.statusBar().showMessage("Nothing to transcribe. Add a recording.")
            return
        try:
            options = self.options()
            create(options.engine, **options.engine_options).require_available()
        except EngineError as exc:
            QtWidgets.QMessageBox.warning(self, "That recognizer is not ready", str(exc))
            return

        run_root = artifact.run_directory(self._output)
        self._worker = TranscribeWorker(jobs, run_root, options)
        self._worker.advanced.connect(self._on_progress)
        self._worker.completed.connect(self._on_completed)
        self._worker.failed.connect(self._on_failed)
        self._worker.finished_all.connect(self._on_finished)
        for row, _ in jobs:
            self.files.topLevelItem(row).setText(3, "queued")
        self.transcribe_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.statusBar().showMessage(f"Transcribing {len(jobs)} recording(s)…")
        self._worker.start()

    def stop(self) -> None:
        if self._worker is not None and self._worker.isRunning():
            self._worker.stop()
            self.statusBar().showMessage("Stopping after the current step…")

    @QtCore.Slot(int, object)
    def _on_progress(self, row: int, progress: Progress) -> None:
        self.progress.setValue(progress.percent)
        item = self.files.topLevelItem(row)
        if item is not None:
            item.setText(3, progress.stage)
        if progress.message:
            self.statusBar().showMessage(progress.message)

    @QtCore.Slot(int, object)
    def _on_completed(self, row: int, bundle: artifact.Bundle) -> None:
        self._bundles[row] = bundle
        item = self.files.topLevelItem(row)
        if item is not None:
            transcript = bundle.transcript
            unresolved = len(transcript.unresolved_issues())
            item.setText(3, _STATUS_DONE)
            item.setText(4, str(transcript.word_count()))
            item.setText(5, str(unresolved) if unresolved else "—")
            item.setData(0, QtCore.Qt.UserRole + 1, bundle)
            item.setToolTip(3, str(bundle.directory))
            self.files.setCurrentItem(item)
        self._show_bundle(bundle)

    @QtCore.Slot(int, str)
    def _on_failed(self, row: int, message: str) -> None:
        item = self.files.topLevelItem(row)
        if item is not None:
            item.setText(3, _STATUS_FAILED)
            item.setToolTip(3, message)
        self.statusBar().showMessage(message.splitlines()[0])

    @QtCore.Slot()
    def _on_finished(self) -> None:
        self.transcribe_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.progress.setValue(0)
        done = sum(
            1
            for row in range(self.files.topLevelItemCount())
            if self.files.topLevelItem(row).text(3) == _STATUS_DONE
        )
        self.statusBar().showMessage(f"Finished. {done} recording(s) transcribed.")
        self._update_buttons()

    # ---- results --------------------------------------------------------

    def _on_selection_changed(self) -> None:
        bundle = self.current_bundle()
        if bundle is not None:
            self._show_bundle(bundle)
        self._update_buttons()

    def current_bundle(self) -> artifact.Bundle | None:
        items = self.files.selectedItems()
        if not items:
            return None
        return items[0].data(0, QtCore.Qt.UserRole + 1)

    def _show_bundle(self, bundle: artifact.Bundle) -> None:
        self.text_view.setPlainText(self.current_text(bundle))
        self._update_buttons()

    def current_text(self, bundle: artifact.Bundle | None = None) -> str:
        bundle = bundle or self.current_bundle()
        if bundle is None:
            return ""
        return export.to_text(
            bundle.transcript, timestamps=self.timestamps_box.isChecked()
        )

    def copy_text(self) -> bool:
        text = self.current_text()
        if not text.strip():
            self.statusBar().showMessage("Nothing to copy yet.")
            return False
        QtWidgets.QApplication.clipboard().setText(text)
        self.statusBar().showMessage("Transcript copied to the clipboard.")
        return True

    def copy_summary(self) -> bool:
        if not self._summary_text.strip():
            self.statusBar().showMessage("Nothing to copy yet.")
            return False
        QtWidgets.QApplication.clipboard().setText(self._summary_text)
        self.statusBar().showMessage("Result copied to the clipboard.")
        return True

    def save_as(self) -> Path | None:
        bundle = self.current_bundle()
        if bundle is None:
            self.statusBar().showMessage("Select a transcribed recording first.")
            return None
        stem = Path(bundle.transcript.media.name).stem
        chosen, selected = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save the transcript",
            str(Path.home() / f"{stem}.docx"),
            "Word document (*.docx);;Plain text (*.txt);;Markdown (*.md);;"
            "Subtitles (*.srt);;WebVTT (*.vtt);;Spreadsheet (*.csv);;Everything (*.json)",
        )
        if not chosen:
            return None
        return self.write_transcript(bundle, Path(chosen))

    def write_transcript(self, bundle: artifact.Bundle, target: Path) -> Path | None:
        """Write one transcript to a chosen path, reporting what happened."""
        try:
            written = export.write(
                bundle.transcript,
                target,
                timestamps=self.timestamps_box.isChecked(),
            )
        except (export.ExportError, OSError) as exc:
            QtWidgets.QMessageBox.warning(self, "Could not save it", str(exc))
            return None
        self.statusBar().showMessage(f"Saved {written}")
        return written

    def open_folder(self) -> None:
        bundle = self.current_bundle()
        if bundle is None:
            self.statusBar().showMessage("Select a transcribed recording first.")
            return
        QtGui.QDesktopServices.openUrl(
            QtCore.QUrl.fromLocalFile(str(bundle.directory))
        )

    # ---- summarizing ----------------------------------------------------

    def current_task(self):
        if self.task_box.currentData() == "custom":
            instruction = self.instruction_box.text().strip()
            if not instruction:
                return None
            return summarize.custom_task(instruction)
        return summarize.task_named(self.task_box.currentData())

    def run_summary(self) -> None:
        bundle = self.current_bundle()
        if bundle is None:
            self.statusBar().showMessage("Select a transcribed recording first.")
            return
        task = self.current_task()
        if task is None:
            self.statusBar().showMessage("Type the instruction you want to use.")
            return
        if self._summary_worker is not None and self._summary_worker.isRunning():
            return
        self.summary_button.setEnabled(False)
        self.summary_view.setPlainText("Working…")
        self._summary_worker = SummaryWorker(
            bundle.transcript,
            task,
            self.provider_box.currentData(),
            bundle.directory,
        )
        self._summary_worker.completed.connect(self._on_summary)
        self._summary_worker.failed.connect(self._on_summary_failed)
        self._summary_worker.progressed.connect(
            lambda n, total: self.statusBar().showMessage(f"Part {n} of {total}…")
        )
        self._summary_worker.start()

    @QtCore.Slot(object)
    def _on_summary(self, result) -> None:
        self._summary_text = result.text
        self.summary_view.setPlainText(result.text)
        self.summary_button.setEnabled(True)
        self.statusBar().showMessage(
            f"{result.task} from {result.provider} ({result.parts} part(s))."
        )

    @QtCore.Slot(str)
    def _on_summary_failed(self, message: str) -> None:
        self.summary_view.setPlainText("")
        self.summary_button.setEnabled(True)
        QtWidgets.QMessageBox.warning(self, "That did not work", message)

    # ---- state ----------------------------------------------------------

    def _update_buttons(self) -> None:
        has_files = self.files.topLevelItemCount() > 0
        has_result = self.current_bundle() is not None
        running = self._worker is not None and self._worker.isRunning()
        self.transcribe_button.setEnabled(has_files and not running)
        self.remove_button.setEnabled(has_files)
        self.clear_button.setEnabled(has_files)
        for button in (self.copy_button, self.save_button, self.folder_button,
                       self.summary_button):
            button.setEnabled(has_result)
        self.copy_summary_button.setEnabled(bool(self._summary_text))

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        if self._worker is not None and self._worker.isRunning():
            self._worker.stop()
            self._worker.wait(5000)
        super().closeEvent(event)


def main(inputs: Sequence[str] = ()) -> int:
    """Open the window. Returns the Qt exit code."""
    application = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    application.setApplicationName("Speech2Text")
    if not media.ffmpeg_available():
        QtWidgets.QMessageBox.critical(
            None,
            "ffmpeg is missing",
            "Speech2Text needs ffmpeg to open audio and video files.\n\n"
            "On Ubuntu: sudo apt install ffmpeg",
        )
        return 1
    window = MainWindow(inputs)
    window.show()
    return application.exec()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv[1:]))
