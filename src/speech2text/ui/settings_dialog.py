"""Settings: which models are here, what to use by default, and the dials.

Downloading a model is the thing most likely to be needed and least likely
to be convenient from a terminal, so it is here with its real progress and
the disk each one is using.
"""

from __future__ import annotations

import sys
from pathlib import Path

from PySide6 import QtCore, QtWidgets

from .. import keys as keystore
from .. import media, summarize
from ..engines import ENGINE_NAMES, describe_all
from ..engines.whisper import (
    MODEL_SIZES,
    WhisperEngine,
    downloaded_bytes,
    model_cache_dir,
    remove_model,
)
from .player import DEFAULT_SPEED, SPEEDS

#: Roughly how large each download is, so a 3 GB one is not a surprise.
#: Approximate and for guidance only; nothing depends on these numbers.
APPROXIMATE_SIZES = {
    "tiny": "~75 MB",
    "base": "~145 MB",
    "small": "~485 MB",
    "medium": "~1.5 GB",
    "large-v3": "~3.1 GB",
    "large-v3-turbo": "~1.6 GB",
    "distil-large-v3": "~1.5 GB",
}

#: What each model is for, in the terms someone choosing one cares about.
MODEL_NOTES = {
    "tiny": "Fastest, roughest. For a quick idea of what was said.",
    "base": "Quick notes where the odd wrong word does not matter.",
    "small": "The default. Good enough to trust, and fits a laptop.",
    "medium": "Better again, and noticeably slower on a processor.",
    "large-v3": "The most accurate. Slow without a large graphics card.",
    "large-v3-turbo": "Close to large-v3, several times faster.",
    "distil-large-v3": "Nearly large-v3 for English, about twice as fast.",
}

COMPUTE_TYPES = ("", "int8", "int8_float16", "float16", "float32")


class DownloadWorker(QtCore.QObject):
    """Fetches a model by running the same command the terminal would.

    Reusing the command means the download behaves identically either way,
    and its progress is real rather than a guess.
    """

    progressed = QtCore.Signal(str)
    done = QtCore.Signal(bool, str)

    def __init__(self, model_size: str, cache: Path, parent=None) -> None:
        super().__init__(parent)
        self.model_size = model_size
        self.cache = cache
        self._process: QtCore.QProcess | None = None

    def start(self) -> None:
        process = QtCore.QProcess(self)
        process.setProgram(sys.executable)
        process.setArguments([
            "-m", "speech2text.cli", "models", "get",
            self.model_size, "--cache", str(self.cache),
        ])
        process.setProcessChannelMode(QtCore.QProcess.MergedChannels)
        process.readyReadStandardOutput.connect(self._on_output)
        process.finished.connect(self._on_finished)
        self._process = process
        process.start()

    def stop(self) -> None:
        if self._process is not None:
            self._process.kill()
            self._process.waitForFinished(2000)
            self._process = None

    @QtCore.Slot()
    def _on_output(self) -> None:
        if self._process is None:
            return
        text = bytes(self._process.readAllStandardOutput()).decode("utf-8", "replace")
        for line in text.replace("\r", "\n").splitlines():
            if line.strip():
                self.progressed.emit(line.strip())

    @QtCore.Slot()
    def _on_finished(self, code: int = 0, status=None) -> None:
        self._process = None
        self.done.emit(code == 0, "" if code == 0 else "the download did not finish")


class SettingsDialog(QtWidgets.QDialog):
    """Everything that can be set, in one place."""

    changed = QtCore.Signal()

    def __init__(self, settings: QtCore.QSettings, parent=None) -> None:
        super().__init__(parent)
        self.settings = settings
        self.cache = model_cache_dir()
        self._download: DownloadWorker | None = None

        self.setWindowTitle("Speech2Text settings")
        self.resize(760, 600)

        layout = QtWidgets.QVBoxLayout(self)
        self.tabs = QtWidgets.QTabWidget()
        self.tabs.addTab(self._build_models_tab(), "Models")
        self.tabs.addTab(self._build_recognition_tab(), "Recognition")
        self.tabs.addTab(self._build_checking_tab(), "Checking")
        self.tabs.addTab(self._build_keys_tab(), "API keys")
        self.tabs.setToolTip("Models to download, what to use by default, and the dials.")
        layout.addWidget(self.tabs)

        self.buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Save | QtWidgets.QDialogButtonBox.Cancel
        )
        self.buttons.setToolTip("Save keeps these settings for next time.")
        self.buttons.accepted.connect(self.save)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)

        self._load()
        self.refresh_models()

    # ---- models ---------------------------------------------------------

    def _build_models_tab(self) -> QtWidgets.QWidget:
        page = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(page)

        note = QtWidgets.QLabel(
            "A model is downloaded once and then runs on this machine with no "
            "network. Bigger is more accurate and slower."
        )
        note.setWordWrap(True)
        layout.addWidget(note)

        self.models = QtWidgets.QTreeWidget()
        self.models.setHeaderLabels(["Model", "Download", "On disk", "What it is for"])
        self.models.setRootIsDecorated(False)
        self.models.setToolTip(
            "Every model available. Choose one and press Download to fetch "
            "it, or Remove to free the disk it is using."
        )
        self.models.setColumnWidth(0, 140)
        self.models.setColumnWidth(1, 90)
        self.models.setColumnWidth(2, 90)
        self.models.currentItemChanged.connect(self._refresh_model_buttons)
        layout.addWidget(self.models, 1)

        buttons = QtWidgets.QHBoxLayout()
        self.download_button = QtWidgets.QPushButton("Download")
        self.download_button.setToolTip(
            "Fetch the selected model now, so the first transcription does "
            "not have to wait for it."
        )
        self.download_button.clicked.connect(self.download_selected)
        buttons.addWidget(self.download_button)

        self.remove_button = QtWidgets.QPushButton("Remove")
        self.remove_button.setToolTip(
            "Delete the selected model from disk. It can be downloaded again."
        )
        self.remove_button.clicked.connect(self.remove_selected)
        buttons.addWidget(self.remove_button)
        buttons.addStretch(1)
        layout.addLayout(buttons)

        self.download_status = QtWidgets.QLabel("")
        self.download_status.setWordWrap(True)
        self.download_status.setToolTip("What the download is doing.")
        layout.addWidget(self.download_status)

        self.download_bar = QtWidgets.QProgressBar()
        self.download_bar.setRange(0, 0)       # a download of unknown length
        self.download_bar.setVisible(False)
        self.download_bar.setToolTip("A download is in progress.")
        layout.addWidget(self.download_bar)

        folder = QtWidgets.QHBoxLayout()
        folder.addWidget(QtWidgets.QLabel("Kept in"))
        self.cache_label = QtWidgets.QLineEdit(str(self.cache))
        self.cache_label.setReadOnly(True)
        self.cache_label.setToolTip(
            "Where downloaded models live. SPEECH2TEXT_MODEL_CACHE sets this; "
            "point it at a bigger disk, or at a cache copied from elsewhere."
        )
        folder.addWidget(self.cache_label)
        layout.addLayout(folder)
        return page

    def refresh_models(self) -> None:
        from ..engines.whisper import default_model

        # Start on the model that would actually be used, so Download and
        # Remove act on something without a click first.
        selected = self.selected_model() or self.settings.value(
            "model", default_model(), type=str
        )
        self.models.clear()
        for size in MODEL_SIZES:
            on_disk = downloaded_bytes(size, self.cache)
            item = QtWidgets.QTreeWidgetItem([
                size,
                APPROXIMATE_SIZES.get(size, "—"),
                media.format_size(on_disk) if on_disk else "—",
                MODEL_NOTES.get(size, ""),
            ])
            item.setData(0, QtCore.Qt.UserRole, size)
            self.models.addTopLevelItem(item)
            if size == selected:
                self.models.setCurrentItem(item)
        self._refresh_model_buttons()

    def selected_model(self) -> str | None:
        item = self.models.currentItem()
        return item.data(0, QtCore.Qt.UserRole) if item else None

    def model_is_here(self, size: str) -> bool:
        return WhisperEngine(size, cache_dir=self.cache).model_ready()

    def _refresh_model_buttons(self, *args) -> None:
        size = self.selected_model()
        busy = self._download is not None
        here = bool(size) and self.model_is_here(size)
        self.download_button.setEnabled(bool(size) and not here and not busy)
        self.remove_button.setEnabled(here and not busy)

    def download_selected(self) -> None:
        size = self.selected_model()
        if not size or self._download is not None:
            return
        self._download = DownloadWorker(size, self.cache, self)
        self._download.progressed.connect(self.download_status.setText)
        self._download.done.connect(self._on_download_done)
        self.download_bar.setVisible(True)
        self.download_status.setText(
            f"Fetching {size} ({APPROXIMATE_SIZES.get(size, 'a large file')})…"
        )
        self._refresh_model_buttons()
        self.buttons.button(QtWidgets.QDialogButtonBox.Save).setEnabled(False)
        self._download.start()

    @QtCore.Slot(bool, str)
    def _on_download_done(self, succeeded: bool, reason: str) -> None:
        self._download = None
        self.download_bar.setVisible(False)
        self.buttons.button(QtWidgets.QDialogButtonBox.Save).setEnabled(True)
        if not succeeded:
            self.download_status.setText(reason or "the download did not finish")
        self.refresh_models()
        self._refresh_recognition_models()

    def remove_selected(self) -> None:
        size = self.selected_model()
        if not size or not self.model_is_here(size):
            return
        answer = QtWidgets.QMessageBox.question(
            self,
            f"Remove {size}?",
            f"Delete the {size} model from disk? It can be downloaded again "
            f"whenever it is wanted.",
        )
        if answer != QtWidgets.QMessageBox.Yes:
            return
        freed = remove_model(size, self.cache)
        self.download_status.setText(
            f"Removed {size}, freeing {media.format_size(freed)}."
            if freed else f"Could not remove {size}."
        )
        self.refresh_models()
        self._refresh_recognition_models()

    # ---- recognition ----------------------------------------------------

    def _build_recognition_tab(self) -> QtWidgets.QWidget:
        page = QtWidgets.QWidget()
        form = QtWidgets.QFormLayout(page)

        self.engine_choice = QtWidgets.QComboBox()
        for described in describe_all():
            self.engine_choice.addItem(described["title"], described["name"])
        self.engine_choice.setToolTip("Which recognizer new transcriptions use.")
        form.addRow("Recognizer", self.engine_choice)

        self.model_choice = QtWidgets.QComboBox()
        self.model_choice.setToolTip(
            "Which model new transcriptions use. One that is not downloaded "
            "yet is fetched the first time it is needed."
        )
        form.addRow("Model", self.model_choice)

        self.device_choice = QtWidgets.QComboBox()
        for label, value in (
            ("Graphics card if it works", "auto"),
            ("Processor only", "cpu"),
            ("Graphics card only", "cuda"),
        ):
            self.device_choice.addItem(label, value)
        self.device_choice.setToolTip(
            "A graphics card is faster when it can run the model at all. The "
            "first setting tries it and uses the processor if it cannot."
        )
        form.addRow("Run on", self.device_choice)

        advanced = QtWidgets.QGroupBox("Advanced")
        advanced.setToolTip(
            "Dials worth leaving alone unless something specific is wrong."
        )
        advanced_form = QtWidgets.QFormLayout(advanced)

        self.compute_choice = QtWidgets.QComboBox()
        for value in COMPUTE_TYPES:
            self.compute_choice.addItem(value or "Choose for me", value)
        self.compute_choice.setToolTip(
            "How precisely the model computes. int8 is smaller and faster, "
            "float16 needs a graphics card. Leave it to be chosen unless a "
            "model will not fit."
        )
        advanced_form.addRow("Precision", self.compute_choice)

        self.beam_spin = QtWidgets.QSpinBox()
        self.beam_spin.setRange(1, 10)
        self.beam_spin.setToolTip(
            "How many alternatives the recognizer keeps while deciding. "
            "Higher is slightly more accurate and slower; 5 is the usual."
        )
        advanced_form.addRow("Search width", self.beam_spin)

        self.vad_box = QtWidgets.QCheckBox("Skip silence")
        self.vad_box.setToolTip(
            "Leave out the stretches with no speech in them. Faster, and it "
            "stops the recognizer inventing words over silence. Turn it off "
            "if quiet speech is being dropped."
        )
        advanced_form.addRow("", self.vad_box)

        self.threads_spin = QtWidgets.QSpinBox()
        self.threads_spin.setRange(0, 64)
        self.threads_spin.setSpecialValueText("Choose for me")
        self.threads_spin.setToolTip(
            "How many processor cores to use. Leave it to be chosen unless "
            "the machine needs to stay responsive for something else."
        )
        advanced_form.addRow("Cores", self.threads_spin)

        form.addRow(advanced)
        return page

    def _refresh_recognition_models(self) -> None:
        chosen = self.model_choice.currentData()
        self.model_choice.clear()
        for size in MODEL_SIZES:
            here = self.model_is_here(size)
            self.model_choice.addItem(
                f"{size}" + ("" if here else "  (not downloaded)"), size
            )
        position = self.model_choice.findData(chosen)
        if position >= 0:
            self.model_choice.setCurrentIndex(position)

    # ---- checking -------------------------------------------------------

    def _build_checking_tab(self) -> QtWidgets.QWidget:
        page = QtWidgets.QWidget()
        form = QtWidgets.QFormLayout(page)

        self.autoplay_default = QtWidgets.QCheckBox("Play each segment as I move to it")
        self.autoplay_default.setToolTip(
            "In the checking window, play a segment as soon as it is "
            "selected, so checking is listen, read, fix, next."
        )
        form.addRow("", self.autoplay_default)

        self.speed_default = QtWidgets.QComboBox()
        for speed in SPEEDS:
            self.speed_default.addItem(f"{speed:g}×", speed)
        self.speed_default.setToolTip(
            "How fast to play back while checking. The pitch stays the same."
        )
        form.addRow("Playback speed", self.speed_default)

        note = QtWidgets.QLabel(
            "<i>Play on</i> in the checking window keeps playing through the "
            "recording and moves the highlight with it, so a recording with "
            "little wrong can be listened through rather than clicked through."
        )
        note.setWordWrap(True)
        form.addRow(note)
        return page

    # ---- keys -----------------------------------------------------------

    def _build_keys_tab(self) -> QtWidgets.QWidget:
        page = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(page)

        note = QtWidgets.QLabel(
            "Keys are needed only for the cloud recognizer and for "
            "summarizing. Transcribing on this machine never reads one. A key "
            "is never written to a log, an error message, or an artifact."
        )
        note.setWordWrap(True)
        layout.addWidget(note)

        self.keys_list = QtWidgets.QTreeWidget()
        self.keys_list.setHeaderLabels(["Provider", "Keys", "Where from"])
        self.keys_list.setRootIsDecorated(False)
        self.keys_list.setToolTip("The keys found, shown masked.")
        layout.addWidget(self.keys_list, 1)

        add = QtWidgets.QHBoxLayout()
        self.key_provider = QtWidgets.QComboBox()
        for name in sorted(summarize.PROVIDERS):
            self.key_provider.addItem(name, name)
        self.key_provider.setToolTip("Which service the key belongs to.")
        add.addWidget(self.key_provider)

        self.key_value = QtWidgets.QLineEdit()
        self.key_value.setEchoMode(QtWidgets.QLineEdit.Password)
        self.key_value.setPlaceholderText("paste a key here")
        self.key_value.setToolTip(
            "The key is stored in ~/.speech2text/keys.json, readable only by "
            "you, and shown masked from then on."
        )
        add.addWidget(self.key_value, 1)

        self.add_key_button = QtWidgets.QPushButton("Add")
        self.add_key_button.setToolTip("Store this key for the chosen provider.")
        self.add_key_button.clicked.connect(self.add_key)
        add.addWidget(self.add_key_button)
        layout.addLayout(add)
        return page

    def refresh_keys(self) -> None:
        self.keys_list.clear()
        found = keystore.providers_with_keys(tuple(summarize.PROVIDERS))
        if not found:
            item = QtWidgets.QTreeWidgetItem(["none found", "", ""])
            self.keys_list.addTopLevelItem(item)
            return
        for provider in sorted(found):
            entries = keystore.keys_for(provider)
            self.keys_list.addTopLevelItem(
                QtWidgets.QTreeWidgetItem([
                    provider,
                    str(len(entries)),
                    ", ".join(sorted({key.source for key in entries})),
                ])
            )

    def add_key(self) -> bool:
        value = self.key_value.text().strip()
        if not value:
            return False
        keystore.save_key(self.key_provider.currentData(), value)
        self.key_value.clear()
        self.refresh_keys()
        return True

    # ---- loading and saving ---------------------------------------------

    def _load(self) -> None:
        from ..engines.whisper import default_device, default_model

        read = self.settings.value
        self._refresh_recognition_models()
        for box, key, fallback in (
            (self.engine_choice, "engine", ENGINE_NAMES[0]),
            (self.model_choice, "model", default_model()),
            (self.device_choice, "device", default_device()),
            (self.compute_choice, "compute_type", ""),
        ):
            position = box.findData(read(key, fallback, type=str))
            if position >= 0:
                box.setCurrentIndex(position)
        self.beam_spin.setValue(int(read("beam_size", 5, type=int)))
        self.vad_box.setChecked(read("vad_filter", True, type=bool))
        self.threads_spin.setValue(int(read("cpu_threads", 0, type=int)))
        self.autoplay_default.setChecked(read("review_autoplay", True, type=bool))
        speed = self.speed_default.findData(
            float(read("review_speed", DEFAULT_SPEED, type=float))
        )
        if speed >= 0:
            self.speed_default.setCurrentIndex(speed)
        self.refresh_keys()

    def save(self) -> None:
        write = self.settings.setValue
        write("engine", self.engine_choice.currentData())
        write("model", self.model_choice.currentData())
        write("device", self.device_choice.currentData())
        write("compute_type", self.compute_choice.currentData())
        write("beam_size", self.beam_spin.value())
        write("vad_filter", self.vad_box.isChecked())
        write("cpu_threads", self.threads_spin.value())
        write("review_autoplay", self.autoplay_default.isChecked())
        write("review_speed", self.speed_default.currentData())
        self.settings.sync()
        self.changed.emit()
        self.accept()

    def reject(self) -> None:
        if self._download is not None:
            self._download.stop()
            self._download = None
        super().reject()
