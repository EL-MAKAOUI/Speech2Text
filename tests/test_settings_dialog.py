"""The settings window: models, defaults and the advanced dials."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")
pytest.importorskip("faster_whisper")

from PySide6 import QtCore, QtWidgets  # noqa: E402

from speech2text.engines.whisper import MODEL_SIZES  # noqa: E402
from speech2text.ui.settings_dialog import (  # noqa: E402
    APPROXIMATE_SIZES,
    MODEL_NOTES,
    SettingsDialog,
)


@pytest.fixture(scope="session")
def application():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture
def settings(tmp_path):
    return QtCore.QSettings(str(tmp_path / "s.ini"), QtCore.QSettings.IniFormat)


@pytest.fixture
def cache(tmp_path):
    made = tmp_path / "models"
    made.mkdir()
    return made


@pytest.fixture
def dialog(application, settings, cache, monkeypatch):
    from speech2text.ui import settings_dialog as module

    monkeypatch.setattr(module, "model_cache_dir", lambda: cache)
    made = SettingsDialog(settings)
    yield made
    made.close()


def pretend_downloaded(cache: Path, model_size: str, size: int = 2048) -> None:
    repo = {"distil-large-v3": "Systran--faster-distil-whisper-large-v3",
            "large-v3-turbo": "mobiuslabsgmbh--faster-whisper-large-v3-turbo"}.get(
        model_size, f"Systran--faster-whisper-{model_size}"
    )
    snapshot = cache / f"models--{repo}" / "snapshots" / "rev"
    snapshot.mkdir(parents=True, exist_ok=True)
    (snapshot / "model.bin").write_bytes(b"x" * size)


class TestModels:
    def test_every_model_is_listed(self, dialog):
        listed = [
            dialog.models.topLevelItem(i).data(0, QtCore.Qt.UserRole)
            for i in range(dialog.models.topLevelItemCount())
        ]
        assert listed == list(MODEL_SIZES)

    def test_each_says_how_big_it_is_and_what_it_is_for(self, dialog):
        for i in range(dialog.models.topLevelItemCount()):
            item = dialog.models.topLevelItem(i)
            size = item.data(0, QtCore.Qt.UserRole)
            assert item.text(1) == APPROXIMATE_SIZES[size]
            assert item.text(3) == MODEL_NOTES[size]

    def test_the_quoted_sizes_and_notes_cover_every_model(self):
        assert set(APPROXIMATE_SIZES) == set(MODEL_SIZES)
        assert set(MODEL_NOTES) == set(MODEL_SIZES)

    def test_a_downloaded_model_shows_its_real_size(self, dialog, cache):
        pretend_downloaded(cache, "base", 3 * 1024 ** 2)
        dialog.refresh_models()
        row = next(
            dialog.models.topLevelItem(i)
            for i in range(dialog.models.topLevelItemCount())
            if dialog.models.topLevelItem(i).data(0, QtCore.Qt.UserRole) == "base"
        )
        assert row.text(2) == "3 MB"

    def test_it_opens_on_the_model_that_would_be_used(self, dialog):
        from speech2text.engines.whisper import default_model

        assert dialog.selected_model() == default_model()

    def test_download_is_offered_for_one_that_is_missing(self, dialog):
        assert dialog.download_button.isEnabled() is True
        assert dialog.remove_button.isEnabled() is False

    def test_remove_is_offered_once_it_is_here(self, dialog, cache):
        pretend_downloaded(cache, dialog.selected_model())
        dialog.refresh_models()
        assert dialog.remove_button.isEnabled() is True
        assert dialog.download_button.isEnabled() is False

    def test_removing_frees_the_disk(self, dialog, cache, monkeypatch):
        size = dialog.selected_model()
        pretend_downloaded(cache, size, 4096)
        dialog.refresh_models()
        monkeypatch.setattr(
            QtWidgets.QMessageBox, "question",
            lambda *a, **k: QtWidgets.QMessageBox.Yes,
        )
        dialog.remove_selected()
        assert dialog.model_is_here(size) is False
        assert "Removed" in dialog.download_status.text()

    def test_removing_can_be_called_off(self, dialog, cache, monkeypatch):
        size = dialog.selected_model()
        pretend_downloaded(cache, size)
        dialog.refresh_models()
        monkeypatch.setattr(
            QtWidgets.QMessageBox, "question",
            lambda *a, **k: QtWidgets.QMessageBox.No,
        )
        dialog.remove_selected()
        assert dialog.model_is_here(size) is True

    def test_downloading_runs_the_same_command_the_terminal_would(self, dialog, cache):
        from speech2text.ui.settings_dialog import DownloadWorker

        worker = DownloadWorker("small", cache)
        process = QtCore.QProcess()
        process.setArguments([
            "-m", "speech2text.cli", "models", "get", "small", "--cache", str(cache)
        ])
        assert worker.model_size == "small" and worker.cache == cache

    def test_the_cache_folder_is_shown(self, dialog, cache):
        assert dialog.cache_label.text() == str(cache)
        assert dialog.cache_label.isReadOnly()


class TestRecognitionDefaults:
    def test_the_advanced_dials_are_saved(self, dialog, settings):
        dialog.beam_spin.setValue(8)
        dialog.vad_box.setChecked(False)
        dialog.threads_spin.setValue(2)
        dialog.compute_choice.setCurrentIndex(dialog.compute_choice.findData("int8"))
        dialog.save()

        assert settings.value("beam_size", type=int) == 8
        assert settings.value("vad_filter", type=bool) is False
        assert settings.value("cpu_threads", type=int) == 2
        assert settings.value("compute_type", type=str) == "int8"

    def test_saved_values_come_back(self, application, settings, cache, monkeypatch):
        from speech2text.ui import settings_dialog as module

        monkeypatch.setattr(module, "model_cache_dir", lambda: cache)
        settings.setValue("beam_size", 7)
        settings.setValue("vad_filter", False)
        reopened = SettingsDialog(settings)
        try:
            assert reopened.beam_spin.value() == 7
            assert reopened.vad_box.isChecked() is False
        finally:
            reopened.close()

    def test_precision_can_be_left_to_the_application(self, dialog):
        assert dialog.compute_choice.currentData() == ""

    def test_a_model_not_downloaded_is_marked_in_the_list(self, dialog):
        labels = [
            dialog.model_choice.itemText(i) for i in range(dialog.model_choice.count())
        ]
        assert any("not downloaded" in label for label in labels)

    def test_every_recognizer_is_offered(self, dialog):
        from speech2text.engines import ENGINE_NAMES

        offered = [
            dialog.engine_choice.itemData(i)
            for i in range(dialog.engine_choice.count())
        ]
        assert offered == list(ENGINE_NAMES)


class TestChecking:
    def test_the_playback_speed_is_saved(self, dialog, settings):
        dialog.speed_default.setCurrentIndex(dialog.speed_default.findData(1.5))
        dialog.save()
        assert settings.value("review_speed", type=float) == 1.5

    def test_playing_as_you_move_is_on_to_begin_with(self, dialog):
        assert dialog.autoplay_default.isChecked() is True


class TestKeys:
    @pytest.fixture(autouse=True)
    def isolated(self, tmp_path, monkeypatch):
        from speech2text import keys as keystore

        monkeypatch.setattr(keystore, "_LLMKIT_STORE", tmp_path / "llmkit.json")
        monkeypatch.setattr(keystore, "_OWN_STORE", tmp_path / "own.json")
        for name in ("GROQ_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY"):
            monkeypatch.delenv(name, raising=False)

    def test_with_no_keys_it_says_so(self, dialog):
        dialog.refresh_keys()
        assert dialog.keys_list.topLevelItem(0).text(0) == "none found"

    def test_a_key_can_be_added_and_is_then_counted(self, dialog):
        dialog.key_provider.setCurrentIndex(dialog.key_provider.findData("groq"))
        dialog.key_value.setText("gsk_a_test_key_value")
        assert dialog.add_key() is True
        rows = [
            dialog.keys_list.topLevelItem(i).text(0)
            for i in range(dialog.keys_list.topLevelItemCount())
        ]
        assert "groq" in rows
        assert dialog.key_value.text() == "", "the box is cleared after storing"

    def test_an_empty_key_is_not_stored(self, dialog):
        dialog.key_value.setText("   ")
        assert dialog.add_key() is False

    def test_the_key_is_never_shown(self, dialog):
        assert dialog.key_value.echoMode() == QtWidgets.QLineEdit.Password
        dialog.key_provider.setCurrentIndex(dialog.key_provider.findData("groq"))
        dialog.key_value.setText("gsk_secret_value_here")
        dialog.add_key()
        shown = " ".join(
            dialog.keys_list.topLevelItem(i).text(c)
            for i in range(dialog.keys_list.topLevelItemCount())
            for c in range(3)
        )
        assert "gsk_secret_value_here" not in shown


class TestPresentation:
    def test_every_control_explains_itself(self, dialog):
        missing = [
            name for name, widget in vars(dialog).items()
            if isinstance(widget, QtWidgets.QWidget) and not widget.toolTip()
        ]
        assert missing == []

    def test_the_tabs_are_the_ones_the_guide_names(self, dialog):
        assert [dialog.tabs.tabText(i) for i in range(dialog.tabs.count())] == [
            "Models", "Recognition", "Checking", "API keys"
        ]
