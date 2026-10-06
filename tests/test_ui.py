"""The window, driven headless.

These exercise the real widgets: adding files, switching engines, copying,
saving and the state of every button. Transcription itself runs in a thread,
so the tests drive the window's own slots with the objects the worker emits.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from conftest import requires_ffmpeg

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")

from PySide6 import QtWidgets  # noqa: E402

from speech2text import artifact, export  # noqa: E402
from speech2text.pipeline import Progress  # noqa: E402
from speech2text.ui.app import MainWindow  # noqa: E402


@pytest.fixture
def remembered_settings(tmp_path):
    """A settings file of this test's own.

    The window remembers what was chosen last time, so without this a test
    that changes a setting would decide what later tests see — and would
    write into whatever the person running the tests had chosen.
    """
    from PySide6 import QtCore

    return QtCore.QSettings(
        str(tmp_path / "settings.ini"), QtCore.QSettings.IniFormat
    )


@pytest.fixture(scope="session")
def application():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture
def window(application, tmp_path, remembered_settings):
    made = MainWindow(settings=remembered_settings)
    made._output = tmp_path / "out"
    made.output_label.setText(str(made._output))
    yield made
    made.close()


@pytest.fixture
def finished_window(window, sample_transcript, tmp_path):
    """A window showing one finished recording."""
    bundle = artifact.write_bundle(sample_transcript, tmp_path / "bundle")
    window.add_files([Path(sample_transcript.media.path)])
    item = QtWidgets.QTreeWidgetItem(["interview.mp4", "video", "0:12", "waiting", "", ""])
    item.setData(0, 256, str(tmp_path / "interview.mp4"))  # Qt.UserRole
    window.files.clear()
    window.files.addTopLevelItem(item)
    window._on_completed(0, bundle)
    window.files.setCurrentItem(item)
    return window


class TestStartingState:
    def test_it_opens_empty_and_most_buttons_are_off(self, window):
        assert window.files.topLevelItemCount() == 0
        assert not window.transcribe_button.isEnabled()
        assert not window.copy_button.isEnabled()
        assert not window.save_button.isEnabled()
        assert not window.stop_button.isEnabled()

    def test_the_badge_starts_local(self, window):
        assert window.badge.text() == "LOCAL • PRIVATE"

    def test_language_defaults_to_automatic(self, window):
        assert window.language_box.currentIndex() == 0
        assert window.selected_language() == []

    def test_every_language_is_offered(self, window):
        from speech2text import languages

        assert window.language_box.count() == len(languages.LANGUAGES) + 1

    def test_every_engine_is_offered(self, window):
        from speech2text.engines import ENGINE_NAMES

        offered = [window.engine_box.itemData(i) for i in range(window.engine_box.count())]
        assert offered == list(ENGINE_NAMES)

    def test_every_control_explains_itself(self, window):
        """AGENTS.md: every control carries a tooltip."""
        missing = [
            name for name, widget in vars(window).items()
            if isinstance(widget, QtWidgets.QWidget) and not widget.toolTip()
        ]
        assert missing == []


@requires_ffmpeg
class TestAddingFiles:
    def test_a_file_is_described_by_what_is_inside_it(self, window, misnamed_video):
        window.add_files([misnamed_video])
        item = window.files.topLevelItem(0)
        assert item.text(0) == "recording.dat"
        assert item.text(1) == "video", "the .dat suffix must not decide this"
        assert item.text(3) == "waiting"

    def test_a_file_with_no_audio_is_marked_rather_than_queued(self, window, silent_video):
        window.add_files([silent_video])
        item = window.files.topLevelItem(0)
        assert item.text(1) == "cannot be read"
        assert item.text(3) == "failed"
        assert "no sound track" in item.toolTip(1)

    def test_the_same_file_is_not_added_twice(self, window, tone_wav):
        window.add_files([tone_wav])
        window.add_files([tone_wav])
        assert window.files.topLevelItemCount() == 1

    def test_adding_a_file_enables_transcribing(self, window, tone_wav):
        window.add_files([tone_wav])
        assert window.transcribe_button.isEnabled()

    def test_clearing_empties_the_list(self, window, tone_wav):
        window.add_files([tone_wav])
        window._clear_files()
        assert window.files.topLevelItemCount() == 0
        assert not window.transcribe_button.isEnabled()

    def test_removing_takes_only_the_selected_row(self, window, tone_wav, misnamed_video):
        window.add_files([tone_wav, misnamed_video])
        window.files.topLevelItem(0).setSelected(True)
        window._remove_selected()
        assert window.files.topLevelItemCount() == 1
        assert window.files.topLevelItem(0).text(0) == "recording.dat"

    def test_starting_with_nothing_says_so_rather_than_failing(self, window):
        window.start()
        assert "Nothing to transcribe" in window.statusBar().currentMessage()


class TestEngineChoice:
    def test_choosing_cloud_warns_that_audio_leaves_the_machine(self, window):
        window.engine_box.setCurrentIndex(window.engine_box.findData("cloud"))
        assert window.badge.text() == "CLOUD • AUDIO IS UPLOADED"
        assert "uploads your audio" in window.statusBar().currentMessage()

    def test_going_back_to_a_local_engine_clears_the_warning(self, window):
        window.engine_box.setCurrentIndex(window.engine_box.findData("cloud"))
        window.engine_box.setCurrentIndex(window.engine_box.findData("whisper"))
        assert window.badge.text() == "LOCAL • PRIVATE"

    def test_the_model_choice_applies_only_to_whisper(self, window):
        window.engine_box.setCurrentIndex(window.engine_box.findData("whisper"))
        assert window.model_box.isEnabled()
        window.engine_box.setCurrentIndex(window.engine_box.findData("sphinx"))
        assert not window.model_box.isEnabled()

    def test_the_chosen_settings_reach_the_options(self, window):
        window.engine_box.setCurrentIndex(window.engine_box.findData("sphinx"))
        window.language_box.setCurrentIndex(window.language_box.findData("fr"))
        options = window.options()
        assert options.engine == "sphinx"
        assert options.languages == ["fr"]

    def test_the_cloud_is_given_a_local_fallback(self, window):
        window.engine_box.setCurrentIndex(window.engine_box.findData("cloud"))
        assert window.options().engine_options["fallback"].name == "whisper"


class TestResults:
    def test_a_finished_recording_shows_its_text_and_counts(self, finished_window):
        transcript = finished_window.current_bundle().transcript
        item = finished_window.files.topLevelItem(0)
        assert item.text(3) == "done"
        assert item.text(4) == str(transcript.word_count())
        assert item.text(5) == str(len(transcript.unresolved_issues()))
        assert "Hello and welcome." in finished_window.text_view.toPlainText()

    def test_the_buttons_come_alive(self, finished_window):
        assert finished_window.copy_button.isEnabled()
        assert finished_window.save_button.isEnabled()
        assert finished_window.folder_button.isEnabled()

    def test_copy_puts_the_transcript_on_the_clipboard(self, finished_window, application):
        assert finished_window.copy_text() is True
        assert "Hello and welcome." in application.clipboard().text()

    def test_copying_with_nothing_selected_does_not_pretend_it_worked(self, window):
        assert window.copy_text() is False
        assert "Nothing to copy" in window.statusBar().currentMessage()

    def test_timestamps_can_be_included(self, finished_window, application):
        finished_window.timestamps_box.setChecked(True)
        finished_window.copy_text()
        assert application.clipboard().text().startswith("[0:00] ")

    @pytest.mark.parametrize("suffix", ["txt", "docx", "srt", "md", "vtt", "csv", "json"])
    def test_every_format_can_be_saved(self, finished_window, tmp_path, suffix):
        target = tmp_path / f"saved.{suffix}"
        written = finished_window.write_transcript(
            finished_window.current_bundle(), target
        )
        assert written == target and target.stat().st_size > 0

    def test_the_saved_text_includes_corrections(self, finished_window, tmp_path):
        bundle = finished_window.current_bundle()
        bundle.transcript.correct(1, "Today we discuss testing.")
        target = finished_window.write_transcript(bundle, tmp_path / "saved.txt")
        assert "discuss testing" in target.read_text()

    def test_saving_somewhere_impossible_is_reported_not_raised(
        self, finished_window, tmp_path, monkeypatch
    ):
        def refuse(*args, **kwargs):
            raise OSError("read-only file system")

        monkeypatch.setattr(export, "write", refuse)
        monkeypatch.setattr(QtWidgets.QMessageBox, "warning", lambda *a, **k: None)
        assert finished_window.write_transcript(
            finished_window.current_bundle(), tmp_path / "x.txt"
        ) is None


class TestProgress:
    def test_the_bar_and_the_row_follow_the_run(self, window, tone_wav):
        window.add_files([tone_wav])
        window._on_progress(0, Progress(stage="recognizing", fraction=0.42, message="Listening"))
        assert window.progress.value() == 42
        assert window.files.topLevelItem(0).text(3) == "recognizing"
        assert window.statusBar().currentMessage() == "Listening"

    def test_a_failure_is_shown_on_its_row_with_the_reason(self, window, tone_wav):
        window.add_files([tone_wav])
        window._on_failed(0, "the model is not downloaded yet")
        item = window.files.topLevelItem(0)
        assert item.text(3) == "failed"
        assert "not downloaded" in item.toolTip(3)

    def test_finishing_re_enables_the_controls(self, window, tone_wav):
        window.add_files([tone_wav])
        window.transcribe_button.setEnabled(False)
        window.stop_button.setEnabled(True)
        window._on_finished()
        assert window.transcribe_button.isEnabled()
        assert not window.stop_button.isEnabled()
        assert window.progress.value() == 0


class TestSummaryTab:
    def test_every_task_is_offered_plus_a_custom_one(self, window):
        from speech2text import summarize

        offered = [window.task_box.itemData(i) for i in range(window.task_box.count())]
        assert offered == list(summarize.TASKS) + ["custom"]

    def test_the_instruction_box_appears_only_for_a_custom_task(self, window):
        window.show()
        window.tabs.setCurrentIndex(1)
        window.task_box.setCurrentIndex(window.task_box.findData("custom"))
        assert window.instruction_box.isVisible()
        window.task_box.setCurrentIndex(0)
        assert not window.instruction_box.isVisible()

    def test_a_custom_task_with_no_instruction_is_not_run(self, window):
        window.task_box.setCurrentIndex(window.task_box.findData("custom"))
        assert window.current_task() is None

    def test_a_preset_task_is_resolved(self, window):
        window.task_box.setCurrentIndex(window.task_box.findData("summary"))
        assert window.current_task().name == "summary"

    def test_running_without_a_recording_says_so(self, window):
        window.run_summary()
        assert "Select a transcribed recording" in window.statusBar().currentMessage()

    def test_a_failure_is_shown_rather_than_crashing(self, window, monkeypatch):
        shown = {}
        monkeypatch.setattr(
            QtWidgets.QMessageBox, "warning",
            lambda parent, title, text: shown.update(title=title, text=text),
        )
        window._on_summary_failed("no API key for groq")
        assert "no API key" in shown["text"]
        assert window.summary_button.isEnabled()

    def test_a_result_is_shown_and_can_be_copied(self, window, application):
        from speech2text.summarize import SummaryResult

        window._on_summary(SummaryResult("summary", "the gist of it", "groq", "m", 1, 50))
        assert window.summary_view.toPlainText() == "the gist of it"
        assert window.copy_summary() is True
        assert application.clipboard().text() == "the gist of it"


class TestOpeningTheChecker:
    def test_checking_needs_a_finished_recording(self, window):
        window.open_review()
        assert "Select a transcribed recording" in window.statusBar().currentMessage()

    def test_it_opens_on_the_selected_recording(self, finished_window):
        finished_window.open_review()
        checker = finished_window._review_window
        try:
            assert checker is not None
            assert checker.directory == finished_window.current_bundle().directory
            assert checker.segments.topLevelItemCount() == 3
        finally:
            if checker is not None:
                checker.close()

    def test_a_correction_made_there_shows_up_here(self, finished_window):
        finished_window.open_review()
        checker = finished_window._review_window
        try:
            checker.autoplay_box.setChecked(False)
            checker.select(1, play=False)
            checker.editor.setPlainText("Today we discuss testing.")
            assert checker.save_correction() is True
        finally:
            checker.close()
        assert "discuss testing" in finished_window.text_view.toPlainText()
        assert finished_window.files.topLevelItem(0).text(5) == "—"


class TestDefaultsAndMemory:
    def test_the_default_model_is_the_engine_s_default(self, window):
        from speech2text.engines.whisper import default_model

        assert window.model_box.currentData() == default_model()

    def test_where_to_run_can_be_chosen(self, window):
        assert [
            window.device_box.itemData(i) for i in range(window.device_box.count())
        ] == ["auto", "cpu", "cuda"]

    def test_the_device_reaches_the_engine_options(self, window):
        window.engine_box.setCurrentIndex(window.engine_box.findData("whisper"))
        window.device_box.setCurrentIndex(window.device_box.findData("cpu"))
        assert window.options().engine_options["device"] == "cpu"

    def test_choices_come_back_next_time(self, application, remembered_settings):
        from speech2text.ui.app import MainWindow

        first = MainWindow(settings=remembered_settings)
        first.engine_box.setCurrentIndex(first.engine_box.findData("sphinx"))
        first.language_box.setCurrentIndex(first.language_box.findData("fr"))
        first.device_box.setCurrentIndex(first.device_box.findData("cpu"))
        first.timestamps_box.setChecked(True)
        first._remember_settings()
        first.close()

        second = MainWindow(settings=remembered_settings)
        try:
            assert second.engine_box.currentData() == "sphinx"
            assert second.language_box.currentData() == "fr"
            assert second.device_box.currentData() == "cpu"
            assert second.timestamps_box.isChecked() is True
        finally:
            second.close()


class TestSettingsFromTheMainWindow:
    def test_the_advanced_dials_reach_the_engine(self, window, remembered_settings):
        remembered_settings.setValue("beam_size", 8)
        remembered_settings.setValue("vad_filter", False)
        remembered_settings.setValue("compute_type", "int8")
        remembered_settings.setValue("cpu_threads", 3)
        window.engine_box.setCurrentIndex(window.engine_box.findData("whisper"))

        options = window.options().engine_options
        assert options["beam_size"] == 8
        assert options["vad_filter"] is False
        assert options["compute_type"] == "int8"
        assert options["cpu_threads"] == 3

    def test_precision_left_unset_is_not_forced_on_the_engine(self, window, remembered_settings):
        remembered_settings.setValue("compute_type", "")
        window.engine_box.setCurrentIndex(window.engine_box.findData("whisper"))
        assert "compute_type" not in window.options().engine_options

    def test_saving_settings_is_taken_up_without_a_restart(self, window, remembered_settings):
        remembered_settings.setValue("model", "tiny")
        remembered_settings.setValue("device", "cpu")
        window._apply_settings()
        assert window.model_box.currentData() == "tiny"
        assert window.device_box.currentData() == "cpu"
        assert "Settings saved" in window.statusBar().currentMessage()

    def test_the_checking_window_opens_with_the_saved_preferences(
        self, finished_window, remembered_settings
    ):
        remembered_settings.setValue("review_autoplay", False)
        remembered_settings.setValue("review_speed", 1.5)
        finished_window.open_review()
        checker = finished_window._review_window
        try:
            assert checker.autoplay_box.isChecked() is False
            assert checker.speed_box.currentData() == 1.5
            assert checker.player.speed == 1.5
        finally:
            checker.close()
