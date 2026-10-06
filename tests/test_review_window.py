"""Checking a transcript against the recording.

The window's job: play a few seconds, show what was written for it, and let
it be fixed or confirmed — without ever losing what the recognizer heard.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6 import QtWidgets  # noqa: E402

from speech2text import artifact  # noqa: E402
from speech2text.model import (  # noqa: E402
    MediaIdentity,
    ReviewIssue,
    Segment,
    Transcript,
)
from speech2text.ui.guide_window import REVIEW_SHORTCUTS  # noqa: E402
from speech2text.ui.review_window import ReviewWindow  # noqa: E402


@pytest.fixture(scope="session")
def application():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture
def bundle(tmp_path):
    """Three segments, two of them flagged, the middle one worst."""
    transcript = Transcript(
        media=MediaIdentity(
            name="talk.mp3",
            path=str(tmp_path / "talk.mp3"),   # deliberately absent
            size_bytes=10,
            duration=20.0,
            container="mp3",
        ),
        segments=[
            Segment(0, 0.0, 5.0, "Hello and welcome.", confidence=0.95),
            Segment(1, 5.0, 12.0, "Today we discus testing.", confidence=0.35),
            Segment(2, 12.0, 20.0, "Thanks for listening.", confidence=0.50),
        ],
        issues=[ReviewIssue(1, "low confidence", 0.35), ReviewIssue(2, "low confidence", 0.50)],
        engine="whisper",
        model="small",
        language_detected="en",
    )
    return artifact.write_bundle(transcript, tmp_path / "bundle")


@pytest.fixture
def window(application, bundle):
    made = ReviewWindow(bundle.directory)
    made.autoplay_box.setChecked(False)      # no audio in a test
    yield made
    made.close()


class TestWhatIsShown:
    def test_every_segment_is_listed_with_its_time(self, window):
        assert window.segments.topLevelItemCount() == 3
        assert window.segments.topLevelItem(0).text(0) == "0:00"
        assert window.segments.topLevelItem(2).text(0) == "0:12"

    def test_the_doubtful_ones_are_marked(self, window):
        assert window.segments.topLevelItem(0).text(3) == ""
        assert window.segments.topLevelItem(1).text(3) == "check"

    def test_confidence_is_shown_as_a_number_and_a_mark(self, window):
        assert "95%" in window.segments.topLevelItem(0).text(1)
        assert window.segments.topLevelItem(1).text(1).startswith("●")
        assert window.segments.topLevelItem(0).text(1).startswith("○")

    def test_the_heading_says_how_much_is_left(self, window):
        assert "2</b> to check" in window.heading.text()
        assert "talk.mp3" in window.heading.text()

    def test_choosing_a_segment_shows_both_texts(self, window):
        window.select(1, play=False)
        assert window.heard.toPlainText() == "Today we discus testing."
        assert window.editor.toPlainText() == "Today we discus testing."
        assert "0:05" in window.position_label.text()

    def test_what_was_heard_cannot_be_typed_over(self, window):
        assert window.heard.isReadOnly()

    def test_every_control_explains_itself(self, window):
        missing = [
            name for name, widget in vars(window).items()
            if isinstance(widget, QtWidgets.QWidget) and not widget.toolTip()
        ]
        assert missing == []


class TestTheQueue:
    def test_it_goes_to_the_least_confident_first(self, window):
        window.go_to_next_issue()
        assert window.current_index == 1

    def test_it_moves_on_rather_than_sticking(self, window):
        window.go_to_next_issue()
        window.go_to_next_issue()
        assert window.current_index == 2

    def test_an_empty_queue_says_so(self, window):
        window.transcript.resolve_issue(1, accepted=True)
        window.transcript.resolve_issue(2, accepted=True)
        window.go_to_next_issue()
        assert "Nothing left to check" in window.statusBar().currentMessage()


class TestCorrecting:
    def test_saving_keeps_the_recognition_and_records_the_edit(self, window, bundle):
        window.select(1, play=False)
        window.editor.setPlainText("Today we discuss testing.")
        assert window.save_correction() is True

        reloaded = artifact.load(bundle.directory)
        assert reloaded.segments[1].text == "Today we discus testing."
        assert reloaded.text_of(1) == "Today we discuss testing."
        assert "discus testing" in (bundle.directory / "transcript.raw.txt").read_text()
        assert "discuss testing" in (bundle.directory / "transcript.txt").read_text()

    def test_saving_clears_it_from_the_queue(self, window):
        window.select(1, play=False)
        window.editor.setPlainText("Today we discuss testing.")
        window.save_correction()
        assert [i.segment_index for i in window.transcript.unresolved_issues()] == [2]
        assert "1 left to check" in window.statusBar().currentMessage()

    def test_the_row_shows_the_new_text(self, window):
        window.select(1, play=False)
        window.editor.setPlainText("Today we discuss testing.")
        window.save_correction()
        row = window.segments.topLevelItem(1)
        assert row.text(2) == "Today we discuss testing."
        assert row.text(3) == "edited"

    def test_saving_is_offered_only_once_something_changed(self, window):
        window.select(1, play=False)
        assert window.save_button.isEnabled() is False
        window.editor.setPlainText("something else")
        assert window.save_button.isEnabled() is True

    def test_saving_an_unchanged_segment_does_nothing(self, window):
        window.select(1, play=False)
        assert window.save_correction() is False
        assert window.transcript.corrections == []

    def test_marking_it_correct_clears_it_without_changing_the_text(self, window, bundle):
        window.select(2, play=False)
        assert window.accept_segment() is True
        reloaded = artifact.load(bundle.directory)
        assert reloaded.text_of(2) == "Thanks for listening."
        assert reloaded.corrections == []
        assert [i.segment_index for i in reloaded.unresolved_issues()] == [1]

    def test_an_edit_can_be_undone(self, window, bundle):
        window.select(1, play=False)
        window.editor.setPlainText("something wrong")
        window.save_correction()
        assert window.revert_button.isEnabled()
        assert window.revert_segment() is True

        reloaded = artifact.load(bundle.directory)
        assert reloaded.text_of(1) == "Today we discus testing."
        assert reloaded.segments[1].text == "Today we discus testing."
        assert window.editor.toPlainText() == "Today we discus testing."

    def test_undo_is_offered_only_where_there_is_an_edit(self, window):
        window.select(0, play=False)
        assert window.revert_button.isEnabled() is False

    def test_the_consumer_file_follows_the_correction(self, window, bundle):
        import json

        window.select(1, play=False)
        window.editor.setPlainText("Today we discuss testing.")
        window.save_correction()
        payload = json.loads((bundle.directory / "consumer.json").read_text())
        assert "discuss testing" in payload["text"]
        assert payload["unresolved_reviews"] == 1


class TestComingBack:
    def test_where_checking_got_to_is_remembered(self, window, bundle):
        window.select(2, play=False)
        assert artifact.load_project(bundle.directory)["review_position"] == 2

    def test_it_opens_where_it_was_left(self, application, bundle):
        artifact.remember_review_position(bundle.directory, 2)
        reopened = ReviewWindow(bundle.directory)
        reopened.autoplay_box.setChecked(False)
        try:
            assert reopened.current_index == 2
        finally:
            reopened.close()

    def test_a_nonsense_saved_position_does_not_stop_it_opening(self, application, bundle):
        project = artifact.load_project(bundle.directory)
        project["review_position"] = 999
        artifact.save_project(bundle.directory, project)
        reopened = ReviewWindow(bundle.directory)
        reopened.autoplay_box.setChecked(False)
        try:
            assert reopened.current_index == 2, "clamped to the last segment"
        finally:
            reopened.close()


class TestAudio:
    def test_a_missing_recording_still_allows_correcting(self, window):
        """The source was deleted; the text is still reviewable."""
        assert window.play_button.isEnabled() is False
        assert "no longer at" in window.statusBar().currentMessage()
        window.select(1, play=False)
        window.editor.setPlainText("still editable")
        assert window.save_correction() is True

    def test_playing_asks_for_the_segment_s_own_times(self, window, monkeypatch):
        asked: list[tuple[float, float]] = []
        monkeypatch.setattr(
            window.player, "play_segment",
            lambda start, end: asked.append((start, end)) or True,
        )
        window.select(1, play=False)
        window.play()
        assert asked == [(5.0, 12.0)]

    def test_stopping_what_is_playing(self, window, monkeypatch):
        monkeypatch.setattr(type(window.player), "playing", property(lambda self: True))
        stopped: list[bool] = []
        monkeypatch.setattr(window.player, "stop", lambda: stopped.append(True))
        window.toggle_play()
        assert stopped == [True]


class TestShortcuts:
    def test_every_bound_key_is_in_the_guide(self, window):
        # The window asserts this as it binds; this states it as a rule.
        assert set(REVIEW_SHORTCUTS) >= {
            "Ctrl+Space", "Ctrl+S", "Ctrl+K", "Ctrl+J", "Ctrl+Right", "Ctrl+Left"
        }

    def test_stepping_moves_one_segment_at_a_time(self, window):
        window.select(0, play=False)
        window.step(1)
        assert window.current_index == 1
        window.step(-1)
        assert window.current_index == 0

    def test_stepping_past_the_end_stays_put(self, window):
        window.select(2, play=False)
        window.step(1)
        assert window.current_index == 2


class TestFindingTheAudio:
    def test_audio_kept_beside_the_transcript_is_used(self, application, bundle):
        kept = bundle.directory / "audio.wav"
        kept.write_bytes(b"RIFF....WAVE")
        made = ReviewWindow(bundle.directory)
        try:
            assert made.player.source == kept
            assert made.play_button.isEnabled()
        finally:
            made.close()

    def test_an_audio_file_passed_in_wins(self, application, bundle, tmp_path):
        given = tmp_path / "given.wav"
        given.write_bytes(b"RIFF....WAVE")
        made = ReviewWindow(bundle.directory, audio=given)
        try:
            assert made.player.source == given
        finally:
            made.close()

    def test_the_original_recording_is_decoded_when_there_is_no_kept_copy(
        self, application, bundle, tone_wav
    ):
        import json

        data = json.loads((bundle.directory / "transcript.json").read_text())
        data["media"]["path"] = str(tone_wav)
        (bundle.directory / "transcript.json").write_text(json.dumps(data))

        made = ReviewWindow(bundle.directory)
        try:
            assert made._decoder is not None, "it should decode the original"
            assert "Preparing" in made.statusBar().currentMessage()
            made._decoder.wait(20000)
        finally:
            made.close()

    def test_decoding_really_produces_playable_audio(self, application, bundle, tone_wav, tmp_path):
        """The worker is run directly, so this needs no event loop."""
        from speech2text import media
        from speech2text.ui.review_window import DecodeWorker

        target = tmp_path / "decoded.wav"
        DecodeWorker(Path(tone_wav), target).run()
        assert target.exists()
        assert media.probe(target).sample_rate == media.TARGET_SAMPLE_RATE

    def test_once_decoded_playing_is_offered(self, window, tmp_path):
        prepared = tmp_path / "prepared.wav"
        prepared.write_bytes(b"RIFF....WAVE")
        window._on_decoded(prepared)
        assert window.play_button.isEnabled()
        assert window.player.source == prepared
        assert "Ready" in window.statusBar().currentMessage()

    def test_audio_that_cannot_be_read_is_reported_not_fatal(self, application, bundle, not_media):
        import json

        data = json.loads((bundle.directory / "transcript.json").read_text())
        data["media"]["path"] = str(not_media)
        (bundle.directory / "transcript.json").write_text(json.dumps(data))

        made = ReviewWindow(bundle.directory)
        try:
            if made._decoder is not None:
                made._decoder.wait(20000)
            made.select(1, play=False)
            made.editor.setPlainText("still correctable")
            assert made.save_correction() is True
        finally:
            made.close()


class TestListeningStraightThrough:
    """Most of a recording is fine, so it should be possible to just listen.

    "Play on" keeps going past the end of a segment and moves the highlight
    with the audio, so a long recording can be heard through and stopped
    only where something is wrong.
    """

    def test_it_plays_onwards_rather_than_stopping_at_the_segment(self, window, monkeypatch):
        asked: list[float] = []
        monkeypatch.setattr(
            window.player, "play_onwards", lambda start: asked.append(start) or True
        )
        window.select(1, play=False)
        window.play_onwards()
        assert asked == [5.0], "it starts at the segment and does not bound the end"
        assert window._following is True

    def test_the_highlight_follows_what_is_being_said(self, window, monkeypatch):
        monkeypatch.setattr(window.player, "play_onwards", lambda start: True)
        monkeypatch.setattr(type(window.player), "playing", property(lambda self: True))
        window.select(0, play=False)
        window.play_onwards()

        window._on_position(6.0)
        assert window.current_index == 1, "6s falls inside the second segment"
        window._on_position(15.0)
        assert window.current_index == 2

    def test_following_does_not_start_a_competing_clip(self, window, monkeypatch):
        """Moving the highlight must not itself play that segment."""
        monkeypatch.setattr(window.player, "play_onwards", lambda start: True)
        monkeypatch.setattr(type(window.player), "playing", property(lambda self: True))
        played: list[tuple] = []
        monkeypatch.setattr(
            window.player, "play_segment", lambda start, end: played.append((start, end))
        )
        window.autoplay_box.setChecked(True)
        window.play_onwards()
        window._on_position(6.0)
        assert played == [], "the continuous play must not be interrupted"

    def test_which_segment_covers_a_moment(self, window):
        assert window.segment_at(0.0) == 0
        assert window.segment_at(6.0) == 1
        assert window.segment_at(19.9) == 2
        assert window.segment_at(1000.0) is None

    def test_a_gap_between_segments_looks_ahead(self, application, tmp_path):
        """Silence between segments should point at what is coming, not nothing."""
        from speech2text import artifact as artifact_module

        transcript = Transcript(
            media=MediaIdentity("a.mp3", str(tmp_path / "a.mp3"), 1, 20.0, "mp3"),
            segments=[Segment(0, 0.0, 2.0, "one"), Segment(1, 10.0, 12.0, "two")],
        )
        written = artifact_module.write_bundle(transcript, tmp_path / "gap")
        made = ReviewWindow(written.directory)
        try:
            assert made.segment_at(5.0) == 1, "in the gap, the next one is coming"
        finally:
            made.close()

    def test_reaching_the_end_stops_following(self, window):
        window._following = True
        window._on_clip_finished()
        assert window._following is False
        assert "end of the recording" in window.statusBar().currentMessage()

    def test_stopping_is_offered_while_it_runs(self, window, monkeypatch):
        monkeypatch.setattr(type(window.player), "playing", property(lambda self: True))
        window._following = True
        window._on_playing_changed()
        assert window.follow_button.text() == "■ Stop"

    def test_one_segment_at_a_time_is_still_possible(self, window, monkeypatch):
        bounded: list[tuple[float, float]] = []
        monkeypatch.setattr(
            window.player, "play_segment",
            lambda start, end: bounded.append((start, end)) or True,
        )
        window.select(1, play=False)
        window.play()
        assert bounded == [(5.0, 12.0)]
        assert window._following is False


class TestSpeed:
    def test_the_offered_speeds_include_slower_and_faster(self, window):
        from speech2text.ui.player import SPEEDS

        offered = [window.speed_box.itemData(i) for i in range(window.speed_box.count())]
        assert offered == list(SPEEDS)
        assert min(offered) < 1.0 < max(offered)

    def test_it_starts_at_normal_speed(self, window):
        assert window.speed_box.currentData() == 1.0

    def test_choosing_a_speed_reaches_the_player(self, window):
        window.speed_box.setCurrentIndex(window.speed_box.findData(1.5))
        assert window.player.speed == 1.5

    def test_changing_speed_mid_listen_carries_on_from_where_it_was(self, window, monkeypatch):
        monkeypatch.setattr(type(window.player), "playing", property(lambda self: True))
        monkeypatch.setattr(type(window.player), "position", property(lambda self: 7.5))
        monkeypatch.setattr(window.player, "stop", lambda: None)
        resumed: list[float] = []
        monkeypatch.setattr(
            window.player, "play_onwards", lambda start: resumed.append(start) or True
        )
        window._following = True
        window.speed_box.setCurrentIndex(window.speed_box.findData(1.5))
        assert resumed == [7.5], "it should not jump back to the segment start"

    def test_the_pitch_is_preserved(self):
        """A faster read must stay listenable, not turn into a chipmunk."""
        from pathlib import Path as _Path

        from speech2text.ui.player import SegmentPlayer

        args = SegmentPlayer.arguments(_Path("/a.wav"), 0, None, speed=1.5)
        assert "atempo=1.500" in args
