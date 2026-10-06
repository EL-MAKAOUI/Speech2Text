"""Playing one stretch of a recording."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6 import QtWidgets  # noqa: E402

from speech2text.ui.player import (  # noqa: E402
    PADDING_SECONDS,
    SPEEDS,
    SegmentPlayer,
)


@pytest.fixture(scope="session")
def application():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


class TestArguments:
    def test_it_plays_exactly_the_segment(self):
        args = SegmentPlayer.arguments(Path("/a.wav"), 10.0, 13.0)
        assert args[-1] == "/a.wav"
        assert float(args[args.index("-ss") + 1]) == pytest.approx(10.0)
        assert float(args[args.index("-t") + 1]) == pytest.approx(3.0)

    def test_padding_gives_the_first_consonant_room(self):
        """A clip cut on the boundary loses the sound most often misheard."""
        args = SegmentPlayer.arguments(Path("/a.wav"), 10.0, 13.0, padding=PADDING_SECONDS)
        assert float(args[args.index("-ss") + 1]) == pytest.approx(10.0 - PADDING_SECONDS)
        assert float(args[args.index("-t") + 1]) == pytest.approx(3.0 + 2 * PADDING_SECONDS)

    def test_playing_onwards_sets_no_end(self):
        args = SegmentPlayer.arguments(Path("/a.wav"), 10.0, None)
        assert "-t" not in args, "it must run to the end of the recording"

    def test_normal_speed_adds_no_filter(self):
        assert "-af" not in SegmentPlayer.arguments(Path("/a.wav"), 0, 1, speed=1.0)

    @pytest.mark.parametrize("speed", [0.5, 1.25, 2.0])
    def test_another_speed_keeps_the_pitch(self, speed):
        args = SegmentPlayer.arguments(Path("/a.wav"), 0, 1, speed=speed)
        assert f"atempo={speed:.3f}" in args

    def test_it_stops_on_its_own(self):
        assert "-autoexit" in SegmentPlayer.arguments(Path("/a.wav"), 0, 1)

    def test_it_opens_no_window(self):
        assert "-nodisp" in SegmentPlayer.arguments(Path("/a.wav"), 0, 1)

    def test_the_start_never_goes_before_the_beginning(self):
        args = SegmentPlayer.arguments(Path("/a.wav"), 0.0, 1.0)
        assert float(args[args.index("-ss") + 1]) == 0.0

    def test_a_segment_with_no_length_is_still_audible(self):
        args = SegmentPlayer.arguments(Path("/a.wav"), 5.0, 5.0)
        assert float(args[args.index("-t") + 1]) > 0


class TestPlayer:
    def test_it_is_not_ready_without_a_source(self, application):
        assert SegmentPlayer().ready is False

    def test_a_missing_file_is_reported_rather_than_played(self, application, tmp_path):
        player = SegmentPlayer()
        player.set_source(tmp_path / "gone.wav")
        failures = []
        player.failed.connect(failures.append)
        assert player.play_segment(0, 1) is False
        assert failures and "not available" in failures[0]

    def test_missing_ffplay_says_where_it_comes_from(self, application, tmp_path, monkeypatch):
        from speech2text.ui import player as player_module

        audio = tmp_path / "a.wav"
        audio.write_bytes(b"RIFF")
        monkeypatch.setattr(player_module, "playback_available", lambda: False)
        player = SegmentPlayer()
        player.set_source(audio)
        failures = []
        player.failed.connect(failures.append)
        assert player.play_segment(0, 1) is False
        assert "ffmpeg" in failures[0]

    def test_nothing_is_playing_to_start_with(self, application):
        assert SegmentPlayer().playing is False

    def test_changing_the_source_stops_what_is_playing(self, application, tmp_path):
        player = SegmentPlayer()
        player.set_source(tmp_path / "one.wav")
        player.set_source(None)
        assert player.source is None and player.playing is False


class TestPosition:
    def test_it_starts_where_it_was_asked_to(self, application):
        player = SegmentPlayer()
        player._from = 12.0
        assert player.position == 12.0

    def test_speed_is_clamped_to_what_the_filter_takes(self, application):
        player = SegmentPlayer()
        player.set_speed(99.0)
        assert player.speed == max(SPEEDS)
        player.set_speed(0.01)
        assert player.speed == min(SPEEDS)

    def test_a_finished_clip_leaves_the_position_at_its_end(self, application):
        """So that playing on from there continues rather than repeats."""
        player = SegmentPlayer()
        player._from, player._until = 5.0, 9.0
        player._on_finished()
        assert player.position == 9.0

    def test_finishing_on_its_own_is_told_apart_from_being_stopped(self, application):
        player = SegmentPlayer()
        events: list[str] = []
        player.finished.connect(lambda: events.append("finished"))
        player.stopped.connect(lambda: events.append("stopped"))
        player._on_finished()
        assert events == ["finished", "stopped"]

    def test_stopping_reports_only_that(self, application, tmp_path):
        player = SegmentPlayer()
        events: list[str] = []
        player.finished.connect(lambda: events.append("finished"))
        player.stopped.connect(lambda: events.append("stopped"))
        player.stop()
        assert "finished" not in events
