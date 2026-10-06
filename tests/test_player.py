"""Playing one stretch of a recording."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6 import QtWidgets  # noqa: E402

from speech2text.ui.player import PADDING_SECONDS, SegmentPlayer  # noqa: E402


@pytest.fixture(scope="session")
def application():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


class TestArguments:
    def test_it_plays_exactly_the_segment_with_a_little_room(self):
        args = SegmentPlayer.arguments(Path("/a.wav"), 10.0, 13.0)
        assert args[-1] == "/a.wav"
        start = float(args[args.index("-ss") + 1])
        length = float(args[args.index("-t") + 1])
        assert start == pytest.approx(10.0 - PADDING_SECONDS)
        assert length == pytest.approx(3.0 + 2 * PADDING_SECONDS)

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
        assert player.play(0, 1) is False
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
        assert player.play(0, 1) is False
        assert "ffmpeg" in failures[0]

    def test_nothing_is_playing_to_start_with(self, application):
        assert SegmentPlayer().playing is False

    def test_changing_the_source_stops_what_is_playing(self, application, tmp_path):
        player = SegmentPlayer()
        player.set_source(tmp_path / "one.wav")
        player.set_source(None)
        assert player.source is None and player.playing is False
