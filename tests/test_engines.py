"""The recognizers, and the registry that picks one."""

from __future__ import annotations

import math
from pathlib import Path

import pytest
from conftest import requires_espeak, requires_ffmpeg, requires_sphinx

from speech2text import engines
from speech2text.engines import base
from speech2text.engines.base import EngineUnavailable, TranscriptionRequest
from speech2text.model import Segment

class TestConfidence:
    def test_a_log_probability_becomes_a_readable_number(self):
        assert base.confidence_from_logprob(0.0) == pytest.approx(1.0)
        assert base.confidence_from_logprob(math.log(0.5)) == pytest.approx(0.5)
        assert base.confidence_from_logprob(None) is None

    def test_it_stays_inside_zero_and_one(self):
        assert base.confidence_from_logprob(5.0) == 1.0
        assert base.confidence_from_logprob(-50.0) == pytest.approx(0.0, abs=1e-9)


class TestReviewQueue:
    def test_only_doubtful_segments_are_queued_worst_first(self):
        segments = [
            Segment(0, 0, 1, "sure", confidence=0.95),
            Segment(1, 1, 2, "unsure", confidence=0.2),
            Segment(2, 2, 3, "shaky", confidence=0.45),
        ]
        issues = base.review_issues(segments, threshold=0.6)
        assert [i.segment_index for i in issues] == [1, 2]

    def test_an_empty_segment_is_queued_even_when_confident(self):
        issues = base.review_issues([Segment(0, 0, 1, "  ", confidence=0.99)])
        assert [i.reason for i in issues] == ["nothing recognized"]

    def test_a_segment_with_no_confidence_is_not_guessed_about(self):
        assert base.review_issues([Segment(0, 0, 1, "cloud text", confidence=None)]) == []


class TestRegistry:
    @pytest.mark.parametrize("name", engines.ENGINE_NAMES)
    def test_every_advertised_engine_can_be_built(self, name):
        engine = engines.create(name)
        assert engine.name == name
        assert engine.title and engine.summary

    def test_an_unknown_engine_lists_the_real_ones(self):
        with pytest.raises(EngineUnavailable, match="whisper"):
            engines.create("telepathy")

    def test_the_default_is_whisper(self):
        assert engines.create().name == engines.DEFAULT_ENGINE == "whisper"

    def test_only_the_cloud_engine_uploads(self):
        described = {d["name"]: d for d in engines.describe_all()}
        assert described["cloud"]["uploads_audio"] is True
        assert described["whisper"]["uploads_audio"] is False
        assert described["sphinx"]["uploads_audio"] is False

    def test_describe_all_covers_every_engine_and_says_why_not(self):
        described = engines.describe_all()
        assert {d["name"] for d in described} == set(engines.ENGINE_NAMES)
        for entry in described:
            assert entry["available"] or entry["reason"], entry["name"]


class TestWhisperAdapter:
    def test_it_knows_whether_the_model_is_already_downloaded(self, tmp_path):
        from speech2text.engines.whisper import WhisperEngine

        engine = WhisperEngine("tiny", cache_dir=tmp_path)
        assert engine.model_ready() is False

    def test_a_cached_model_is_recognised(self, tmp_path):
        from speech2text.engines.whisper import WhisperEngine

        snapshot = tmp_path / "models--Systran--faster-whisper-tiny" / "snapshots" / "abc"
        snapshot.mkdir(parents=True)
        (snapshot / "model.bin").write_bytes(b"weights")
        assert WhisperEngine("tiny", cache_dir=tmp_path).model_ready() is True

    def test_the_cache_can_be_moved_with_an_environment_variable(self, tmp_path, monkeypatch):
        from speech2text.engines import whisper as whisper_module

        monkeypatch.setenv("SPEECH2TEXT_MODEL_CACHE", str(tmp_path / "elsewhere"))
        assert whisper_module.model_cache_dir() == tmp_path / "elsewhere"

    def test_a_failed_download_explains_what_to_do(self, tmp_path, monkeypatch):
        from speech2text.engines.whisper import WhisperEngine

        engine = WhisperEngine("tiny", cache_dir=tmp_path)
        failure = engine._load_failure(RuntimeError("403 Forbidden"))
        message = str(failure)
        assert isinstance(failure, EngineUnavailable)
        assert "huggingface.co" in message
        assert "SPEECH2TEXT_MODEL_CACHE" in message
        assert "sphinx" in message, "it must name the engine that needs no download"

    def test_a_model_that_is_present_but_broken_is_reported_differently(self, tmp_path):
        from speech2text.engines.whisper import WhisperEngine

        snapshot = tmp_path / "models--Systran--faster-whisper-tiny" / "snapshots" / "abc"
        snapshot.mkdir(parents=True)
        (snapshot / "model.bin").write_bytes(b"not really weights")
        engine = WhisperEngine("tiny", cache_dir=tmp_path)
        assert "would not load" in str(engine._load_failure(RuntimeError("bad magic")))


@requires_sphinx
class TestSphinxAdapter:
    def test_it_refuses_a_language_it_cannot_read(self, tone_wav):
        engine = engines.create("sphinx")
        request = TranscriptionRequest(audio=Path(tone_wav), duration=3.0, languages=("fr",))
        with pytest.raises(EngineUnavailable, match="only understands English"):
            engine.transcribe(request)

    def test_pronunciation_variants_and_markers_are_cleaned_away(self):
        from speech2text.engines.sphinx import _clean, _NON_WORDS

        assert _clean("probably(2)") == "probably"
        assert _NON_WORDS.match("<sil>") and _NON_WORDS.match("[NOISE]")
        assert not _NON_WORDS.match("hello")

    @requires_ffmpeg
    @requires_espeak
    def test_it_really_transcribes_real_speech(self, spoken_wav):
        """End to end against genuine audio, not a stub.

        PocketSphinx is not accurate enough to assert the words, so this
        checks the shape of what it returns: ordered, timed, bounded
        segments that carry words and a confidence.
        """
        engine = engines.create("sphinx")
        seen: list[float] = []
        result = engine.transcribe(
            TranscriptionRequest(
                audio=Path(spoken_wav), duration=8.0, progress=seen.append
            )
        )
        assert result.segments, "real speech must produce at least one segment"
        assert result.language == "en"
        assert result.model and result.engine_version

        for position, segment in enumerate(result.segments):
            assert segment.index == position, "indexes must be consecutive"
            assert segment.text.strip()
            assert segment.end > segment.start
            assert 0.0 <= segment.confidence <= 1.0
            assert segment.words, "word timings are what the review view needs"
            assert segment.words[0].start >= segment.start - 0.01
            assert segment.words[-1].end <= segment.end + 0.01

        starts = [s.start for s in result.segments]
        assert starts == sorted(starts), "segments must be in time order"
        assert seen and seen[-1] == 1.0

    @requires_ffmpeg
    def test_audio_with_no_speech_does_not_invent_a_transcript(self, tone_wav):
        result = engines.create("sphinx").transcribe(
            TranscriptionRequest(audio=Path(tone_wav), duration=3.0)
        )
        words = " ".join(s.text for s in result.segments).split()
        assert len(words) <= 4, f"a sine tone produced {words!r}"
