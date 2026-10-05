"""One recording, end to end."""

from __future__ import annotations

import json

import pytest
from conftest import requires_espeak, requires_ffmpeg, requires_sphinx

from speech2text import artifact, media
from speech2text.engines.base import EngineError, SpeechEngine
from speech2text.pipeline import (
    Cancelled,
    Run,
    TranscribeOptions,
    transcribe_file,
    transcribe_many,
)


@requires_ffmpeg
class TestOneRecording:
    def test_a_bundle_is_produced_with_everything_in_it(self, tone_wav, tmp_path, fake_engine):
        bundle = transcribe_file(tone_wav, tmp_path / "out", engine=fake_engine)
        assert bundle.json_path.exists()
        assert bundle.text_path.read_text().strip() == "first segment\nsecond segment"
        assert json.loads(bundle.consumer_path.read_text())["kind"] == "speech-transcript"

    def test_the_source_is_identified_by_content_not_name(
        self, misnamed_video, tmp_path, fake_engine
    ):
        bundle = transcribe_file(misnamed_video, tmp_path / "out", engine=fake_engine)
        identity = bundle.transcript.media
        assert identity.name == "recording.dat"
        assert identity.container == "mov", "the .dat suffix must not decide this"
        assert identity.has_video and identity.video_codec == "h264"

    def test_the_recording_is_fingerprinted(self, tone_wav, tmp_path, fake_engine):
        bundle = transcribe_file(tone_wav, tmp_path / "out", engine=fake_engine)
        assert bundle.transcript.media.sha256 == media.sha256(tone_wav)

    def test_the_engine_receives_decoded_sixteen_kilohertz_audio(
        self, stereo_audio, tmp_path
    ):
        """Checked while the engine runs: the audio is temporary and is gone after."""
        from speech2text.engines.base import EngineResult, SpeechEngine

        observed = {}

        class Inspecting(SpeechEngine):
            name = "inspecting"

            def transcribe(self, request):
                info = media.probe(request.audio)
                observed["rate"] = info.sample_rate
                observed["channels"] = info.channels
                observed["path"] = request.audio
                return EngineResult(segments=[], language="en")

        transcribe_file(stereo_audio, tmp_path / "out", engine=Inspecting())
        assert observed["rate"] == 16000
        assert observed["channels"] == 1
        assert not observed["path"].exists(), "the decoded audio must be cleaned up"

    def test_the_chosen_language_reaches_the_engine(self, tone_wav, tmp_path, fake_engine):
        transcribe_file(
            tone_wav, tmp_path / "out",
            TranscribeOptions(languages=["fr"]), engine=fake_engine,
        )
        assert fake_engine.requests[0].languages == ("fr",)

    def test_automatic_language_passes_nothing_and_records_what_was_found(
        self, tone_wav, tmp_path, fake_engine
    ):
        bundle = transcribe_file(tone_wav, tmp_path / "out", engine=fake_engine)
        assert fake_engine.requests[0].languages == ()
        assert bundle.transcript.language_detected == "en"
        assert bundle.transcript.languages_requested == ()

    def test_doubtful_segments_become_a_review_queue(self, tone_wav, tmp_path, fake_engine):
        bundle = transcribe_file(tone_wav, tmp_path / "out", engine=fake_engine)
        assert [i.segment_index for i in bundle.transcript.unresolved_issues()] == [1]

    def test_progress_covers_every_stage_and_never_goes_backwards(
        self, tone_wav, tmp_path, fake_engine
    ):
        seen = []
        transcribe_file(
            tone_wav, tmp_path / "out", engine=fake_engine,
            on_progress=lambda p: seen.append((p.stage, p.fraction)),
        )
        stages = [stage for stage, _ in seen]
        for expected in ("identifying", "decoding", "recognizing", "writing"):
            assert expected in stages, expected
        fractions = [fraction for _, fraction in seen]
        assert fractions == sorted(fractions)
        assert fractions[-1] == 1.0

    def test_the_decoded_audio_is_not_left_behind(self, tone_wav, tmp_path, fake_engine):
        bundle = transcribe_file(tone_wav, tmp_path / "out", engine=fake_engine)
        assert not (bundle.directory / "audio.wav").exists()

    def test_the_audio_can_be_kept_when_asked(self, tone_wav, tmp_path, fake_engine):
        bundle = transcribe_file(
            tone_wav, tmp_path / "out",
            TranscribeOptions(keep_audio=True), engine=fake_engine,
        )
        assert (bundle.directory / "audio.wav").exists()

    def test_how_long_it_took_is_recorded(self, tone_wav, tmp_path, fake_engine):
        bundle = transcribe_file(tone_wav, tmp_path / "out", engine=fake_engine)
        assert bundle.transcript.processing_seconds >= 0


@requires_ffmpeg
class TestFailures:
    def test_a_file_with_no_sound_is_refused_before_any_work(
        self, silent_video, tmp_path, fake_engine
    ):
        with pytest.raises(media.MediaError, match="no sound track"):
            transcribe_file(silent_video, tmp_path / "out", engine=fake_engine)
        assert fake_engine.requests == []

    def test_an_engine_that_cannot_run_is_reported(self, tone_wav, tmp_path):
        class Unavailable(SpeechEngine):
            name = "broken"

            def availability(self):
                return False, "this engine needs something it does not have"

            def transcribe(self, request):  # pragma: no cover - never reached
                raise AssertionError("must not be called")

        with pytest.raises(EngineError, match="does not have"):
            transcribe_file(tone_wav, tmp_path / "out", engine=Unavailable())


@requires_ffmpeg
class TestStopping:
    def test_a_run_stops_when_asked(self, tone_wav, tmp_path, fake_engine):
        run = Run(tone_wav, tmp_path / "out", engine=fake_engine)
        run.on_progress = lambda progress: run.stop()
        with pytest.raises(Cancelled):
            run.execute()

    def test_stopping_leaves_no_half_written_transcript(self, tone_wav, tmp_path, fake_engine):
        destination = tmp_path / "out"
        run = Run(tone_wav, destination, engine=fake_engine)
        run.on_progress = lambda progress: run.stop()
        with pytest.raises(Cancelled):
            run.execute()
        assert not (destination / artifact.TRANSCRIPT_JSON).exists()


@requires_ffmpeg
class TestBatches:
    def test_a_batch_shares_one_run_folder_numbered_in_order(
        self, tone_wav, misnamed_video, tmp_path, fake_engine
    ):
        bundles = transcribe_many(
            [tone_wav, misnamed_video], tmp_path, engine=fake_engine
        )
        assert len(bundles) == 2
        assert bundles[0].directory.parent == bundles[1].directory.parent
        assert bundles[0].directory.name.startswith("001-")
        assert bundles[1].directory.name.startswith("002-")

    def test_one_bad_file_does_not_lose_the_rest(
        self, silent_video, tone_wav, tmp_path, fake_engine
    ):
        reported: list[tuple] = []
        bundles = transcribe_many(
            [silent_video, tone_wav], tmp_path, engine=fake_engine,
            on_file=lambda position, path, bundle, error: reported.append(
                (path.name, bundle is not None, error)
            ),
        )
        assert len(bundles) == 1
        assert reported[0][0] == "silent.mp4" and reported[0][1] is False
        assert isinstance(reported[0][2], media.MediaError)
        assert reported[1][1] is True


@requires_ffmpeg
@requires_espeak
@requires_sphinx
class TestWithARealEngine:
    def test_real_speech_through_the_whole_pipeline(self, spoken_wav, tmp_path):
        """No stubs: real audio, a real recognizer, a real bundle on disk."""
        bundle = transcribe_file(
            spoken_wav, tmp_path / "out", TranscribeOptions(engine="sphinx")
        )
        transcript = bundle.transcript

        assert transcript.segments, "real speech produced no segments"
        assert transcript.word_count() > 0
        assert transcript.engine == "sphinx"
        assert transcript.language_detected == "en"

        text = bundle.text_path.read_text().strip()
        assert text and text == transcript.corrected_text().strip()
        assert bundle.raw_text_path.read_text() == transcript.raw_text()

        consumer = json.loads(bundle.consumer_path.read_text())
        assert consumer["text"] == text
        assert consumer["word_count"] == transcript.word_count()

        # And the bundle reopens as what was written.
        assert artifact.load(bundle.directory).to_dict() == transcript.to_dict()


@requires_ffmpeg
class TestReadingBeforeItFinishes:
    """A long recording should be readable before the run ends."""

    def test_the_preview_is_written_while_the_engine_works(self, tone_wav, tmp_path):
        from speech2text.engines.base import EngineResult, SpeechEngine
        from speech2text.model import Segment

        destination = tmp_path / "out"
        seen: list[str] = []

        class Streaming(SpeechEngine):
            name = "streaming"

            def transcribe(self, request):
                first = Segment(0, 0, 1, "the first thing said")
                request.emit(first)
                seen.append(
                    (destination / artifact.TRANSCRIPT_PARTIAL_TXT).read_text()
                )
                second = Segment(1, 1, 2, "and the second")
                request.emit(second)
                seen.append(
                    (destination / artifact.TRANSCRIPT_PARTIAL_TXT).read_text()
                )
                return EngineResult(segments=[first, second], language="en")

        bundle = transcribe_file(tone_wav, destination, engine=Streaming())

        assert "the first thing said" in seen[0]
        assert "and the second" not in seen[0], "the preview grows as it goes"
        assert "and the second" in seen[1]
        assert "in progress" in seen[0], "the preview must explain itself"

    def test_the_preview_is_replaced_by_the_real_transcript(self, tone_wav, tmp_path, fake_engine):
        destination = tmp_path / "out"
        bundle = transcribe_file(tone_wav, destination, engine=fake_engine)
        assert not (destination / artifact.TRANSCRIPT_PARTIAL_TXT).exists()
        assert bundle.text_path.exists()

    def test_the_preview_can_be_turned_off(self, tone_wav, tmp_path, fake_engine):
        transcribe_file(
            tone_wav, tmp_path / "out",
            TranscribeOptions(write_partial=False), engine=fake_engine,
        )
        assert fake_engine.requests[0].on_segment is None

    def test_an_engine_that_cannot_stream_still_works(self, tone_wav, tmp_path):
        """Emitting is optional; the returned result is what counts."""
        from speech2text.engines.base import EngineResult, SpeechEngine
        from speech2text.model import Segment

        class Silent(SpeechEngine):
            name = "silent"

            def transcribe(self, request):
                return EngineResult(
                    segments=[Segment(0, 0, 1, "all at once")], language="en"
                )

        bundle = transcribe_file(tone_wav, tmp_path / "out", engine=Silent())
        assert bundle.text_path.read_text().strip() == "all at once"
