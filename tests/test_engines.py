"""The recognizers, and the registry that picks one."""

from __future__ import annotations

import math
from pathlib import Path

import pytest
from conftest import (
    requires_espeak,
    requires_ffmpeg,
    requires_sphinx,
    requires_whisper,
)

from speech2text import engines
from speech2text.engines import base
from speech2text.engines.base import EngineUnavailable, TranscriptionRequest
from speech2text.engines.whisper import MODEL_SIZES
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


@requires_whisper
class TestModelCacheLayout:
    """Each size comes from its own repository, and they are not uniformly named."""

    @pytest.mark.parametrize(
        "model_size,repository",
        [
            ("tiny", "Systran/faster-whisper-tiny"),
            ("base", "Systran/faster-whisper-base"),
            ("large-v3", "Systran/faster-whisper-large-v3"),
            ("large-v3-turbo", "mobiuslabsgmbh/faster-whisper-large-v3-turbo"),
            ("distil-large-v3", "Systran/faster-distil-whisper-large-v3"),
        ],
    )
    def test_a_downloaded_model_is_found_wherever_it_came_from(
        self, tmp_path, model_size, repository
    ):
        from speech2text.engines.whisper import WhisperEngine

        engine = WhisperEngine(model_size, cache_dir=tmp_path)
        assert engine.model_ready() is False

        snapshot = (
            tmp_path / ("models--" + repository.replace("/", "--")) / "snapshots" / "rev"
        )
        snapshot.mkdir(parents=True)
        (snapshot / "model.bin").write_bytes(b"weights")
        assert engine.model_ready() is True, f"{model_size} was not found in the cache"

    @pytest.mark.parametrize("model_size", MODEL_SIZES)
    def test_every_offered_size_is_one_the_library_can_fetch(self, model_size):
        from faster_whisper.utils import _MODELS

        assert model_size in _MODELS, f"{model_size} is not a model faster-whisper knows"

    def test_the_size_on_disk_is_reported(self, tmp_path):
        from speech2text.engines.whisper import downloaded_bytes

        assert downloaded_bytes("base", tmp_path) == 0
        snapshot = (
            tmp_path / "models--Systran--faster-whisper-base" / "snapshots" / "rev"
        )
        snapshot.mkdir(parents=True)
        (snapshot / "model.bin").write_bytes(b"x" * 2048)
        assert downloaded_bytes("base", tmp_path) == 2048

    def test_a_symlinked_snapshot_is_not_counted_twice(self, tmp_path):
        """huggingface_hub stores one blob and links it into the snapshot."""
        from speech2text.engines.whisper import downloaded_bytes

        repo = tmp_path / "models--Systran--faster-whisper-base"
        blobs = repo / "blobs"
        snapshot = repo / "snapshots" / "rev"
        blobs.mkdir(parents=True)
        snapshot.mkdir(parents=True)
        blob = blobs / "abc123"
        blob.write_bytes(b"x" * 4096)
        (snapshot / "model.bin").symlink_to(blob)
        assert downloaded_bytes("base", tmp_path) == 4096

    def test_weights_reached_only_through_a_link_are_still_counted(self, tmp_path):
        """The blob can live outside the folder, so the size must follow links."""
        from speech2text.engines.whisper import downloaded_bytes

        elsewhere = tmp_path / "somewhere-else"
        elsewhere.mkdir()
        blob = elsewhere / "weights"
        blob.write_bytes(b"x" * 8192)

        snapshot = (
            tmp_path / "models--Systran--faster-whisper-base" / "snapshots" / "rev"
        )
        snapshot.mkdir(parents=True)
        (snapshot / "model.bin").symlink_to(blob)
        assert downloaded_bytes("base", tmp_path) == 8192

    def test_a_broken_link_does_not_break_the_count(self, tmp_path):
        from speech2text.engines.whisper import downloaded_bytes

        snapshot = (
            tmp_path / "models--Systran--faster-whisper-base" / "snapshots" / "rev"
        )
        snapshot.mkdir(parents=True)
        (snapshot / "real.bin").write_bytes(b"x" * 512)
        (snapshot / "model.bin").symlink_to(tmp_path / "gone")
        assert downloaded_bytes("base", tmp_path) == 512

    def test_a_local_folder_of_weights_is_accepted(self, tmp_path):
        from speech2text.engines.whisper import WhisperEngine

        local = tmp_path / "my-own-model"
        local.mkdir()
        assert WhisperEngine(str(local), cache_dir=tmp_path).model_ready() is True


@requires_whisper
class TestDeviceFallback:
    """A laptop GPU is often too small for a large model.

    There is no portable way to ask a GPU how much memory it has, so the
    engine tries it and uses the processor instead when it will not fit.
    """

    CUDA_OOM = "CUDA failed with error out of memory"

    @staticmethod
    def _fake_whisper(monkeypatch, fails_on=("cuda",)):
        """Stand in for WhisperModel, failing on the named devices."""
        import faster_whisper

        attempted: list[tuple[str, str]] = []

        class FakeModel:
            def __init__(self, size, device="cpu", compute_type="int8", **kwargs):
                attempted.append((device, compute_type))
                if device in fails_on:
                    raise RuntimeError(TestDeviceFallback.CUDA_OOM)

        monkeypatch.setattr(faster_whisper, "WhisperModel", FakeModel)
        return attempted

    def test_a_gpu_too_small_for_the_model_falls_back_to_the_processor(
        self, tmp_path, monkeypatch
    ):
        from speech2text.engines.whisper import WhisperEngine

        attempted = self._fake_whisper(monkeypatch)
        engine = WhisperEngine("large-v3", device="auto", cache_dir=tmp_path)
        monkeypatch.setattr(engine, "_cuda_available", lambda: True)

        assert engine.load() is not None
        assert [device for device, _ in attempted] == ["cuda", "cpu"]
        assert engine.device_used == "cpu"
        assert "could not load" in engine.fell_back_to_cpu
        assert "out of memory" in engine.fell_back_to_cpu

    def test_the_fallback_uses_a_precision_the_processor_supports(
        self, tmp_path, monkeypatch
    ):
        from speech2text.engines.whisper import WhisperEngine

        attempted = self._fake_whisper(monkeypatch)
        engine = WhisperEngine("large-v3", device="auto", cache_dir=tmp_path)
        monkeypatch.setattr(engine, "_cuda_available", lambda: True)
        engine.load()
        assert attempted[0][1] == "float16", "the GPU attempt uses half precision"
        assert attempted[1][1] == "int8", "the processor attempt does not"

    def test_a_working_gpu_is_used_and_nothing_is_said(self, tmp_path, monkeypatch):
        from speech2text.engines.whisper import WhisperEngine

        attempted = self._fake_whisper(monkeypatch, fails_on=())
        engine = WhisperEngine("base", device="auto", cache_dir=tmp_path)
        monkeypatch.setattr(engine, "_cuda_available", lambda: True)
        engine.load()
        assert [device for device, _ in attempted] == ["cuda"]
        assert engine.device_used == "cuda"
        assert engine.fell_back_to_cpu is None

    def test_choosing_the_gpu_explicitly_is_not_overridden(self, tmp_path, monkeypatch):
        """Asked for the GPU and told why it failed beats a silent downgrade."""
        from speech2text.engines.whisper import WhisperEngine

        attempted = self._fake_whisper(monkeypatch)
        engine = WhisperEngine("large-v3", device="cuda", cache_dir=tmp_path)
        snapshot = (
            tmp_path / "models--Systran--faster-whisper-large-v3" / "snapshots" / "r"
        )
        snapshot.mkdir(parents=True)
        (snapshot / "model.bin").write_bytes(b"weights")

        with pytest.raises(Exception) as caught:
            engine.load()
        assert [device for device, _ in attempted] == ["cuda"]
        assert "--device cpu" in str(caught.value)
        assert "'medium'" in str(caught.value), "it should name a smaller model"

    def test_the_processor_failing_is_reported_not_retried(self, tmp_path, monkeypatch):
        from speech2text.engines.whisper import WhisperEngine

        attempted = self._fake_whisper(monkeypatch, fails_on=("cpu",))
        engine = WhisperEngine("base", device="cpu", cache_dir=tmp_path)
        with pytest.raises(Exception):
            engine.load()
        assert len(attempted) == 1

    def test_what_actually_ran_is_recorded_in_the_artifact(self, tmp_path, monkeypatch):
        """Running on the processor must be visible, not just slow."""
        from speech2text.engines.whisper import WhisperEngine

        engine = WhisperEngine("large-v3", device="auto", cache_dir=tmp_path)
        engine.device_used = "cpu"
        engine.fell_back_to_cpu = "the GPU could not load 'large-v3' (out of memory)"
        described = engine._describe_run()
        assert "on cpu" in described
        assert "out of memory" in described

    @pytest.mark.parametrize(
        "detail,expected",
        [
            ("CUDA failed with error out of memory", True),
            ("cuDNN failed", False),
            ("OOM when allocating", True),
        ],
    )
    def test_running_out_of_memory_is_recognised(self, detail, expected):
        from speech2text.engines.whisper import _is_out_of_memory

        assert _is_out_of_memory(detail) is expected


@requires_whisper
class TestWhisperReadsDecodedAudioDirectly:
    """The pipeline already decoded the audio, so the engine hands over samples.

    Giving faster-whisper a path makes it decode the file again through PyAV,
    whose API has broken under it before — "open() got an unexpected keyword
    argument 'metadata_errors'". Passing samples avoids that path entirely.
    """

    def test_samples_are_read_from_the_decoded_wav(self, spoken_wav):
        numpy = pytest.importorskip("numpy")
        from speech2text.engines.whisper import _read_samples

        samples = _read_samples(Path(spoken_wav))
        assert samples.dtype == numpy.float32
        assert samples.ndim == 1 and len(samples) > 0
        assert -1.0 <= float(samples.min()) and float(samples.max()) <= 1.0

    @requires_ffmpeg
    def test_audio_that_is_not_what_the_pipeline_produces_is_refused(
        self, stereo_audio, tmp_path
    ):
        pytest.importorskip("numpy")
        from speech2text import media
        from speech2text.engines.base import EngineError
        from speech2text.engines.whisper import _read_samples

        wrong = media.decode_to_wav(
            stereo_audio, tmp_path / "wrong.wav", sample_rate=8000, channels=2
        )
        with pytest.raises(EngineError, match="16000 Hz mono"):
            _read_samples(wrong)

    def test_the_model_is_given_samples_and_never_a_path(self, spoken_wav, tmp_path, monkeypatch):
        """The regression guard: a path would re-enter PyAV."""
        numpy = pytest.importorskip("numpy")
        import faster_whisper

        from speech2text.engines.base import TranscriptionRequest
        from speech2text.engines.whisper import WhisperEngine

        handed: dict = {}

        class FakeInfo:
            duration = 1.0
            language = "en"
            language_probability = 0.99

        class FakeModel:
            def __init__(self, *args, **kwargs):
                pass

            def transcribe(self, audio, **kwargs):
                handed["audio"] = audio
                return iter(()), FakeInfo()

        monkeypatch.setattr(faster_whisper, "WhisperModel", FakeModel)
        engine = WhisperEngine("base", device="cpu", cache_dir=tmp_path)
        engine.transcribe(
            TranscriptionRequest(audio=Path(spoken_wav), duration=1.0)
        )
        assert isinstance(handed["audio"], numpy.ndarray)
        assert not isinstance(handed["audio"], (str, Path))


class TestWhisperAvailabilityMessages:
    def test_a_missing_library_says_how_to_install_it(self, monkeypatch):
        import builtins

        from speech2text.engines.whisper import WhisperEngine

        real_import = builtins.__import__

        def refuse(name, *args, **kwargs):
            if name == "faster_whisper":
                raise ImportError("No module named 'faster_whisper'", name="faster_whisper")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", refuse)
        usable, reason = WhisperEngine().availability()
        assert not usable and "[whisper]" in reason

    def test_a_broken_dependency_names_the_real_culprit(self, monkeypatch):
        """Saying "not installed" sends someone to fix the wrong thing."""
        import builtins

        from speech2text.engines.whisper import WhisperEngine

        real_import = builtins.__import__

        def refuse(name, *args, **kwargs):
            if name == "faster_whisper":
                raise ImportError("libavcodec.so.60: cannot open shared object", name="av")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", refuse)
        usable, reason = WhisperEngine().availability()
        assert not usable
        assert "'av'" in reason and "libavcodec" in reason
        assert "force-reinstall" in reason
        assert "is not installed" not in reason


@requires_whisper
class TestGpuFailsDuringRecognition:
    """A GPU can accept the model and still be unable to run it.

    A missing cuBLAS or cuDNN only shows up once inference starts, part of
    the way through the recording — "Library libcublas.so.12 is not found or
    cannot be loaded". Loading succeeding is not proof the GPU works.
    """

    CUBLAS = "Library libcublas.so.12 is not found or cannot be loaded"

    @staticmethod
    def _model_that_runs_only_on(device_that_works, monkeypatch, used):
        import faster_whisper

        class FakeInfo:
            duration = 2.0
            language = "en"
            language_probability = 0.98

        class FakeSegment:
            def __init__(self):
                self.start, self.end = 0.0, 2.0
                self.text = " recognized words "
                self.avg_logprob, self.no_speech_prob, self.words = -0.1, 0.01, ()

        class FakeModel:
            def __init__(self, size, device="cpu", **kwargs):
                self.device = device
                used.append(("load", device))

            def transcribe(self, audio, **kwargs):
                used.append(("run", self.device))
                if self.device != device_that_works:
                    # Lazy, like the real one: the failure comes on iteration.
                    def failing():
                        yield from ()
                        raise RuntimeError(TestGpuFailsDuringRecognition.CUBLAS)

                    return failing(), FakeInfo()
                return iter([FakeSegment()]), FakeInfo()

        monkeypatch.setattr(faster_whisper, "WhisperModel", FakeModel)

    def _engine(self, tmp_path, monkeypatch, device="auto"):
        from speech2text.engines.whisper import WhisperEngine

        engine = WhisperEngine("small", device=device, cache_dir=tmp_path)
        monkeypatch.setattr(engine, "_cuda_available", lambda: True)
        return engine

    def test_it_moves_to_the_processor_and_still_produces_the_text(
        self, spoken_wav, tmp_path, monkeypatch
    ):
        from speech2text.engines.base import TranscriptionRequest

        used: list[tuple[str, str]] = []
        self._model_that_runs_only_on("cpu", monkeypatch, used)
        engine = self._engine(tmp_path, monkeypatch)

        result = engine.transcribe(
            TranscriptionRequest(audio=Path(spoken_wav), duration=2.0)
        )

        assert [s.text for s in result.segments] == ["recognized words"]
        assert used == [("load", "cuda"), ("run", "cuda"), ("load", "cpu"), ("run", "cpu")]
        assert engine.device_used == "cpu"
        assert "libcublas" in engine.fell_back_to_cpu
        assert "on cpu" in result.engine_version

    def test_the_person_is_told_it_moved(self, spoken_wav, tmp_path, monkeypatch):
        from speech2text.engines.base import TranscriptionRequest

        self._model_that_runs_only_on("cpu", monkeypatch, [])
        engine = self._engine(tmp_path, monkeypatch)
        notes: list[str] = []
        engine.transcribe(
            TranscriptionRequest(
                audio=Path(spoken_wav), duration=2.0, on_note=notes.append
            )
        )
        assert any("libcublas" in note for note in notes)

    def test_a_working_gpu_is_not_second_guessed(self, spoken_wav, tmp_path, monkeypatch):
        from speech2text.engines.base import TranscriptionRequest

        used: list[tuple[str, str]] = []
        self._model_that_runs_only_on("cuda", monkeypatch, used)
        engine = self._engine(tmp_path, monkeypatch)
        engine.transcribe(TranscriptionRequest(audio=Path(spoken_wav), duration=2.0))
        assert used == [("load", "cuda"), ("run", "cuda")]
        assert engine.fell_back_to_cpu is None

    def test_choosing_the_gpu_explicitly_is_reported_not_overridden(
        self, spoken_wav, tmp_path, monkeypatch
    ):
        from speech2text.engines.base import EngineError, TranscriptionRequest

        used: list[tuple[str, str]] = []
        self._model_that_runs_only_on("cpu", monkeypatch, used)
        engine = self._engine(tmp_path, monkeypatch, device="cuda")
        with pytest.raises(EngineError) as caught:
            engine.transcribe(TranscriptionRequest(audio=Path(spoken_wav), duration=2.0))
        assert "--device cpu" in str(caught.value)
        assert [step for step, _ in used] == ["load", "run"], "it must not retry"

    def test_a_failure_that_is_not_the_gpu_s_is_not_retried(
        self, spoken_wav, tmp_path, monkeypatch
    ):
        import faster_whisper

        from speech2text.engines.base import EngineError, TranscriptionRequest

        attempts: list[str] = []

        class FakeModel:
            def __init__(self, size, device="cpu", **kwargs):
                pass

            def transcribe(self, audio, **kwargs):
                attempts.append("run")
                raise RuntimeError("the audio is malformed")

        monkeypatch.setattr(faster_whisper, "WhisperModel", FakeModel)
        engine = self._engine(tmp_path, monkeypatch)
        with pytest.raises(EngineError, match="malformed"):
            engine.transcribe(TranscriptionRequest(audio=Path(spoken_wav), duration=2.0))
        assert attempts == ["run"], "only a GPU problem earns a second attempt"

    def test_the_processor_failing_too_is_reported_once(
        self, spoken_wav, tmp_path, monkeypatch
    ):
        from speech2text.engines.base import EngineError, TranscriptionRequest

        used: list[tuple[str, str]] = []
        self._model_that_runs_only_on("nothing", monkeypatch, used)
        engine = self._engine(tmp_path, monkeypatch)
        with pytest.raises(EngineError, match="libcublas"):
            engine.transcribe(TranscriptionRequest(audio=Path(spoken_wav), duration=2.0))
        assert [step for step, _ in used] == ["load", "run", "load", "run"]
