"""Whisper, through faster-whisper. The default recognizer.

The model is downloaded once and then runs entirely on this machine: nothing
is uploaded, and no network is needed after the first run.
"""

from __future__ import annotations

import os
import wave
from dataclasses import replace
from pathlib import Path

from ..media import TARGET_SAMPLE_RATE
from ..model import Segment, Word
from .base import (
    EngineError,
    EngineResult,
    EngineUnavailable,
    SpeechEngine,
    TranscriptionRequest,
    confidence_from_logprob,
)

#: Bigger is more accurate and slower. ``base`` is the compromise that fits a
#: laptop; ``large-v3`` is the one to use when the words actually matter.
MODEL_SIZES = (
    "tiny", "base", "small", "medium",
    "large-v3", "large-v3-turbo", "distil-large-v3",
)
DEFAULT_MODEL = "base"

_CACHE_ENV = "SPEECH2TEXT_MODEL_CACHE"


def model_cache_dir() -> Path:
    """Where downloaded weights live. Set ``SPEECH2TEXT_MODEL_CACHE`` to move them."""
    override = os.environ.get(_CACHE_ENV)
    if override:
        return Path(override).expanduser()
    base = os.environ.get("XDG_CACHE_HOME") or (Path.home() / ".cache")
    return Path(base).expanduser() / "speech2text" / "models"


def _cache_folders(model_size: str) -> list[str]:
    """The cache folder names a model's weights could be under.

    faster-whisper knows which repository serves each size, and they are not
    uniformly named — ``distil-large-v3`` comes from
    ``Systran/faster-distil-whisper-large-v3`` and ``large-v3-turbo`` from a
    different organisation again. Asking the library beats pattern-matching
    on the name; the glob is only a fallback for when it cannot be asked.
    """
    try:
        from faster_whisper.utils import _MODELS

        repository = _MODELS.get(model_size)
    except Exception:  # pragma: no cover - the library is optional
        repository = None
    if repository:
        return ["models--" + repository.replace("/", "--")]
    return [f"models--*faster-whisper-{model_size}"]


def _is_downloaded(model_size: str, cache: Path) -> bool:
    """True when the weights are already on disk, so a run needs no network."""
    if Path(model_size).expanduser().is_dir():
        return True
    # huggingface_hub lays a repository out as models--<org>--<name>/snapshots/<rev>.
    for pattern in _cache_folders(model_size):
        for candidate in cache.glob(pattern):
            if any(candidate.glob("snapshots/*/model.bin")):
                return True
    return (cache / model_size / "model.bin").exists()


def downloaded_bytes(model_size: str, cache: Path) -> int:
    """How much disk a model's weights take, or 0 when it is not here.

    A cached model is the same bytes reachable by more than one name: a
    snapshot entry is usually a link to a blob, and the blob may not even sit
    inside the folder being measured. Following the links and counting each
    underlying file once gets the real figure whatever layout the hub uses.
    """
    total = 0
    counted: set[tuple[int, int]] = set()
    for pattern in _cache_folders(model_size):
        for candidate in cache.glob(pattern):
            for item in candidate.rglob("*"):
                try:
                    if not item.is_file():        # follows links; skips broken ones
                        continue
                    info = item.stat()
                except OSError:  # pragma: no cover - unreadable entry
                    continue
                identity = (info.st_dev, info.st_ino)
                if identity in counted:
                    continue
                counted.add(identity)
                total += info.st_size
    return total


class WhisperEngine(SpeechEngine):
    name = "whisper"
    title = "Standard (Whisper)"
    summary = "Runs on this machine. Accurate, and supports 99 languages."
    uploads_audio = False

    def __init__(
        self,
        model_size: str = DEFAULT_MODEL,
        *,
        device: str = "auto",
        compute_type: str | None = None,
        beam_size: int = 5,
        vad_filter: bool = True,
        cache_dir: Path | None = None,
        cpu_threads: int = 0,
    ) -> None:
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.beam_size = beam_size
        self.vad_filter = vad_filter
        self.cpu_threads = cpu_threads
        self.cache_dir = Path(cache_dir) if cache_dir else model_cache_dir()
        self._model = None
        #: Where the model actually ended up running, known after loading.
        self.device_used: str | None = None
        #: Set when the GPU could not hold the model and the CPU was used
        #: instead, so the run can say so rather than appearing to just be slow.
        self.fell_back_to_cpu: str | None = None

    # ---- availability ---------------------------------------------------

    def availability(self) -> tuple[bool, str]:
        try:
            import faster_whisper  # noqa: F401
        except ImportError as exc:
            # faster-whisper pulls in ctranslate2, tokenizers and PyAV. When
            # one of those is the thing that is broken, saying "it is not
            # installed" sends someone off to fix the wrong problem.
            missing = getattr(exc, "name", None) or ""
            if missing in ("", "faster_whisper"):
                return False, (
                    "faster-whisper is not installed. Install it with:\n"
                    "    pip install -e '.[whisper]'"
                )
            return False, (
                f"faster-whisper cannot load: {missing!r} is missing or broken "
                f"({exc}).\nReinstalling usually fixes it:\n"
                f"    pip install --force-reinstall -e '.[whisper]'"
            )
        return True, ""

    def model_ready(self) -> bool:
        """True when this model's weights are cached and no download is needed."""
        return _is_downloaded(self.model_size, self.cache_dir)

    # ---- loading --------------------------------------------------------

    def _resolve_device(self) -> tuple[str, str]:
        device = self.device
        if device == "auto":
            device = "cuda" if self._cuda_available() else "cpu"
        compute = self.compute_type or ("float16" if device == "cuda" else "int8")
        return device, compute

    @staticmethod
    def _cuda_available() -> bool:
        try:
            import ctranslate2

            return ctranslate2.get_cuda_device_count() > 0
        except Exception:  # pragma: no cover - depends on the machine
            return False

    def load(self):
        """Load the model, downloading it the first time it is used.

        A laptop GPU is often too small for a large model, and there is no
        portable way to ask how much memory it has: ctranslate2 does not
        report it and torch is not a dependency here. So when the device was
        chosen automatically, the GPU is tried and the CPU is used instead if
        it will not fit. Slower is better than refusing to run.
        """
        if self._model is not None:
            return self._model
        self.require_available()
        from faster_whisper import WhisperModel

        device, compute_type = self._resolve_device()
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        attempts = [(device, compute_type)]
        if device == "cuda" and self.device == "auto":
            attempts.append(("cpu", self.compute_type or "int8"))

        failure: Exception | None = None
        for attempt, (on_device, with_compute) in enumerate(attempts):
            try:
                model = WhisperModel(
                    self.model_size,
                    device=on_device,
                    compute_type=with_compute,
                    download_root=str(self.cache_dir),
                    cpu_threads=self.cpu_threads,
                )
            except Exception as exc:
                failure = exc
                continue
            self._model = model
            self.device_used = on_device
            if attempt:
                self.fell_back_to_cpu = (
                    f"the GPU could not load '{self.model_size}' "
                    f"({_short(failure)}), so it is running on the CPU"
                )
            return self._model

        raise self._load_failure(failure) from failure

    def _load_failure(self, exc: Exception | None) -> EngineError:
        """Say what to do about it, rather than re-raising a library traceback."""
        detail = _short(exc)
        if self.model_ready():
            advice = ""
            if _is_out_of_memory(detail):
                smaller = _smaller_than(self.model_size)
                advice = (
                    f"\nThere was not enough memory for this model. "
                    f"Run it on the processor instead with '--device cpu', "
                    f"or use a smaller model"
                    + (f" such as '{smaller}'" if smaller else "")
                    + "."
                )
            return EngineError(
                f"the {self.model_size} model is on disk but would not load: "
                f"{detail}{advice}"
            )
        return EngineUnavailable(
            f"the '{self.model_size}' model is not downloaded yet and fetching it "
            f"failed: {detail}\n"
            f"The weights come from huggingface.co, so the first run of a model "
            f"needs network access to it. Once downloaded to\n"
            f"    {self.cache_dir}\n"
            f"everything afterwards is offline. If that host is blocked here, "
            f"copy the cache folder from a machine that can reach it, point "
            f"{_CACHE_ENV} at it, or use '--engine sphinx', which needs no "
            f"download."
        )

    # ---- recognition ----------------------------------------------------

    def transcribe(self, request: TranscriptionRequest) -> EngineResult:
        model = self.load()
        if self.fell_back_to_cpu:
            # Running on the CPU looks like nothing but slowness unless it is
            # said; the artifact records what actually ran.
            request.note(self.fell_back_to_cpu)
        language = request.language
        try:
            raw_segments, info = model.transcribe(
                _read_samples(request.audio),
                language=language,
                beam_size=self.beam_size,
                vad_filter=self.vad_filter,
                word_timestamps=True,
                initial_prompt=request.initial_prompt,
            )
        except Exception as exc:  # pragma: no cover - library-specific
            raise EngineError(f"whisper could not read the audio: {exc}") from exc

        total = getattr(info, "duration", None) or request.duration or 0.0
        detected = getattr(info, "language", None)
        probability = getattr(info, "language_probability", None)

        segments: list[Segment] = []
        # faster-whisper recognizes lazily, so this loop is the actual work and
        # the only place progress can be reported from.
        for position, raw in enumerate(raw_segments):
            text = (raw.text or "").strip()
            words = tuple(
                Word(
                    text=w.word,
                    start=float(w.start),
                    end=float(w.end),
                    probability=getattr(w, "probability", None),
                )
                for w in (getattr(raw, "words", None) or ())
                if w.start is not None and w.end is not None
            )
            segment = Segment(
                index=position,
                start=float(raw.start),
                end=float(raw.end),
                text=text,
                words=words,
                confidence=confidence_from_logprob(getattr(raw, "avg_logprob", None)),
                language=detected,
                no_speech_probability=getattr(raw, "no_speech_prob", None),
            )
            segments.append(segment)
            request.emit(segment)
            if total:
                request.report(float(raw.end) / total)
        request.report(1.0)

        # Drop segments Whisper itself believes are silence misheard as words.
        segments = [
            s for s in segments
            if s.text and not (
                (s.no_speech_probability or 0) > 0.9 and (s.confidence or 1) < 0.3
            )
        ]
        segments = [replace(s, index=i) for i, s in enumerate(segments)]

        return EngineResult(
            segments=segments,
            language=detected,
            language_probability=probability,
            model=self.model_size,
            engine_version=self._describe_run(),
        )

    def _describe_run(self) -> str:
        """What actually ran, for the artifact: library, device, any fallback."""
        parts = [_library_version() or "faster-whisper"]
        if self.device_used:
            parts.append(f"on {self.device_used}")
        description = " ".join(parts)
        if self.fell_back_to_cpu:
            description += f" ({self.fell_back_to_cpu})"
        return description


def _read_samples(path: Path):
    """The decoded audio as the float32 samples the model expects.

    The pipeline has already produced 16 kHz mono 16-bit PCM with ffmpeg, so
    the samples are handed over directly rather than giving faster-whisper a
    path and letting it decode the file a second time with PyAV. That skips
    a redundant decode, and it keeps this engine clear of PyAV's API, which
    has changed under faster-whisper more than once.
    """
    import numpy

    with wave.open(str(path), "rb") as handle:
        channels = handle.getnchannels()
        width = handle.getsampwidth()
        rate = handle.getframerate()
        frames = handle.readframes(handle.getnframes())

    if channels != 1 or width != 2 or rate != TARGET_SAMPLE_RATE:
        raise EngineError(
            f"expected {TARGET_SAMPLE_RATE} Hz mono 16-bit audio, got "
            f"{rate} Hz, {channels} channel(s), {width * 8}-bit"
        )
    # WAV PCM is little-endian; say so rather than depending on the platform.
    return numpy.frombuffer(frames, dtype="<i2").astype(numpy.float32) / 32768.0


def _short(exc: Exception | None) -> str:
    if exc is None:  # pragma: no cover - only reached with no attempts
        return "no reason given"
    return str(exc).strip() or exc.__class__.__name__


def _is_out_of_memory(detail: str) -> bool:
    lowered = detail.lower()
    return "out of memory" in lowered or "oom" in lowered.split()


def _smaller_than(model_size: str) -> str | None:
    """The next size down, to suggest when one will not fit."""
    ladder = ("tiny", "base", "small", "medium", "large-v3")
    if model_size in ladder:
        position = ladder.index(model_size)
        return ladder[position - 1] if position else None
    # The large variants are all of a size; point at one that fits a laptop.
    return "small"


def _library_version() -> str | None:
    try:
        from importlib.metadata import version

        return f"faster-whisper {version('faster-whisper')}"
    except Exception:  # pragma: no cover
        return None
