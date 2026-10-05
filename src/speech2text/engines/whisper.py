"""Whisper, through faster-whisper. The default recognizer.

The model is downloaded once and then runs entirely on this machine: nothing
is uploaded, and no network is needed after the first run.
"""

from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path

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


def _is_downloaded(model_size: str, cache: Path) -> bool:
    """True when the weights are already on disk, so a run needs no network."""
    if Path(model_size).expanduser().is_dir():
        return True
    # huggingface_hub lays a repository out as models--<org>--<name>/snapshots/<rev>.
    for candidate in cache.glob(f"models--*faster-whisper-{model_size}"):
        if any(candidate.glob("snapshots/*/model.bin")):
            return True
    return (cache / model_size / "model.bin").exists()


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

    # ---- availability ---------------------------------------------------

    def availability(self) -> tuple[bool, str]:
        try:
            import faster_whisper  # noqa: F401
        except ImportError:
            return False, (
                "faster-whisper is not installed. Install it with:\n"
                "    pip install -e '.[whisper]'"
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
        """Load the model, downloading it the first time it is used."""
        if self._model is not None:
            return self._model
        self.require_available()
        from faster_whisper import WhisperModel

        device, compute_type = self._resolve_device()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        try:
            self._model = WhisperModel(
                self.model_size,
                device=device,
                compute_type=compute_type,
                download_root=str(self.cache_dir),
                cpu_threads=self.cpu_threads,
            )
        except Exception as exc:
            raise self._load_failure(exc) from exc
        return self._model

    def _load_failure(self, exc: Exception) -> EngineError:
        """Say what to do about it, rather than re-raising a library traceback."""
        detail = str(exc).strip() or exc.__class__.__name__
        if self.model_ready():
            return EngineError(
                f"the {self.model_size} model is on disk but would not load: {detail}"
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
        language = request.language
        try:
            raw_segments, info = model.transcribe(
                str(request.audio),
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
            engine_version=_library_version(),
        )


def _library_version() -> str | None:
    try:
        from importlib.metadata import version

        return f"faster-whisper {version('faster-whisper')}"
    except Exception:  # pragma: no cover
        return None
