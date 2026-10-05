"""PocketSphinx: the recognizer that needs no download.

Its English model ships inside the Python package, so this engine works on a
machine with no network at all and is what makes an offline install possible.
It is markedly less accurate than Whisper and understands only English — it is
here as a fallback and a way to try the application immediately, not as the
recommended way to transcribe something that matters.
"""

from __future__ import annotations

import re
import wave
from dataclasses import replace

from ..model import Segment, Word
from .base import (
    EngineError,
    EngineResult,
    EngineUnavailable,
    SpeechEngine,
    TranscriptionRequest,
)

# PocketSphinx reports times in frames; its default rate is 100 per second.
_FRAMES_PER_SECOND = 100.0

# Tokens the decoder emits that are not words.
_NON_WORDS = re.compile(r"^(<[^>]*>|\[[^\]]*\]|\+\+.*\+\+)$")
# "probably(2)" is the second pronunciation of "probably", not a different word.
_VARIANT = re.compile(r"\(\d+\)$")

#: A gap longer than this ends a segment, which is how sentences are found.
_SILENCE_GAP = 0.35
#: No segment runs longer than this, so subtitles stay readable.
_MAX_SEGMENT_SECONDS = 12.0


class SphinxEngine(SpeechEngine):
    name = "sphinx"
    title = "Offline (PocketSphinx)"
    summary = "English only, no download, low accuracy. Works with no network."
    uploads_audio = False

    def __init__(self, *, silence_gap: float = _SILENCE_GAP,
                 max_segment_seconds: float = _MAX_SEGMENT_SECONDS) -> None:
        self.silence_gap = silence_gap
        self.max_segment_seconds = max_segment_seconds

    def availability(self) -> tuple[bool, str]:
        try:
            import pocketsphinx  # noqa: F401
        except ImportError:
            return False, (
                "pocketsphinx is not installed. Install it with:\n"
                "    pip install -e '.[sphinx]'"
            )
        return True, ""

    def transcribe(self, request: TranscriptionRequest) -> EngineResult:
        self.require_available()
        language = request.language
        if language and language != "en":
            raise EngineUnavailable(
                f"the offline engine only understands English, and this run asked "
                f"for {language!r}. Use the Whisper engine for other languages."
            )

        from pocketsphinx import Decoder

        try:
            audio = wave.open(str(request.audio), "rb")
        except (wave.Error, OSError) as exc:
            raise EngineError(f"could not open the decoded audio: {exc}") from exc

        with audio:
            if audio.getnchannels() != 1 or audio.getsampwidth() != 2:
                raise EngineError(
                    "the offline engine needs 16-bit mono audio; "
                    "the decoder should have produced it"
                )
            sample_rate = audio.getframerate()
            decoder = Decoder(samprate=sample_rate)
            total_frames = audio.getnframes() or 1
            decoder.start_utt()
            read = 0
            while True:
                block = audio.readframes(16_384)
                if not block:
                    break
                decoder.process_raw(block)
                read += len(block) // 2
                request.report(read / total_frames)
            decoder.end_utt()

        words = [
            Word(
                text=_clean(item.word),
                start=item.start_frame / _FRAMES_PER_SECOND,
                end=item.end_frame / _FRAMES_PER_SECOND,
                probability=float(item.prob),
            )
            for item in decoder.seg()
            if not _NON_WORDS.match(item.word)
        ]
        request.report(1.0)

        segments = self._group(words)
        return EngineResult(
            segments=segments,
            language="en",
            language_probability=None if language else 1.0,
            model="en-us (bundled)",
            engine_version=_library_version(),
        )

    def _group(self, words: list[Word]) -> list[Segment]:
        """Collect words into segments, breaking at pauses.

        PocketSphinx decodes a recording as one long utterance, so the pauses
        between words are the only sentence boundaries available.
        """
        groups: list[list[Word]] = []
        for word in words:
            if not word.text:
                continue
            if groups:
                current = groups[-1]
                gap = word.start - current[-1].end
                span = word.end - current[0].start
                if gap <= self.silence_gap and span <= self.max_segment_seconds:
                    current.append(word)
                    continue
            groups.append([word])

        segments = []
        for index, group in enumerate(groups):
            probabilities = [w.probability for w in group if w.probability is not None]
            segments.append(
                Segment(
                    index=index,
                    start=group[0].start,
                    end=group[-1].end,
                    text=" ".join(w.text for w in group),
                    words=tuple(group),
                    confidence=(
                        sum(probabilities) / len(probabilities)
                        if probabilities
                        else None
                    ),
                    language="en",
                )
            )
        return [replace(s, index=i) for i, s in enumerate(segments)]


def _clean(word: str) -> str:
    return _VARIANT.sub("", word).replace("_", " ").strip()


def _library_version() -> str | None:
    try:
        from importlib.metadata import version

        return f"pocketsphinx {version('pocketsphinx')}"
    except Exception:  # pragma: no cover
        return None
