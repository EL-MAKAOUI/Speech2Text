"""The contract every recognizer implements.

Engines are adapters. Adding one should not touch the artifact model, the
exports, or the window: it implements :class:`SpeechEngine` and registers a
name. The pipeline only ever sees decoded 16 kHz mono audio, so an engine
never has to know what container the recording arrived in.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

from ..model import ReviewIssue, Segment

ProgressFn = Callable[[float], None]
#: Called with each segment as soon as it is recognized, so a long recording
#: can be read before the run ends. Engines that cannot stream simply omit it.
SegmentFn = Callable[[Segment], None]

# Below this, a segment goes in the review queue. Whisper's own average token
# probability sits near 0.9 on clean speech and falls off sharply on guesses.
DEFAULT_REVIEW_THRESHOLD = 0.6


class EngineError(RuntimeError):
    """Recognition failed for a reason worth showing a user."""


class EngineUnavailable(EngineError):
    """The engine cannot run here: missing package, missing model, no key."""


@dataclass(frozen=True)
class TranscriptionRequest:
    """Everything an engine needs for one recording."""

    audio: Path
    duration: float
    languages: Sequence[str] = ()       # empty means: detect it
    progress: ProgressFn | None = None
    on_segment: SegmentFn | None = None
    initial_prompt: str | None = None

    @property
    def language(self) -> str | None:
        """The one language to recognize in, or ``None`` to detect."""
        return self.languages[0] if self.languages else None

    def report(self, fraction: float) -> None:
        if self.progress is not None:
            self.progress(max(0.0, min(1.0, fraction)))

    def emit(self, segment: Segment) -> None:
        """Hand over a finished segment, where the engine can produce them
        one at a time. Never required: the result is what counts."""
        if self.on_segment is not None:
            self.on_segment(segment)


@dataclass
class EngineResult:
    """What an engine produced, before it becomes an artifact."""

    segments: list[Segment] = field(default_factory=list)
    language: str | None = None
    language_probability: float | None = None
    model: str | None = None
    engine_version: str | None = None


class SpeechEngine(ABC):
    """A recognizer. Implementations live beside this file."""

    name: str = "engine"
    #: Shown in the window's Recognition list.
    title: str = "Engine"
    #: One line on when to reach for it.
    summary: str = ""
    #: True when audio leaves the machine, which the UI must say plainly.
    uploads_audio: bool = False

    @abstractmethod
    def transcribe(self, request: TranscriptionRequest) -> EngineResult:
        """Recognize the speech in ``request.audio``."""

    def availability(self) -> tuple[bool, str]:
        """``(usable, why not)``. Checked before a run, and by the window."""
        return True, ""

    @property
    def available(self) -> bool:
        return self.availability()[0]

    def require_available(self) -> None:
        usable, reason = self.availability()
        if not usable:
            raise EngineUnavailable(reason)


def confidence_from_logprob(average_logprob: float | None) -> float | None:
    """Turn an average log probability into a 0..1 confidence.

    Whisper reports the mean log probability of a segment's tokens. Its
    exponential is the mean per-token probability, which is a number a person
    can reason about and compare against a threshold.
    """
    if average_logprob is None:
        return None
    try:
        return max(0.0, min(1.0, math.exp(float(average_logprob))))
    except (OverflowError, ValueError):  # pragma: no cover - guards bad input
        return None


def review_issues(
    segments: Sequence[Segment],
    threshold: float = DEFAULT_REVIEW_THRESHOLD,
) -> list[ReviewIssue]:
    """Flag the segments worth a second listen, least confident first.

    This is what turns "check a two-hour recording" into a finite list.
    """
    issues: list[ReviewIssue] = []
    for segment in segments:
        if segment.confidence is not None and segment.confidence < threshold:
            issues.append(
                ReviewIssue(
                    segment_index=segment.index,
                    reason="low confidence",
                    confidence=segment.confidence,
                )
            )
        elif not segment.text.strip():
            issues.append(
                ReviewIssue(
                    segment_index=segment.index,
                    reason="nothing recognized",
                    confidence=segment.confidence,
                )
            )
    return sorted(
        issues, key=lambda i: (i.confidence if i.confidence is not None else 1.0)
    )
