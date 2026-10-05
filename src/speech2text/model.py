"""The transcript artifact model.

The rule that shapes this module: recognized text is immutable. A human edit
never overwrites what the engine produced, it is appended as a correction, so
both the original and the current text are recoverable from the same file for
as long as the artifact exists.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from typing import Any, Iterable, Iterator

SCHEMA_VERSION = "1.0"


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _round(value: float | None, places: int = 3) -> float | None:
    return None if value is None else round(float(value), places)


@dataclass(frozen=True)
class Word:
    """One recognized word with its place in the audio."""

    text: str
    start: float
    end: float
    probability: float | None = None

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "text": self.text,
            "start": _round(self.start),
            "end": _round(self.end),
        }
        if self.probability is not None:
            data["probability"] = _round(self.probability, 4)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Word":
        return cls(
            text=data["text"],
            start=float(data["start"]),
            end=float(data["end"]),
            probability=(
                None if data.get("probability") is None else float(data["probability"])
            ),
        )


@dataclass(frozen=True)
class Segment:
    """A recognized stretch of speech.

    ``text`` is what the engine returned and is never edited in place. Read
    the current text through :meth:`Transcript.text_of` instead.
    """

    index: int
    start: float
    end: float
    text: str
    words: tuple[Word, ...] = ()
    confidence: float | None = None
    language: str | None = None
    speaker: str | None = None
    no_speech_probability: float | None = None

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "index": self.index,
            "start": _round(self.start),
            "end": _round(self.end),
            "text": self.text,
        }
        if self.words:
            data["words"] = [w.to_dict() for w in self.words]
        for key, value in (
            ("confidence", _round(self.confidence, 4)),
            ("language", self.language),
            ("speaker", self.speaker),
            ("no_speech_probability", _round(self.no_speech_probability, 4)),
        ):
            if value is not None:
                data[key] = value
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Segment":
        return cls(
            index=int(data["index"]),
            start=float(data["start"]),
            end=float(data["end"]),
            text=data["text"],
            words=tuple(Word.from_dict(w) for w in data.get("words", ())),
            confidence=(
                None if data.get("confidence") is None else float(data["confidence"])
            ),
            language=data.get("language"),
            speaker=data.get("speaker"),
            no_speech_probability=(
                None
                if data.get("no_speech_probability") is None
                else float(data["no_speech_probability"])
            ),
        )


@dataclass(frozen=True)
class Correction:
    """A human edit to one segment, kept beside the recognition, never over it."""

    segment_index: int
    text: str
    created_at: str = field(default_factory=_now)
    note: str | None = None
    origin: str = "human"

    def to_dict(self) -> dict[str, Any]:
        data = {
            "segment_index": self.segment_index,
            "text": self.text,
            "created_at": self.created_at,
            "origin": self.origin,
        }
        if self.note:
            data["note"] = self.note
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Correction":
        return cls(
            segment_index=int(data["segment_index"]),
            text=data["text"],
            created_at=data.get("created_at", _now()),
            note=data.get("note"),
            origin=data.get("origin", "human"),
        )


@dataclass(frozen=True)
class ReviewIssue:
    """A segment the engine was unsure about, for the review queue."""

    segment_index: int
    reason: str
    confidence: float | None = None
    resolved: bool = False
    accepted: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "segment_index": self.segment_index,
            "reason": self.reason,
            "confidence": _round(self.confidence, 4),
            "resolved": self.resolved,
            "accepted": self.accepted,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ReviewIssue":
        return cls(
            segment_index=int(data["segment_index"]),
            reason=data["reason"],
            confidence=(
                None if data.get("confidence") is None else float(data["confidence"])
            ),
            resolved=bool(data.get("resolved", False)),
            accepted=bool(data.get("accepted", False)),
        )


@dataclass(frozen=True)
class MediaIdentity:
    """What was transcribed, recorded so a bundle can be traced to its source.

    ``container`` and the codec fields come from inspecting the file, not from
    its name, so they stay truthful for a misnamed or extension-less file.
    """

    name: str
    path: str
    size_bytes: int
    duration: float
    container: str
    audio_codec: str | None = None
    video_codec: str | None = None
    sample_rate: int | None = None
    channels: int | None = None
    sha256: str | None = None
    has_video: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "path": self.path,
            "size_bytes": self.size_bytes,
            "duration": _round(self.duration),
            "container": self.container,
            "audio_codec": self.audio_codec,
            "video_codec": self.video_codec,
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "sha256": self.sha256,
            "has_video": self.has_video,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MediaIdentity":
        return cls(
            name=data["name"],
            path=data["path"],
            size_bytes=int(data["size_bytes"]),
            duration=float(data["duration"]),
            container=data["container"],
            audio_codec=data.get("audio_codec"),
            video_codec=data.get("video_codec"),
            sample_rate=data.get("sample_rate"),
            channels=data.get("channels"),
            sha256=data.get("sha256"),
            has_video=bool(data.get("has_video", False)),
        )


@dataclass
class Transcript:
    """The whole artifact: what was transcribed, by what, and every edit since."""

    media: MediaIdentity
    segments: list[Segment] = field(default_factory=list)
    corrections: list[Correction] = field(default_factory=list)
    issues: list[ReviewIssue] = field(default_factory=list)
    engine: str = "unknown"
    model: str | None = None
    engine_version: str | None = None
    languages_requested: tuple[str, ...] = ()
    language_detected: str | None = None
    language_probability: float | None = None
    created_at: str = field(default_factory=_now)
    processing_seconds: float | None = None
    schema: str = SCHEMA_VERSION

    # ---- text -----------------------------------------------------------

    def latest_correction(self, index: int) -> Correction | None:
        """The correction in force for a segment, which is the last one made."""
        found = None
        for correction in self.corrections:
            if correction.segment_index == index:
                found = correction
        return found

    def text_of(self, index: int) -> str:
        """The current text of a segment: its correction if any, else the raw."""
        correction = self.latest_correction(index)
        if correction is not None:
            return correction.text
        return self.segments[index].text

    def raw_text(self) -> str:
        """The recognition exactly as produced, never affected by corrections."""
        return "\n".join(s.text.strip() for s in self.segments).strip() + "\n"

    def corrected_text(self) -> str:
        """The current text, including every correction."""
        lines = [self.text_of(s.index).strip() for s in self.segments]
        return "\n".join(lines).strip() + "\n"

    def plain_text(self) -> str:
        """The current text as flowing prose, for pasting somewhere else."""
        joined = " ".join(self.text_of(s.index).strip() for s in self.segments)
        return " ".join(joined.split())

    # ---- editing --------------------------------------------------------

    def correct(self, index: int, text: str, note: str | None = None,
                origin: str = "human") -> Correction:
        """Record a new text for a segment and resolve any issue against it."""
        if not 0 <= index < len(self.segments):
            raise IndexError(f"no segment {index}")
        correction = Correction(
            segment_index=index, text=text, note=note, origin=origin
        )
        self.corrections.append(correction)
        self.resolve_issue(index)
        return correction

    def resolve_issue(self, index: int, accepted: bool = False) -> None:
        """Mark a segment's issues done. ``accepted`` keeps the text as it is."""
        self.issues = [
            replace(issue, resolved=True, accepted=accepted or issue.accepted)
            if issue.segment_index == index and not issue.resolved
            else issue
            for issue in self.issues
        ]

    def unresolved_issues(self) -> list[ReviewIssue]:
        """The review queue: least confident first, so the worst is seen first."""
        open_issues = [i for i in self.issues if not i.resolved]
        return sorted(
            open_issues,
            key=lambda i: (i.confidence if i.confidence is not None else 1.0),
        )

    # ---- derived --------------------------------------------------------

    @property
    def duration(self) -> float:
        if self.segments:
            return max(self.media.duration, self.segments[-1].end)
        return self.media.duration

    def word_count(self) -> int:
        return len(self.plain_text().split())

    def corrected_segment_indexes(self) -> set[int]:
        return {c.segment_index for c in self.corrections}

    def __iter__(self) -> Iterator[Segment]:
        return iter(self.segments)

    def __len__(self) -> int:
        return len(self.segments)

    # ---- serialisation --------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "created_at": self.created_at,
            "engine": self.engine,
            "engine_version": self.engine_version,
            "model": self.model,
            "languages_requested": list(self.languages_requested),
            "language_detected": self.language_detected,
            "language_probability": _round(self.language_probability, 4),
            "processing_seconds": _round(self.processing_seconds, 2),
            "media": self.media.to_dict(),
            "segments": [s.to_dict() for s in self.segments],
            "corrections": [c.to_dict() for c in self.corrections],
            "issues": [i.to_dict() for i in self.issues],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Transcript":
        schema = data.get("schema", SCHEMA_VERSION)
        if schema.split(".")[0] != SCHEMA_VERSION.split(".")[0]:
            raise ValueError(
                f"transcript schema {schema} is not readable by this version "
                f"(expected {SCHEMA_VERSION})"
            )
        return cls(
            media=MediaIdentity.from_dict(data["media"]),
            segments=[Segment.from_dict(s) for s in data.get("segments", ())],
            corrections=[Correction.from_dict(c) for c in data.get("corrections", ())],
            issues=[ReviewIssue.from_dict(i) for i in data.get("issues", ())],
            engine=data.get("engine", "unknown"),
            model=data.get("model"),
            engine_version=data.get("engine_version"),
            languages_requested=tuple(data.get("languages_requested", ())),
            language_detected=data.get("language_detected"),
            language_probability=data.get("language_probability"),
            created_at=data.get("created_at", _now()),
            processing_seconds=data.get("processing_seconds"),
            schema=schema,
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2) + "\n"

    @classmethod
    def from_json(cls, text: str) -> "Transcript":
        return cls.from_dict(json.loads(text))


def renumber(segments: Iterable[Segment]) -> list[Segment]:
    """Give segments consecutive indexes, which the artifact relies on."""
    return [replace(s, index=i) for i, s in enumerate(segments)]
