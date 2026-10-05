"""The output bundle: what is written to disk for every transcription.

A bundle is a folder, and the folder is the integration boundary. Another
application reads ``consumer.json`` and needs nothing from this package — no
engine, no model, no import.

| file | |
|---|---|
| ``transcript.txt`` | the text, including corrections |
| ``transcript.raw.txt`` | the recognition as produced, never modified |
| ``transcript.json`` | everything: timings, words, confidence, corrections |
| ``consumer.json`` | a small, stable contract for other applications |
| ``project.json`` | the job: where review got to, when it ran |

``transcript.json`` is the one to keep. The rest is regenerated from it.
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .model import Transcript

TRANSCRIPT_JSON = "transcript.json"
TRANSCRIPT_TXT = "transcript.txt"
TRANSCRIPT_RAW_TXT = "transcript.raw.txt"
TRANSCRIPT_PARTIAL_TXT = "transcript.partial.txt"
CONSUMER_JSON = "consumer.json"
PROJECT_JSON = "project.json"

CONSUMER_SCHEMA = "1.0"

_PARTIAL_HEADER = (
    "# Speech2Text — in progress\n"
    "# This file is rewritten as the recording is recognized and is replaced\n"
    "# by transcript.txt when it finishes.\n\n"
)


def slug(name: str, limit: int = 48) -> str:
    """A filesystem-safe folder name that still looks like the recording."""
    stem = Path(name).stem
    normalized = unicodedata.normalize("NFKD", stem)
    ascii_only = normalized.encode("ascii", "ignore").decode("ascii")
    cleaned = re.sub(r"[^A-Za-z0-9]+", "-", ascii_only).strip("-").lower()
    if not cleaned:
        # A name with no Latin characters at all still needs a folder.
        cleaned = "recording"
    return cleaned[:limit].strip("-") or "recording"


def run_directory(root: str | Path, when: datetime | None = None) -> Path:
    """One folder per run, named for when it started."""
    moment = when or datetime.now()
    return Path(root) / moment.strftime("%Y%m%d-%H%M%S-%f")


def allocate(run_root: str | Path, name: str, position: int = 1) -> Path:
    """``001-interview`` inside the run folder, so a batch keeps its order."""
    directory = Path(run_root) / f"{position:03d}-{slug(name)}"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


# ---------------------------------------------------------------------------
# consumer contract
# ---------------------------------------------------------------------------


def consumer_payload(transcript: Transcript) -> dict[str, Any]:
    """The small, stable view other applications read.

    Deliberately flat and deliberately small: text, where it came from, and
    how much of it still wants checking. Adding to it is allowed; changing
    the meaning of a field is a schema change.
    """
    segments = [
        {
            "start": round(segment.start, 3),
            "end": round(segment.end, 3),
            "text": transcript.text_of(segment.index),
        }
        for segment in transcript.segments
        if transcript.text_of(segment.index).strip()
    ]
    return {
        "schema": CONSUMER_SCHEMA,
        "kind": "speech-transcript",
        "produced_by": "speech2text",
        "source": {
            "name": transcript.media.name,
            "sha256": transcript.media.sha256,
            "duration": round(transcript.media.duration, 3),
            "has_video": transcript.media.has_video,
        },
        "language": transcript.language_detected,
        "engine": transcript.engine,
        "model": transcript.model,
        "text": transcript.corrected_text().strip(),
        "word_count": transcript.word_count(),
        "segments": segments,
        "unresolved_reviews": len(transcript.unresolved_issues()),
        "corrections": len(transcript.corrections),
        "created_at": transcript.created_at,
    }


# ---------------------------------------------------------------------------
# project bookkeeping
# ---------------------------------------------------------------------------


def default_project(transcript: Transcript, source: Path | None = None) -> dict[str, Any]:
    return {
        "schema": "1.0",
        "source_path": str(source) if source else transcript.media.path,
        "source_name": transcript.media.name,
        "created_at": transcript.created_at,
        "updated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "engine": transcript.engine,
        "model": transcript.model,
        "language": transcript.language_detected,
        "duration": round(transcript.media.duration, 3),
        "review_position": 0,
        "completed": True,
    }


def load_project(directory: str | Path) -> dict[str, Any]:
    path = Path(directory) / PROJECT_JSON
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def save_project(directory: str | Path, data: dict[str, Any]) -> Path:
    path = Path(directory) / PROJECT_JSON
    data = dict(data)
    data["updated_at"] = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def remember_review_position(directory: str | Path, index: int) -> None:
    """Save where reviewing got to, so a long recording resumes there."""
    project = load_project(directory) or {"schema": "1.0"}
    project["review_position"] = int(index)
    save_project(directory, project)


# ---------------------------------------------------------------------------
# writing and reading
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Bundle:
    """A written bundle and the paths inside it."""

    directory: Path
    transcript: Transcript

    @property
    def json_path(self) -> Path:
        return self.directory / TRANSCRIPT_JSON

    @property
    def text_path(self) -> Path:
        return self.directory / TRANSCRIPT_TXT

    @property
    def raw_text_path(self) -> Path:
        return self.directory / TRANSCRIPT_RAW_TXT

    @property
    def consumer_path(self) -> Path:
        return self.directory / CONSUMER_JSON


def write_bundle(
    transcript: Transcript,
    directory: str | Path,
    *,
    source: Path | None = None,
    project: dict[str, Any] | None = None,
) -> Bundle:
    """Write every file of a bundle, creating the folder if needed."""
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    _write(target / TRANSCRIPT_JSON, transcript.to_json())
    _write(target / TRANSCRIPT_RAW_TXT, transcript.raw_text())
    _write(target / TRANSCRIPT_TXT, transcript.corrected_text())
    _write(
        target / CONSUMER_JSON,
        json.dumps(consumer_payload(transcript), ensure_ascii=False, indent=2) + "\n",
    )
    save_project(target, project or default_project(transcript, source))
    # The partial file only exists while a run is in flight.
    partial = target / TRANSCRIPT_PARTIAL_TXT
    if partial.exists():
        partial.unlink()
    return Bundle(directory=target, transcript=transcript)


def write_partial(directory: str | Path, text: str) -> Path:
    """Write what has been recognized so far, so a long run can be read early."""
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    path = target / TRANSCRIPT_PARTIAL_TXT
    _write(path, _PARTIAL_HEADER + text.rstrip() + "\n")
    return path


def refresh(directory: str | Path) -> Bundle:
    """Rewrite the derived files after a correction, leaving the raw alone."""
    target = Path(directory)
    transcript = load(target)
    _write(target / TRANSCRIPT_JSON, transcript.to_json())
    _write(target / TRANSCRIPT_TXT, transcript.corrected_text())
    _write(
        target / CONSUMER_JSON,
        json.dumps(consumer_payload(transcript), ensure_ascii=False, indent=2) + "\n",
    )
    return Bundle(directory=target, transcript=transcript)


def load(path: str | Path) -> Transcript:
    """Read a transcript from a bundle folder or straight from its JSON file."""
    candidate = Path(path)
    if candidate.is_dir():
        candidate = candidate / TRANSCRIPT_JSON
    if not candidate.exists():
        raise FileNotFoundError(f"no transcript at {candidate}")
    return Transcript.from_json(candidate.read_text(encoding="utf-8"))


def _write(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


# ---------------------------------------------------------------------------
# finding earlier work
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PreviousRun:
    """A bundle found on disk, described well enough to choose between them."""

    directory: Path
    name: str
    created_at: str
    duration: float
    language: str | None
    engine: str
    unresolved_reviews: int
    word_count: int

    @property
    def label(self) -> str:
        from .media import format_duration

        parts = [self.name, format_duration(self.duration)]
        if self.unresolved_reviews:
            parts.append(f"{self.unresolved_reviews} to check")
        return " · ".join(parts)


def find_previous(root: str | Path, limit: int = 200) -> list[PreviousRun]:
    """Every bundle under a folder, most recent first.

    Reads ``consumer.json`` where it exists because it is small, and falls
    back to the full transcript so bundles written by anything else still
    appear.
    """
    base = Path(root)
    if not base.exists():
        return []
    found: list[PreviousRun] = []
    for json_path in sorted(base.rglob(TRANSCRIPT_JSON), reverse=True):
        directory = json_path.parent
        described = _describe(directory)
        if described is not None:
            found.append(described)
        if len(found) >= limit:
            break
    return sorted(found, key=lambda run: run.created_at, reverse=True)


def _describe(directory: Path) -> PreviousRun | None:
    consumer_path = directory / CONSUMER_JSON
    data: dict[str, Any] | None = None
    if consumer_path.exists():
        try:
            data = json.loads(consumer_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            data = None
    if data is not None and data.get("kind") == "speech-transcript":
        source = data.get("source", {})
        return PreviousRun(
            directory=directory,
            name=source.get("name", directory.name),
            created_at=data.get("created_at", ""),
            duration=float(source.get("duration", 0.0)),
            language=data.get("language"),
            engine=data.get("engine", "unknown"),
            unresolved_reviews=int(data.get("unresolved_reviews", 0)),
            word_count=int(data.get("word_count", 0)),
        )
    try:
        transcript = load(directory)
    except (OSError, ValueError, FileNotFoundError):
        return None
    return PreviousRun(
        directory=directory,
        name=transcript.media.name,
        created_at=transcript.created_at,
        duration=transcript.media.duration,
        language=transcript.language_detected,
        engine=transcript.engine,
        unresolved_reviews=len(transcript.unresolved_issues()),
        word_count=transcript.word_count(),
    )


def iter_bundles(root: str | Path) -> Iterable[Path]:
    for path in Path(root).rglob(TRANSCRIPT_JSON):
        yield path.parent
