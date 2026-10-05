"""One recording, from the file someone dropped in to a written bundle.

The stages are always the same: identify the file, decode its audio, recognize
it, write the bundle. Everything specific to a format lives in :mod:`media` and
everything specific to a recognizer lives in :mod:`engines`, so this module
stays the place to read to understand what happens and in what order.
"""

from __future__ import annotations

import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

from . import artifact, media
from .engines import (
    DEFAULT_ENGINE,
    DEFAULT_REVIEW_THRESHOLD,
    SpeechEngine,
    TranscriptionRequest,
    create,
    review_issues,
)
from .model import MediaIdentity, Segment, Transcript

#: What a stage is called while it runs, and how much of the bar it owns.
_STAGES = (
    ("identifying", 0.00, 0.02),
    ("decoding", 0.02, 0.12),
    ("recognizing", 0.12, 0.96),
    ("writing", 0.96, 1.00),
)
_SPAN = {name: (start, end) for name, start, end in _STAGES}


class Cancelled(RuntimeError):
    """A run was stopped on purpose. Not an error to report as a failure."""


@dataclass
class Progress:
    """Where a run has got to, as the window and the command line show it."""

    stage: str = "waiting"
    fraction: float = 0.0
    message: str = ""
    elapsed: float = 0.0

    @property
    def percent(self) -> int:
        return int(round(self.fraction * 100))


ProgressSink = Callable[[Progress], None]


@dataclass
class TranscribeOptions:
    """Everything a run can be told to do differently."""

    engine: str = DEFAULT_ENGINE
    engine_options: dict = field(default_factory=dict)
    languages: Sequence[str] = ()
    review_threshold: float = DEFAULT_REVIEW_THRESHOLD
    write_partial: bool = True
    keep_audio: bool = False
    initial_prompt: str | None = None


class Run:
    """A single transcription, so the window can watch and stop it."""

    def __init__(
        self,
        source: str | Path,
        destination: str | Path,
        options: TranscribeOptions | None = None,
        *,
        engine: SpeechEngine | None = None,
        on_progress: ProgressSink | None = None,
    ) -> None:
        self.source = Path(source)
        self.destination = Path(destination)
        self.options = options or TranscribeOptions()
        self.on_progress = on_progress
        self._engine = engine
        self._stop = False
        self._started = 0.0
        self._partial: list[str] = []
        self.progress = Progress()

    # ---- control --------------------------------------------------------

    def stop(self) -> None:
        """Ask the run to stop. It stops at the next progress report."""
        self._stop = True

    @property
    def stopping(self) -> bool:
        return self._stop

    # ---- progress -------------------------------------------------------

    def _report(self, stage: str, within: float = 1.0, message: str = "") -> None:
        if self._stop:
            raise Cancelled(f"stopped while {stage}")
        start, end = _SPAN.get(stage, (0.0, 1.0))
        fraction = start + (end - start) * max(0.0, min(1.0, within))
        self.progress = Progress(
            stage=stage,
            fraction=fraction,
            message=message or self.progress.message,
            elapsed=time.monotonic() - self._started if self._started else 0.0,
        )
        if self.on_progress is not None:
            self.on_progress(self.progress)

    # ---- the work -------------------------------------------------------

    def execute(self) -> artifact.Bundle:
        self._started = time.monotonic()
        options = self.options

        self._report("identifying", 0.0, f"Reading {self.source.name}")
        info = media.probe(self.source)
        self._report("identifying", 1.0, f"{info.kind}: {info.describe()}")

        engine = self._engine or create(options.engine, **options.engine_options)
        engine.require_available()

        self.destination.mkdir(parents=True, exist_ok=True)
        workspace = tempfile.TemporaryDirectory(prefix="speech2text-")
        try:
            audio_path = (
                self.destination / "audio.wav"
                if options.keep_audio
                else Path(workspace.name) / "audio.wav"
            )
            self._report("decoding", 0.0, "Extracting the audio")
            media.decode_to_wav(
                self.source,
                audio_path,
                duration_hint=info.duration,
                progress=lambda done: self._report("decoding", done),
            )

            self._report("recognizing", 0.0, f"Listening with {engine.title}")
            result = engine.transcribe(
                TranscriptionRequest(
                    audio=audio_path,
                    duration=info.duration,
                    languages=tuple(options.languages),
                    progress=self._recognition_progress(),
                    on_segment=self._collect_partial if options.write_partial else None,
                    initial_prompt=options.initial_prompt,
                )
            )

            self._report("writing", 0.0, "Writing the transcript")
            transcript = Transcript(
                media=self._identity(info),
                segments=result.segments,
                issues=review_issues(result.segments, options.review_threshold),
                engine=engine.name,
                model=result.model,
                engine_version=result.engine_version,
                languages_requested=tuple(options.languages),
                language_detected=result.language or (
                    options.languages[0] if options.languages else None
                ),
                language_probability=result.language_probability,
                processing_seconds=time.monotonic() - self._started,
            )
            bundle = artifact.write_bundle(
                transcript, self.destination, source=self.source
            )
            self._report("writing", 1.0, "Done")
            return bundle
        finally:
            workspace.cleanup()

    def _recognition_progress(self) -> Callable[[float], None]:
        def report(done: float) -> None:
            self._report("recognizing", done)
        return report

    def _collect_partial(self, segment: Segment) -> None:
        """Keep the text readable while a long recording is still running.

        Written to ``transcript.partial.txt`` and replaced by the real
        transcript when the run ends, so a two-hour file can be read from
        before it finishes.
        """
        text = segment.text.strip()
        if not text:
            return
        self._partial.append(text)
        try:
            artifact.write_partial(self.destination, "\n".join(self._partial))
        except OSError:
            # Being unable to write the preview must never fail the run.
            pass

    def _identity(self, info: media.MediaInfo) -> MediaIdentity:
        return MediaIdentity(
            name=self.source.name,
            path=str(self.source.resolve()),
            size_bytes=info.size_bytes,
            duration=info.duration,
            container=info.container,
            audio_codec=info.audio_codec,
            video_codec=info.video_codec,
            sample_rate=info.sample_rate,
            channels=info.channels,
            sha256=media.sha256(self.source),
            has_video=info.has_video,
        )


def transcribe_file(
    source: str | Path,
    destination: str | Path,
    options: TranscribeOptions | None = None,
    *,
    engine: SpeechEngine | None = None,
    on_progress: ProgressSink | None = None,
) -> artifact.Bundle:
    """Transcribe one recording into a bundle. The simple way in."""
    return Run(
        source, destination, options, engine=engine, on_progress=on_progress
    ).execute()


def transcribe_many(
    sources: Sequence[str | Path],
    output_root: str | Path,
    options: TranscribeOptions | None = None,
    *,
    engine: SpeechEngine | None = None,
    on_progress: ProgressSink | None = None,
    on_file: Callable[[int, Path, artifact.Bundle | None, Exception | None], None] | None = None,
) -> list[artifact.Bundle]:
    """Transcribe a batch into one run folder, keeping going past a failure.

    One unreadable file in a folder of recordings must not lose the rest, so
    a failure is reported through ``on_file`` and the batch continues.
    """
    run_root = artifact.run_directory(output_root)
    bundles: list[artifact.Bundle] = []
    for position, source in enumerate(sources, start=1):
        path = Path(source)
        destination = artifact.allocate(run_root, path.name, position)
        try:
            bundle = transcribe_file(
                path, destination, options, engine=engine, on_progress=on_progress
            )
            bundles.append(bundle)
            if on_file is not None:
                on_file(position, path, bundle, None)
        except Cancelled:
            raise
        except Exception as exc:
            if on_file is not None:
                on_file(position, path, None, exc)
            else:
                raise
    return bundles
