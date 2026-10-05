"""Reading whatever file the user dropped in.

Nothing here looks at a file extension. A recording is identified by decoding
its first frames, so ``interview.dat``, ``voice-memo`` with no suffix at all,
and a ``.mp3`` that is really a video all behave the same way: if ffmpeg can
find audio in it, it can be transcribed.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

# Every engine here wants the same thing: 16 kHz mono signed 16-bit PCM.
TARGET_SAMPLE_RATE = 16_000
TARGET_CHANNELS = 1

ProgressFn = Callable[[float], None]


class MediaError(RuntimeError):
    """A file could not be read as media, with a reason worth showing a user."""


class FFmpegMissing(MediaError):
    """ffmpeg or ffprobe is not installed."""


@dataclass(frozen=True)
class MediaInfo:
    """What inspecting a file actually found in it."""

    path: Path
    size_bytes: int
    duration: float
    container: str
    has_audio: bool
    has_video: bool
    audio_codec: str | None = None
    video_codec: str | None = None
    sample_rate: int | None = None
    channels: int | None = None

    @property
    def kind(self) -> str:
        """``video``, ``audio``, or ``data`` — from the streams, not the name."""
        if self.has_video and self.has_audio:
            return "video"
        if self.has_video:
            return "video (silent)"
        if self.has_audio:
            return "audio"
        return "data"

    def describe(self) -> str:
        bits = [self.container or "unknown container"]
        if self.has_video and self.video_codec:
            bits.append(f"video {self.video_codec}")
        if self.audio_codec:
            channels = {1: "mono", 2: "stereo"}.get(self.channels or 0, None)
            audio = f"audio {self.audio_codec}"
            if self.sample_rate:
                audio += f" {self.sample_rate} Hz"
            if channels:
                audio += f" {channels}"
            bits.append(audio)
        return ", ".join(bits)


def _tool(name: str) -> str:
    found = shutil.which(name)
    if not found:
        raise FFmpegMissing(
            f"{name} is not installed. It is what lets Speech2Text open any "
            f"audio or video file. On Ubuntu: sudo apt install ffmpeg"
        )
    return found


def ffmpeg_available() -> bool:
    return bool(shutil.which("ffmpeg") and shutil.which("ffprobe"))


def _run(args: Sequence[str], timeout: float | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        list(args),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=timeout,
    )


def _as_float(value: object) -> float | None:
    try:
        number = float(str(value))
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def probe(path: str | os.PathLike[str]) -> MediaInfo:
    """Identify a file by its content.

    Raises :class:`MediaError` when the file does not exist, is empty, holds no
    media ffmpeg understands, or holds media but no audio to transcribe.
    """
    source = Path(path)
    if not source.exists():
        raise MediaError(f"{source} does not exist")
    if source.is_dir():
        raise MediaError(f"{source} is a folder, not a media file")
    size = source.stat().st_size
    if size == 0:
        raise MediaError(f"{source.name} is empty")

    result = _run([
        _tool("ffprobe"), "-v", "error",
        "-print_format", "json",
        "-show_format", "-show_streams",
        str(source),
    ])
    if result.returncode != 0:
        detail = (result.stderr or "").strip().splitlines()
        reason = detail[-1] if detail else "ffprobe could not read it"
        raise MediaError(f"{source.name} is not audio or video that can be read ({reason})")
    try:
        parsed = json.loads(result.stdout or "{}")
    except json.JSONDecodeError as exc:  # pragma: no cover - ffprobe always emits JSON
        raise MediaError(f"{source.name}: could not understand ffprobe output") from exc

    streams = parsed.get("streams", [])
    container_info = parsed.get("format", {})
    audio = next((s for s in streams if s.get("codec_type") == "audio"), None)
    video = next(
        (
            s
            for s in streams
            # A cover image inside an MP3 is a video stream that is not a video.
            if s.get("codec_type") == "video"
            and s.get("disposition", {}).get("attached_pic", 0) != 1
        ),
        None,
    )

    duration = _as_float(container_info.get("duration"))
    if duration is None and audio is not None:
        duration = _as_float(audio.get("duration"))
    if duration is None:
        duration = _measured_duration(source)

    info = MediaInfo(
        path=source,
        size_bytes=size,
        duration=duration or 0.0,
        container=(container_info.get("format_name") or "unknown").split(",")[0],
        has_audio=audio is not None,
        has_video=video is not None,
        audio_codec=(audio or {}).get("codec_name"),
        video_codec=(video or {}).get("codec_name"),
        sample_rate=int(audio["sample_rate"]) if (audio or {}).get("sample_rate") else None,
        channels=(audio or {}).get("channels"),
    )

    if not info.has_audio:
        what = "a video with no sound track" if info.has_video else "not audio or video"
        raise MediaError(
            f"{source.name} has nothing to transcribe: it is {what}."
        )
    return info


def _measured_duration(source: Path) -> float | None:
    """Decode the audio to find out how long it is, for streams that omit it."""
    result = _run([
        _tool("ffprobe"), "-v", "error",
        "-select_streams", "a:0",
        "-count_packets", "-show_entries", "stream=duration,nb_read_packets",
        "-print_format", "json", str(source),
    ])
    if result.returncode != 0:
        return None
    try:
        streams = json.loads(result.stdout or "{}").get("streams", [])
    except json.JSONDecodeError:
        return None
    return _as_float(streams[0].get("duration")) if streams else None


def decode_to_wav(
    path: str | os.PathLike[str],
    destination: str | os.PathLike[str],
    *,
    sample_rate: int = TARGET_SAMPLE_RATE,
    channels: int = TARGET_CHANNELS,
    duration_hint: float | None = None,
    progress: ProgressFn | None = None,
) -> Path:
    """Decode any readable media to the PCM WAV every engine expects.

    A video's picture is discarded here; only its sound reaches the recognizer.
    ``progress`` is called with a 0..1 fraction while long files decode.
    """
    source = Path(path)
    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)

    args = [
        _tool("ffmpeg"), "-nostdin", "-y",
        "-loglevel", "error",
        "-i", str(source),
        "-vn", "-sn", "-dn",            # drop picture, subtitles, data
        "-map", "0:a:0",                 # the first audio track
        "-ac", str(channels),
        "-ar", str(sample_rate),
        "-c:a", "pcm_s16le",
        "-f", "wav",
    ]
    if progress is not None:
        args += ["-progress", "pipe:1", "-nostats"]
    args.append(str(target))

    if progress is None:
        result = _run(args)
        if result.returncode != 0:
            raise MediaError(_decode_failure(source, result.stderr))
        return _verify_decoded(source, target)

    process = subprocess.Popen(
        args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    total = duration_hint or 0.0
    assert process.stdout is not None
    for line in process.stdout:
        key, _, value = line.strip().partition("=")
        if key == "out_time_us" and total > 0:
            microseconds = _as_float(value)
            if microseconds is not None:
                progress(max(0.0, min(1.0, microseconds / 1e6 / total)))
    stderr = process.stderr.read() if process.stderr else ""
    if process.wait() != 0:
        raise MediaError(_decode_failure(source, stderr))
    progress(1.0)
    return _verify_decoded(source, target)


def _decode_failure(source: Path, stderr: str | None) -> str:
    lines = (stderr or "").strip().splitlines()
    reason = lines[-1] if lines else "ffmpeg failed"
    return f"could not decode the audio of {source.name}: {reason}"


def _verify_decoded(source: Path, target: Path) -> Path:
    # An empty WAV means ffmpeg wrote a header and no samples: silence, or a
    # track it could not decode. Either way there is nothing to recognize.
    if not target.exists() or target.stat().st_size <= 44:
        raise MediaError(f"{source.name} decoded to no audio at all")
    return target


def sha256(path: str | os.PathLike[str], chunk: int = 1 << 20) -> str:
    """Hash a file so a bundle can be matched back to the exact recording."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def format_duration(seconds: float) -> str:
    """``1:03:07`` or ``4:12`` — how long something is, for reading."""
    seconds = max(0, int(round(seconds)))
    hours, rest = divmod(seconds, 3600)
    minutes, secs = divmod(rest, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"
