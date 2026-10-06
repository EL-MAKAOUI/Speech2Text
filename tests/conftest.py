"""Fixtures that build real media files with ffmpeg.

The point of this application is reading whatever file it is handed, so the
tests use genuine encoded files rather than stubs: a video, an audio file
with the wrong extension, a file with no extension, an MP3 carrying cover
art, a video with no sound, and things that are not media at all.
"""

from __future__ import annotations

import importlib.util
import shutil
import subprocess
import sys
import wave
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

HAS_FFMPEG = bool(shutil.which("ffmpeg") and shutil.which("ffprobe"))
HAS_ESPEAK = bool(shutil.which("espeak-ng") or shutil.which("espeak"))
HAS_SPHINX = importlib.util.find_spec("pocketsphinx") is not None
HAS_WHISPER = importlib.util.find_spec("faster_whisper") is not None

requires_ffmpeg = pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg is not installed")
requires_espeak = pytest.mark.skipif(not HAS_ESPEAK, reason="espeak-ng is not installed")
# The optional recognizers are extras, so a minimal install must skip, not fail.
requires_sphinx = pytest.mark.skipif(
    not HAS_SPHINX, reason="pocketsphinx is not installed"
)
requires_whisper = pytest.mark.skipif(
    not HAS_WHISPER, reason="faster-whisper is not installed"
)

SPOKEN_SENTENCE = (
    "The quick brown fox jumps over the lazy dog. "
    "This recording is a test of the transcription system."
)


def _run(args: list[str]) -> None:
    result = subprocess.run(args, capture_output=True, text=True)
    if result.returncode != 0:  # pragma: no cover - a broken test environment
        raise RuntimeError(f"{args[0]} failed: {result.stderr[-400:]}")


@pytest.fixture(scope="session")
def media_root(tmp_path_factory) -> Path:
    return tmp_path_factory.mktemp("media")


@pytest.fixture(scope="session")
def spoken_wav(media_root: Path) -> Path:
    """Real speech, synthesized, at the rate the engines want."""
    if not HAS_ESPEAK:
        pytest.skip("espeak-ng is not installed")
    raw = media_root / "spoken-raw.wav"
    speaker = shutil.which("espeak-ng") or shutil.which("espeak")
    _run([speaker, "-v", "en-us", "-s", "140", "-w", str(raw), SPOKEN_SENTENCE])
    target = media_root / "spoken.wav"
    _run([
        "ffmpeg", "-y", "-loglevel", "error", "-i", str(raw),
        "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", str(target),
    ])
    return target


@pytest.fixture(scope="session")
def tone_wav(media_root: Path) -> Path:
    """Audio with no speech in it, for tests that only need a readable file."""
    if not HAS_FFMPEG:
        pytest.skip("ffmpeg is not installed")
    target = media_root / "tone.wav"
    _run([
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "lavfi", "-i", "sine=frequency=440:duration=3",
        "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", str(target),
    ])
    return target


@pytest.fixture(scope="session")
def video_with_audio(media_root: Path, tone_wav: Path) -> Path:
    target = media_root / "clip.mp4"
    _run([
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "lavfi", "-i", "testsrc=size=320x240:rate=10",
        "-i", str(tone_wav), "-shortest",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", str(target),
    ])
    return target


@pytest.fixture(scope="session")
def misnamed_video(media_root: Path, video_with_audio: Path) -> Path:
    """A video called ``.dat``: the case the application exists to handle."""
    target = media_root / "recording.dat"
    shutil.copy(video_with_audio, target)
    return target


@pytest.fixture(scope="session")
def extensionless_audio(media_root: Path, tone_wav: Path) -> Path:
    target = media_root / "voice-memo"
    shutil.copy(tone_wav, target)
    return target


@pytest.fixture(scope="session")
def wav_called_mp3(media_root: Path, tone_wav: Path) -> Path:
    target = media_root / "actually-a-wav.mp3"
    shutil.copy(tone_wav, target)
    return target


@pytest.fixture(scope="session")
def mp3_with_cover_art(media_root: Path, tone_wav: Path) -> Path:
    """An MP3 whose cover image is a video stream, but is not a video."""
    target = media_root / "with-cover.mp3"
    _run([
        "ffmpeg", "-y", "-loglevel", "error", "-i", str(tone_wav),
        "-f", "lavfi", "-i", "color=c=red:s=64x64:d=1",
        "-map", "0:a", "-map", "1:v", "-c:v", "mjpeg",
        "-disposition:v", "attached_pic", "-c:a", "libmp3lame", str(target),
    ])
    return target


@pytest.fixture(scope="session")
def silent_video(media_root: Path) -> Path:
    target = media_root / "silent.mp4"
    _run([
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "lavfi", "-i", "testsrc=size=160x120:rate=5", "-t", "2",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", str(target),
    ])
    return target


@pytest.fixture(scope="session")
def stereo_audio(media_root: Path) -> Path:
    target = media_root / "stereo.flac"
    _run([
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "lavfi", "-i", "sine=frequency=330:duration=2",
        "-ac", "2", "-ar", "44100", str(target),
    ])
    return target


@pytest.fixture
def not_media(tmp_path: Path) -> Path:
    target = tmp_path / "notes.txt"
    target.write_text("this is plainly not a recording", encoding="utf-8")
    return target


@pytest.fixture
def empty_file(tmp_path: Path) -> Path:
    target = tmp_path / "empty.wav"
    target.touch()
    return target


# ---------------------------------------------------------------------------
# model helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_transcript():
    """A small transcript with a correction and an open review issue."""
    from speech2text.model import MediaIdentity, ReviewIssue, Segment, Transcript

    transcript = Transcript(
        media=MediaIdentity(
            name="interview.mp4",
            path="/recordings/interview.mp4",
            size_bytes=2048,
            duration=12.5,
            container="mov",
            audio_codec="aac",
            video_codec="h264",
            sample_rate=48000,
            channels=2,
            sha256="a" * 64,
            has_video=True,
        ),
        segments=[
            Segment(0, 0.0, 3.25, "Hello and welcome.", confidence=0.95),
            Segment(1, 3.25, 7.5, "Today we discus testing.", confidence=0.42),
            Segment(2, 7.5, 12.5, "Thanks for listening.", confidence=0.88),
        ],
        issues=[ReviewIssue(1, "low confidence", 0.42)],
        engine="whisper",
        model="base",
        language_detected="en",
        language_probability=0.99,
    )
    return transcript


@pytest.fixture
def fake_engine():
    """An engine with no dependencies, so the pipeline can be tested alone."""
    from speech2text.engines.base import EngineResult, SpeechEngine
    from speech2text.model import Segment, Word

    class FakeEngine(SpeechEngine):
        name = "fake"
        title = "Fake"
        summary = "for tests"

        def __init__(self) -> None:
            self.requests = []

        def transcribe(self, request):
            self.requests.append(request)
            segments = [
                Segment(
                    0, 0.0, 1.5, "first segment",
                    words=(Word("first", 0.0, 0.7, 0.9), Word("segment", 0.7, 1.5, 0.8)),
                    confidence=0.9,
                ),
                Segment(1, 1.5, 3.0, "second segment", confidence=0.3),
            ]
            # A real streaming engine hands each segment over as it finds it.
            request.emit(segments[0])
            request.report(0.5)
            request.emit(segments[1])
            request.report(1.0)
            return EngineResult(
                segments=segments,
                language=request.language or "en",
                language_probability=0.97,
                model="fake-1",
                engine_version="fake 1.0",
            )

    return FakeEngine()


def read_wav_duration(path: Path) -> float:
    with wave.open(str(path), "rb") as handle:
        return handle.getnframes() / handle.getframerate()
