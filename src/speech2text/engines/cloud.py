"""The optional cloud recognizer.

This is never a default. It needs an explicit choice and a key, and while it
is selected the window says plainly that the audio leaves the machine. Any
failure falls back to the local engine, so a recording is never lost to a
network problem.

Everything here speaks plain HTTP, so no vendor SDK is a dependency.
"""

from __future__ import annotations

import io
import json
import mimetypes
import random
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
import wave
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Sequence

from ..keys import ApiKey, first_key
from ..model import Segment
from .base import (
    EngineError,
    EngineResult,
    EngineUnavailable,
    SpeechEngine,
    TranscriptionRequest,
)

#: Providers cap an upload at around 25 MB. 16 kHz mono PCM is 32 KB a second,
#: so eight minutes is about 15 MB: clear of the cap with room for the headers.
CHUNK_SECONDS = 8 * 60
MAX_UPLOAD_BYTES = 24 * 1024 * 1024

_RETRY_STATUS = frozenset({408, 409, 429, 500, 502, 503, 504})
_MAX_ATTEMPTS = 4


@dataclass
class CloudChunk:
    """One provider response for one slice of audio."""

    text: str
    language: str | None = None
    #: ``(start, end, text)`` in seconds from the start of the chunk. Empty
    #: when the provider returns text without timings.
    spans: list[tuple[float, float, str]] = field(default_factory=list)


class CloudProvider(ABC):
    """One hosted service that turns audio into text."""

    name: str = "cloud"
    #: Shown so a person can tell what their audio is being sent to.
    endpoint_host: str = ""

    @abstractmethod
    def transcribe(self, audio: bytes, language: str | None, key: ApiKey) -> CloudChunk:
        ...

    def describe(self) -> str:
        return f"{self.name} ({self.endpoint_host})"


# ---------------------------------------------------------------------------
# HTTP plumbing
# ---------------------------------------------------------------------------


class CloudHttpError(EngineError):
    """A request failed. The key is never included in the message."""

    def __init__(self, message: str, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


def _request(
    url: str,
    *,
    data: bytes | None = None,
    headers: dict[str, str] | None = None,
    method: str = "POST",
    timeout: float = 300.0,
    sleep=time.sleep,
) -> bytes:
    """POST with retries on the failures that are worth retrying.

    429 and 5xx are transient and are retried with backoff. Any other 4xx is
    a bad request or a bad key: retrying it only wastes quota.
    """
    last: Exception | None = None
    for attempt in range(_MAX_ATTEMPTS):
        request = urllib.request.Request(
            url, data=data, headers=headers or {}, method=method
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read()
        except urllib.error.HTTPError as exc:
            body = _safe_body(exc)
            if exc.code not in _RETRY_STATUS or attempt == _MAX_ATTEMPTS - 1:
                raise CloudHttpError(
                    f"the service answered {exc.code} {exc.reason}{body}", exc.code
                ) from exc
            last = exc
        except urllib.error.URLError as exc:
            if attempt == _MAX_ATTEMPTS - 1:
                raise CloudHttpError(f"could not reach the service: {exc.reason}") from exc
            last = exc
        # Full jitter: several keys retrying together must not sync up.
        sleep(min(30.0, (2 ** attempt)) * (0.5 + random.random() / 2))
    raise CloudHttpError(f"gave up after {_MAX_ATTEMPTS} attempts: {last}")


def _safe_body(exc: urllib.error.HTTPError) -> str:
    """The service's own explanation, trimmed, never echoing the request."""
    try:
        raw = exc.read().decode("utf-8", "replace").strip()
    except Exception:  # pragma: no cover - body already consumed
        return ""
    if not raw:
        return ""
    try:
        parsed = json.loads(raw)
        message = parsed.get("error", parsed)
        if isinstance(message, dict):
            message = message.get("message") or json.dumps(message)
    except json.JSONDecodeError:
        message = raw
    return f" — {str(message)[:300]}"


def _multipart(fields: dict[str, str], filename: str, content: bytes) -> tuple[bytes, str]:
    """Build a multipart/form-data body without pulling in a dependency."""
    boundary = f"----speech2text{uuid.uuid4().hex}"
    buffer = io.BytesIO()
    for name, value in fields.items():
        buffer.write(f"--{boundary}\r\n".encode())
        buffer.write(f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode())
        buffer.write(f"{value}\r\n".encode())
    mime = mimetypes.guess_type(filename)[0] or "application/octet-stream"
    buffer.write(f"--{boundary}\r\n".encode())
    buffer.write(
        f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'.encode()
    )
    buffer.write(f"Content-Type: {mime}\r\n\r\n".encode())
    buffer.write(content)
    buffer.write(f"\r\n--{boundary}--\r\n".encode())
    return buffer.getvalue(), f"multipart/form-data; boundary={boundary}"


# ---------------------------------------------------------------------------
# Providers
# ---------------------------------------------------------------------------


class OpenAICompatibleProvider(CloudProvider):
    """Any service exposing ``/v1/audio/transcriptions``.

    That covers OpenAI and Groq, which both run Whisper server-side and return
    real segment timings when asked for ``verbose_json``.
    """

    def __init__(self, name: str, base_url: str, model: str) -> None:
        self.name = name
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.endpoint_host = urllib.parse.urlparse(self.base_url).netloc

    def transcribe(self, audio: bytes, language: str | None, key: ApiKey) -> CloudChunk:
        fields = {"model": self.model, "response_format": "verbose_json"}
        if language:
            fields["language"] = language
        body, content_type = _multipart(fields, "audio.wav", audio)
        raw = _request(
            f"{self.base_url}/audio/transcriptions",
            data=body,
            headers={
                "Authorization": f"Bearer {key.value}",
                "Content-Type": content_type,
            },
        )
        return self._parse(raw)

    @staticmethod
    def _parse(raw: bytes) -> CloudChunk:
        try:
            payload = json.loads(raw.decode("utf-8", "replace"))
        except json.JSONDecodeError as exc:
            raise CloudHttpError("the service returned something that is not JSON") from exc
        spans = [
            (float(s["start"]), float(s["end"]), str(s.get("text", "")).strip())
            for s in payload.get("segments", [])
            if s.get("start") is not None and s.get("end") is not None
        ]
        return CloudChunk(
            text=str(payload.get("text", "")).strip(),
            language=payload.get("language"),
            spans=spans,
        )


class GeminiProvider(CloudProvider):
    """Google's Gemini, which reads audio inline in a generateContent call.

    It returns text without timings, so spans are estimated. See the guide.
    """

    name = "gemini"
    endpoint_host = "generativelanguage.googleapis.com"

    def __init__(self, model: str = "gemini-2.5-flash") -> None:
        self.model = model

    def transcribe(self, audio: bytes, language: str | None, key: ApiKey) -> CloudChunk:
        import base64

        instruction = (
            "Transcribe this audio exactly as spoken. Return only the "
            "transcript, with no commentary, headings or timestamps."
        )
        if language:
            instruction += f" The speech is in {language}."
        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": instruction},
                        {
                            "inline_data": {
                                "mime_type": "audio/wav",
                                "data": base64.b64encode(audio).decode("ascii"),
                            }
                        },
                    ]
                }
            ],
            "generationConfig": {"temperature": 0.0},
        }
        raw = _request(
            f"https://{self.endpoint_host}/v1beta/models/{self.model}:generateContent",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "x-goog-api-key": key.value,
            },
        )
        try:
            parsed = json.loads(raw.decode("utf-8", "replace"))
            parts = parsed["candidates"][0]["content"]["parts"]
            text = "".join(p.get("text", "") for p in parts).strip()
        except (json.JSONDecodeError, KeyError, IndexError) as exc:
            raise CloudHttpError("the model returned no transcript") from exc
        return CloudChunk(text=text, language=language)


#: What ``--cloud-provider`` accepts.
PROVIDERS: dict[str, "callable"] = {
    "groq": lambda model=None: OpenAICompatibleProvider(
        "groq", "https://api.groq.com/openai/v1", model or "whisper-large-v3"
    ),
    "openai": lambda model=None: OpenAICompatibleProvider(
        "openai", "https://api.openai.com/v1", model or "whisper-1"
    ),
    "gemini": lambda model=None: GeminiProvider(model or "gemini-2.5-flash"),
}


def build_provider(name: str, model: str | None = None) -> CloudProvider:
    try:
        return PROVIDERS[name.casefold()](model)
    except KeyError:
        known = ", ".join(sorted(PROVIDERS))
        raise EngineUnavailable(f"unknown cloud provider {name!r}. Known: {known}")


# ---------------------------------------------------------------------------
# The engine
# ---------------------------------------------------------------------------


def split_wav(path: Path, seconds: float = CHUNK_SECONDS) -> list[tuple[float, bytes]]:
    """Cut decoded audio into uploadable pieces, each a complete WAV file.

    Returns ``(offset_seconds, wav_bytes)``, so a chunk's timings can be moved
    back to where it belongs in the recording.
    """
    with wave.open(str(path), "rb") as source:
        rate = source.getframerate()
        channels = source.getnchannels()
        width = source.getsampwidth()
        per_chunk = max(1, int(rate * seconds))
        chunks: list[tuple[float, bytes]] = []
        index = 0
        while True:
            frames = source.readframes(per_chunk)
            if not frames:
                break
            buffer = io.BytesIO()
            with wave.open(buffer, "wb") as sink:
                sink.setnchannels(channels)
                sink.setsampwidth(width)
                sink.setframerate(rate)
                sink.writeframes(frames)
            chunks.append((index * per_chunk / rate, buffer.getvalue()))
            index += 1
    return chunks


def _estimate_spans(text: str, duration: float) -> list[tuple[float, float, str]]:
    """Spread text over a chunk's duration when the provider gives no timings.

    These are estimates. They are good enough to follow along and to cut
    subtitles, and they are not real word timings — the guide says so.
    """
    sentences = [part.strip() for part in _split_sentences(text) if part.strip()]
    if not sentences:
        return []
    weights = [max(1, len(s)) for s in sentences]
    total = sum(weights)
    spans: list[tuple[float, float, str]] = []
    elapsed = 0.0
    for sentence, weight in zip(sentences, weights):
        span = duration * weight / total
        spans.append((elapsed, elapsed + span, sentence))
        elapsed += span
    return spans


def _split_sentences(text: str) -> list[str]:
    import re

    # Keep the terminator with its sentence; handles Arabic ؟ and ۔ too.
    parts = re.split(r"(?<=[.!?؟۔…])\s+", text.replace("\n", " "))
    return parts if parts else [text]


class CloudEngine(SpeechEngine):
    name = "cloud"
    title = "Cloud"
    summary = "Most accurate. Your audio is uploaded to a third party."
    uploads_audio = True

    def __init__(
        self,
        provider: CloudProvider | str = "groq",
        *,
        model: str | None = None,
        fallback: SpeechEngine | None = None,
        chunk_seconds: float = CHUNK_SECONDS,
    ) -> None:
        self.provider = (
            build_provider(provider, model) if isinstance(provider, str) else provider
        )
        self.fallback = fallback
        self.chunk_seconds = chunk_seconds
        #: Set when a run fell back, so the artifact records what really ran.
        self.fell_back_because: str | None = None

    def availability(self) -> tuple[bool, str]:
        if first_key(self.provider.name) is None:
            return False, (
                f"no API key for {self.provider.name}. Add one with:\n"
                f"    speech2text keys add --provider {self.provider.name}\n"
                f"or set the matching environment variable."
            )
        return True, ""

    def transcribe(self, request: TranscriptionRequest) -> EngineResult:
        self.fell_back_because = None
        key = first_key(self.provider.name)
        if key is None:
            # Never upload without a key, and never silently do nothing.
            return self._fall_back(request, self.availability()[1])
        try:
            return self._transcribe_cloud(request, key)
        except EngineError as exc:
            return self._fall_back(request, str(exc))

    def _transcribe_cloud(self, request: TranscriptionRequest, key: ApiKey) -> EngineResult:
        chunks = split_wav(request.audio, self.chunk_seconds)
        if not chunks:
            raise EngineError("there was no audio to send")
        for offset, blob in chunks:
            if len(blob) > MAX_UPLOAD_BYTES:  # pragma: no cover - guarded by chunking
                raise EngineError(
                    f"a {len(blob) // (1024 * 1024)} MB chunk is larger than the "
                    f"upload limit; lower the chunk length"
                )

        segments: list[Segment] = []
        language: str | None = None
        for position, (offset, blob) in enumerate(chunks):
            result = self.provider.transcribe(blob, request.language, key)
            language = language or result.language
            chunk_duration = _wav_duration(blob)
            spans = result.spans or _estimate_spans(result.text, chunk_duration)
            estimated = not result.spans
            for start, end, text in spans:
                if not text:
                    continue
                segments.append(
                    Segment(
                        index=len(segments),
                        start=offset + start,
                        end=offset + end,
                        text=text,
                        language=language,
                        # No provider here reports per-segment confidence, so
                        # the review queue stays empty rather than inventing one.
                        confidence=None,
                        speaker=None,
                    )
                )
            request.report((position + 1) / len(chunks))
            self._estimated_timings = estimated

        if not segments:
            raise EngineError("the service returned no transcript")
        return EngineResult(
            segments=[replace(s, index=i) for i, s in enumerate(segments)],
            language=language or request.language,
            language_probability=None,
            model=getattr(self.provider, "model", None),
            engine_version=f"cloud:{self.provider.name}",
        )

    def _fall_back(self, request: TranscriptionRequest, reason: str) -> EngineResult:
        if self.fallback is None:
            raise EngineError(
                f"the cloud recognizer could not run and there is no local "
                f"engine to fall back to: {reason}"
            )
        self.fell_back_because = reason
        result = self.fallback.transcribe(request)
        result.engine_version = (
            f"{result.engine_version or self.fallback.name} "
            f"(fell back from cloud: {reason.splitlines()[0]})"
        )
        return result


def _wav_duration(blob: bytes) -> float:
    with wave.open(io.BytesIO(blob), "rb") as handle:
        return handle.getnframes() / float(handle.getframerate() or 1)
