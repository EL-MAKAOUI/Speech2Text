"""The optional cloud recognizer: never without a key, never losing a recording."""

from __future__ import annotations

import io
import json
import urllib.error
import wave
from pathlib import Path

import pytest

from speech2text.engines import cloud
from speech2text.engines.base import EngineError, EngineResult, SpeechEngine, TranscriptionRequest
from speech2text.keys import ApiKey
from speech2text.model import Segment

KEY = ApiKey("groq", "gsk_secret_value_do_not_leak", "test")


@pytest.fixture
def wav_file(tmp_path) -> Path:
    """Twenty seconds of silence, enough to be split into chunks."""
    path = tmp_path / "audio.wav"
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(b"\x00\x00" * 16000 * 20)
    return path


class RecordingProvider(cloud.CloudProvider):
    """A provider that answers without a network, so behaviour can be checked."""

    name = "groq"

    def __init__(self, chunk: cloud.CloudChunk | None = None, fail: Exception | None = None):
        self.calls: list[tuple[int, str | None]] = []
        self._chunk = chunk or cloud.CloudChunk(text="hello there")
        self._fail = fail

    def transcribe(self, audio, language, key):
        self.calls.append((len(audio), language))
        if self._fail:
            raise self._fail
        return self._chunk


class LocalStub(SpeechEngine):
    name = "whisper"
    title = "Standard"

    def __init__(self):
        self.called = False

    def transcribe(self, request):
        self.called = True
        return EngineResult(
            segments=[Segment(0, 0, 1, "recovered locally")],
            language="en",
            model="stub",
            engine_version="stub 1.0",
        )


class TestChunking:
    def test_long_audio_is_split_into_complete_wav_files(self, wav_file):
        chunks = cloud.split_wav(wav_file, seconds=8)
        assert len(chunks) == 3
        for offset, blob in chunks:
            with wave.open(io.BytesIO(blob), "rb") as handle:
                assert handle.getframerate() == 16000
                assert handle.getnchannels() == 1
        assert [round(o) for o, _ in chunks] == [0, 8, 16]

    def test_short_audio_is_one_chunk(self, wav_file):
        assert len(cloud.split_wav(wav_file, seconds=600)) == 1

    def test_the_default_chunk_stays_under_the_upload_limit(self):
        # 16 kHz mono 16-bit is 32,000 bytes a second.
        assert cloud.CHUNK_SECONDS * 32_000 < cloud.MAX_UPLOAD_BYTES


class TestEstimatedTimings:
    def test_text_is_spread_over_the_chunk(self):
        spans = cloud._estimate_spans("One. Two. Three.", 9.0)
        assert len(spans) == 3
        assert spans[0][0] == 0.0
        assert spans[-1][1] == pytest.approx(9.0)

    def test_the_spans_do_not_overlap_or_go_backwards(self):
        spans = cloud._estimate_spans("A. Longer sentence here. B.", 10.0)
        for before, after in zip(spans, spans[1:]):
            assert before[1] == pytest.approx(after[0])

    def test_empty_text_produces_nothing(self):
        assert cloud._estimate_spans("   ", 5.0) == []


class TestEngine:
    def test_without_a_key_it_does_not_upload_and_falls_back(self, wav_file, monkeypatch):
        monkeypatch.setattr(cloud, "first_key", lambda provider: None)
        provider = RecordingProvider()
        local = LocalStub()
        engine = cloud.CloudEngine(provider, fallback=local)

        result = engine.transcribe(TranscriptionRequest(audio=wav_file, duration=20))

        assert provider.calls == [], "nothing may be uploaded without a key"
        assert local.called
        assert result.segments[0].text == "recovered locally"
        assert "fell back from cloud" in result.engine_version

    def test_a_failure_falls_back_rather_than_losing_the_recording(self, wav_file, monkeypatch):
        monkeypatch.setattr(cloud, "first_key", lambda provider: KEY)
        local = LocalStub()
        engine = cloud.CloudEngine(
            # The message real code produces, from _request's own formatting.
            RecordingProvider(
                fail=cloud.CloudHttpError(
                    "the service answered 503 Service Unavailable", 503
                )
            ),
            fallback=local,
        )
        result = engine.transcribe(TranscriptionRequest(audio=wav_file, duration=20))
        assert local.called
        assert result.segments[0].text == "recovered locally"
        assert engine.fell_back_because and "503" in engine.fell_back_because

    def test_with_no_fallback_a_failure_is_reported(self, wav_file, monkeypatch):
        monkeypatch.setattr(cloud, "first_key", lambda provider: KEY)
        engine = cloud.CloudEngine(RecordingProvider(fail=EngineError("nope")), fallback=None)
        with pytest.raises(EngineError, match="no local engine to fall back to"):
            engine.transcribe(TranscriptionRequest(audio=wav_file, duration=20))

    def test_every_chunk_is_sent_and_offsets_are_applied(self, wav_file, monkeypatch):
        monkeypatch.setattr(cloud, "first_key", lambda provider: KEY)
        provider = RecordingProvider(
            cloud.CloudChunk(text="a sentence", spans=[(0.0, 2.0, "a sentence")])
        )
        engine = cloud.CloudEngine(provider, chunk_seconds=8)
        result = engine.transcribe(TranscriptionRequest(audio=wav_file, duration=20))

        assert len(provider.calls) == 3
        assert [round(s.start) for s in result.segments] == [0, 8, 16]
        assert [s.index for s in result.segments] == [0, 1, 2]

    def test_the_language_is_passed_through(self, wav_file, monkeypatch):
        monkeypatch.setattr(cloud, "first_key", lambda provider: KEY)
        provider = RecordingProvider()
        cloud.CloudEngine(provider, chunk_seconds=600).transcribe(
            TranscriptionRequest(audio=wav_file, duration=20, languages=("fr",))
        )
        assert provider.calls[0][1] == "fr"

    def test_no_confidence_is_invented_for_cloud_text(self, wav_file, monkeypatch):
        monkeypatch.setattr(cloud, "first_key", lambda provider: KEY)
        engine = cloud.CloudEngine(RecordingProvider(), chunk_seconds=600)
        result = engine.transcribe(TranscriptionRequest(audio=wav_file, duration=20))
        assert all(s.confidence is None for s in result.segments)

    def test_it_declares_that_it_uploads(self):
        assert cloud.CloudEngine(RecordingProvider()).uploads_audio is True


class TestProviders:
    def test_the_openai_shape_is_parsed_including_timings(self):
        payload = {
            "text": "hello world",
            "language": "english",
            "segments": [
                {"start": 0.0, "end": 1.5, "text": " hello"},
                {"start": 1.5, "end": 3.0, "text": " world"},
            ],
        }
        chunk = cloud.OpenAICompatibleProvider._parse(json.dumps(payload).encode())
        assert chunk.text == "hello world"
        assert chunk.spans == [(0.0, 1.5, "hello"), (1.5, 3.0, "world")]

    def test_a_response_without_segments_still_yields_text(self):
        chunk = cloud.OpenAICompatibleProvider._parse(b'{"text": "just words"}')
        assert chunk.text == "just words" and chunk.spans == []

    def test_a_response_that_is_not_json_is_reported(self):
        with pytest.raises(cloud.CloudHttpError, match="not JSON"):
            cloud.OpenAICompatibleProvider._parse(b"<html>502</html>")

    def test_known_providers_are_built_with_sensible_models(self):
        assert cloud.build_provider("groq").model == "whisper-large-v3"
        assert cloud.build_provider("openai").model == "whisper-1"
        assert "gemini" in cloud.build_provider("gemini").model

    def test_an_unknown_provider_lists_the_real_ones(self):
        with pytest.raises(Exception, match="groq"):
            cloud.build_provider("nonesuch")

    def test_the_host_is_named_so_a_person_can_see_where_audio_goes(self):
        assert cloud.build_provider("groq").endpoint_host == "api.groq.com"


class TestHttp:
    def test_a_multipart_body_carries_the_file_and_the_fields(self):
        body, content_type = cloud._multipart({"model": "whisper-large-v3"}, "a.wav", b"RIFF")
        assert "multipart/form-data; boundary=" in content_type
        assert b'name="model"' in body and b"whisper-large-v3" in body
        assert b'filename="a.wav"' in body and b"RIFF" in body

    def test_retryable_statuses_are_retried_then_reported(self, monkeypatch):
        attempts = []

        def always_503(request, timeout=None):
            attempts.append(1)
            raise urllib.error.HTTPError("u", 503, "Service Unavailable", {}, io.BytesIO(b""))

        monkeypatch.setattr(cloud.urllib.request, "urlopen", always_503)
        with pytest.raises(cloud.CloudHttpError) as caught:
            cloud._request("https://example.test/x", data=b"", sleep=lambda _: None)
        assert len(attempts) == cloud._MAX_ATTEMPTS
        assert caught.value.status == 503

    def test_a_bad_request_is_not_retried(self, monkeypatch):
        attempts = []

        def always_400(request, timeout=None):
            attempts.append(1)
            raise urllib.error.HTTPError("u", 400, "Bad Request", {}, io.BytesIO(b""))

        monkeypatch.setattr(cloud.urllib.request, "urlopen", always_400)
        with pytest.raises(cloud.CloudHttpError):
            cloud._request("https://example.test/x", data=b"", sleep=lambda _: None)
        assert len(attempts) == 1, "retrying a bad request only wastes quota"

    def test_an_unauthorised_key_is_not_retried_either(self, monkeypatch):
        attempts = []

        def always_401(request, timeout=None):
            attempts.append(1)
            raise urllib.error.HTTPError("u", 401, "Unauthorized", {}, io.BytesIO(b""))

        monkeypatch.setattr(cloud.urllib.request, "urlopen", always_401)
        with pytest.raises(cloud.CloudHttpError):
            cloud._request("https://example.test/x", data=b"", sleep=lambda _: None)
        assert len(attempts) == 1

    def test_the_service_s_own_explanation_is_included(self, monkeypatch):
        body = io.BytesIO(json.dumps({"error": {"message": "model is overloaded"}}).encode())

        def failing(request, timeout=None):
            raise urllib.error.HTTPError("u", 400, "Bad Request", {}, body)

        monkeypatch.setattr(cloud.urllib.request, "urlopen", failing)
        with pytest.raises(cloud.CloudHttpError, match="model is overloaded"):
            cloud._request("https://example.test/x", data=b"", sleep=lambda _: None)

    def test_a_request_carries_the_key_as_a_bearer_token(self, monkeypatch):
        captured = {}

        class Response:
            def read(self):
                return b'{"text": "ok"}'

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

        def capture(request, timeout=None):
            captured["headers"] = request.headers
            return Response()

        monkeypatch.setattr(cloud.urllib.request, "urlopen", capture)
        cloud.build_provider("groq").transcribe(b"RIFF", "en", KEY)
        assert captured["headers"]["Authorization"] == f"Bearer {KEY.value}"


class TestSecrecy:
    def test_the_key_is_not_in_a_failure_message(self, wav_file, monkeypatch):
        monkeypatch.setattr(cloud, "first_key", lambda provider: KEY)
        engine = cloud.CloudEngine(
            RecordingProvider(fail=cloud.CloudHttpError("401 Unauthorized", 401)),
            fallback=LocalStub(),
        )
        engine.transcribe(TranscriptionRequest(audio=wav_file, duration=20))
        assert KEY.value not in (engine.fell_back_because or "")

    def test_the_key_is_not_in_the_unavailability_reason(self, monkeypatch):
        monkeypatch.setattr(cloud, "first_key", lambda provider: KEY)
        usable, reason = cloud.CloudEngine("groq").availability()
        assert usable and KEY.value not in reason
