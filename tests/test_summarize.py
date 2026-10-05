"""The optional language-model step."""

from __future__ import annotations

import pytest

from speech2text import summarize
from speech2text.keys import ApiKey

KEY = ApiKey("fake", "fake-key-value", "test")


class Recorder(summarize.TextProvider):
    name = "fake"
    model = "fake-1"

    def __init__(self, reply: str = "a result", fail: Exception | None = None):
        self.calls: list[tuple[str, str]] = []
        self._reply = reply
        self._fail = fail

    def complete(self, instruction, text, key):
        self.calls.append((instruction, text))
        if self._fail:
            raise self._fail
        return self._reply


class TestTasks:
    def test_the_offered_tasks_are_what_the_window_lists(self):
        assert set(summarize.TASKS) == {
            "summary", "key-points", "actions", "minutes", "outline", "tidy"
        }

    def test_every_task_has_an_instruction_and_a_way_to_combine_parts(self):
        for task in summarize.TASKS.values():
            assert task.instruction.strip() and task.combine.strip()
            assert task.title.strip()

    @pytest.mark.parametrize("name", ["summary", "key-points", "actions", "minutes", "tidy"])
    def test_tasks_guard_against_fabrication(self, name):
        """A summary that invents a decision is worse than no summary."""
        instruction = summarize.TASKS[name].instruction.lower()
        guards = (
            "do not add", "do not invent", "rather than inventing",
            "do not invent attendees", "only points actually made",
            "do not summarize, shorten or add",
        )
        assert any(guard in instruction for guard in guards), instruction

    def test_an_unknown_task_lists_the_real_ones(self):
        with pytest.raises(summarize.SummaryError, match="summary"):
            summarize.task_named("make-it-rhyme")

    def test_a_custom_instruction_is_used_verbatim(self):
        task = summarize.custom_task("list every question asked")
        assert task.instruction == "list every question asked"
        assert task.name == "custom"


class TestSplitting:
    def test_short_text_is_left_alone(self):
        assert summarize.split_text("a short transcript", 1000) == ["a short transcript"]

    def test_long_text_is_split_without_losing_words(self):
        text = "Sentence number one. " * 400
        chunks = summarize.split_text(text, 500)
        assert len(chunks) > 1
        assert all(len(chunk) <= 500 for chunk in chunks)
        assert " ".join(chunks).split() == text.split()

    def test_lines_are_preferred_as_boundaries(self):
        text = "\n".join(["line " + str(i) * 20 for i in range(10)])
        for chunk in summarize.split_text(text, 200):
            assert not chunk.startswith(" ")

    def test_a_single_enormous_sentence_is_still_split(self):
        text = "word " * 5000
        chunks = summarize.split_text(text, 400)
        assert all(len(chunk) <= 400 for chunk in chunks)

    def test_empty_text_produces_no_chunks(self):
        assert summarize.split_text("   ") == []

    def test_token_estimate_is_proportional(self):
        assert summarize.estimated_tokens("x" * 400) == 100


class TestRunning:
    def test_a_short_transcript_is_one_request(self):
        provider = Recorder()
        result = summarize.run_task(
            "a short transcript", summarize.task_named("summary"),
            provider=provider, key=KEY,
        )
        assert len(provider.calls) == 1
        assert result.parts == 1
        assert result.text == "a result"
        assert result.provider == "fake" and result.model == "fake-1"

    def test_a_long_transcript_is_mapped_then_combined(self):
        provider = Recorder()
        text = "Sentence one. " * 400
        result = summarize.run_task(
            text, summarize.task_named("summary"),
            provider=provider, key=KEY, chunk_chars=500,
        )
        assert result.parts > 1
        assert len(provider.calls) == result.parts + 1, "one pass per part, then a combine"
        assert provider.calls[-1][0] == summarize.TASKS["summary"].combine

    def test_progress_is_reported_for_each_part(self):
        seen: list[tuple[int, int]] = []
        summarize.run_task(
            "Sentence one. " * 400, summarize.task_named("summary"),
            provider=Recorder(), key=KEY, chunk_chars=500,
            on_progress=lambda n, total: seen.append((n, total)),
        )
        assert seen[0][0] == 1
        assert seen[-1][0] == seen[-1][1]

    def test_without_a_key_it_says_how_to_add_one(self, monkeypatch):
        monkeypatch.setattr(summarize, "first_key", lambda provider: None)
        with pytest.raises(summarize.SummaryError, match="keys add"):
            summarize.run_task("text", summarize.task_named("summary"), provider=Recorder())

    def test_empty_text_is_refused_before_any_request(self):
        provider = Recorder()
        with pytest.raises(summarize.SummaryError, match="no text"):
            summarize.run_task("   ", summarize.task_named("summary"),
                               provider=provider, key=KEY)
        assert provider.calls == []

    def test_a_failed_request_is_reported_plainly(self):
        from speech2text.engines.cloud import CloudHttpError

        provider = Recorder(fail=CloudHttpError("the service answered 429"))
        with pytest.raises(summarize.SummaryError, match="429"):
            summarize.run_task("text", summarize.task_named("summary"),
                               provider=provider, key=KEY)


class TestOnATranscript:
    def test_it_sends_the_corrected_text(self, sample_transcript):
        sample_transcript.correct(1, "Today we discuss testing.")
        provider = Recorder()
        summarize.summarize_transcript(sample_transcript, "summary",
                                       provider=provider, key=KEY)
        sent = provider.calls[0][1]
        assert "discuss testing" in sent and "discus testing" not in sent

    def test_the_result_is_written_beside_the_bundle(self, sample_transcript, tmp_path):
        result = summarize.summarize_transcript(
            sample_transcript, "key-points", provider=Recorder("- one\n- two"), key=KEY
        )
        path = summarize.write_result(result, tmp_path, "interview.mp4")
        assert path.name == "key-points.md"
        body = path.read_text()
        assert "- one" in body and "interview.mp4" in body

    def test_the_markdown_names_what_produced_it(self):
        result = summarize.SummaryResult("summary", "text", "groq", "llama", 1, 100)
        assert "groq llama" in result.to_markdown("a.mp3")


class TestProviders:
    def test_the_known_providers_have_models(self):
        for name in summarize.PROVIDERS:
            assert summarize.build_provider(name).model

    def test_an_unknown_provider_is_refused(self):
        with pytest.raises(summarize.SummaryError, match="groq"):
            summarize.build_provider("nonesuch")
