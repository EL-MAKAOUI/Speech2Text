"""The rule the model exists to enforce: recognition is never overwritten."""

from __future__ import annotations

import json

import pytest

from speech2text.model import (
    SCHEMA_VERSION,
    Correction,
    MediaIdentity,
    ReviewIssue,
    Segment,
    Transcript,
    Word,
    renumber,
)


class TestRawTextIsImmutable:
    def test_correcting_leaves_the_recognition_alone(self, sample_transcript):
        before = sample_transcript.segments[1].text
        sample_transcript.correct(1, "Today we discuss testing.")
        assert sample_transcript.segments[1].text == before
        assert "discus testing" in sample_transcript.raw_text()
        assert "discuss testing" in sample_transcript.corrected_text()

    def test_the_latest_correction_wins_and_earlier_ones_are_kept(self, sample_transcript):
        sample_transcript.correct(1, "first attempt")
        sample_transcript.correct(1, "second attempt")
        assert sample_transcript.text_of(1) == "second attempt"
        assert len(sample_transcript.corrections) == 2
        assert sample_transcript.corrections[0].text == "first attempt"

    def test_a_correction_survives_a_round_trip(self, sample_transcript):
        sample_transcript.correct(1, "Today we discuss testing.", note="heard it again")
        restored = Transcript.from_json(sample_transcript.to_json())
        assert restored.text_of(1) == "Today we discuss testing."
        assert restored.segments[1].text == "Today we discus testing."
        assert restored.corrections[0].note == "heard it again"

    def test_correcting_a_segment_that_does_not_exist_is_refused(self, sample_transcript):
        with pytest.raises(IndexError):
            sample_transcript.correct(99, "nope")


class TestReviewQueue:
    def test_least_confident_comes_first(self):
        transcript = Transcript(
            media=_media(),
            segments=[Segment(i, i, i + 1, f"line {i}") for i in range(3)],
            issues=[
                ReviewIssue(0, "low confidence", 0.8),
                ReviewIssue(1, "low confidence", 0.2),
                ReviewIssue(2, "low confidence", 0.5),
            ],
        )
        assert [i.segment_index for i in transcript.unresolved_issues()] == [1, 2, 0]

    def test_correcting_resolves_the_issue(self, sample_transcript):
        assert len(sample_transcript.unresolved_issues()) == 1
        sample_transcript.correct(1, "fixed")
        assert sample_transcript.unresolved_issues() == []

    def test_accepting_clears_the_issue_without_changing_the_text(self, sample_transcript):
        original = sample_transcript.text_of(1)
        sample_transcript.resolve_issue(1, accepted=True)
        assert sample_transcript.unresolved_issues() == []
        assert sample_transcript.text_of(1) == original
        assert sample_transcript.corrections == []
        assert sample_transcript.issues[0].accepted is True


class TestText:
    def test_plain_text_is_one_flowing_line(self, sample_transcript):
        assert "\n" not in sample_transcript.plain_text()
        assert sample_transcript.plain_text().startswith("Hello and welcome.")

    def test_word_count_follows_corrections(self, sample_transcript):
        before = sample_transcript.word_count()
        sample_transcript.correct(1, "one two three four five six seven")
        assert sample_transcript.word_count() == before - 4 + 7

    def test_an_empty_transcript_produces_empty_text(self):
        empty = Transcript(media=_media())
        assert empty.corrected_text().strip() == ""
        assert empty.word_count() == 0
        assert len(empty) == 0


class TestSerialisation:
    def test_round_trip_keeps_everything(self, sample_transcript):
        sample_transcript.correct(0, "Hello, and welcome.")
        restored = Transcript.from_json(sample_transcript.to_json())
        assert restored.to_dict() == sample_transcript.to_dict()

    def test_words_survive_the_round_trip(self):
        transcript = Transcript(
            media=_media(),
            segments=[
                Segment(0, 0, 1, "hi there",
                        words=(Word("hi", 0, 0.4, 0.99), Word("there", 0.4, 1.0, 0.95)))
            ],
        )
        restored = Transcript.from_json(transcript.to_json())
        assert restored.segments[0].words[1].text == "there"
        assert restored.segments[0].words[1].probability == pytest.approx(0.95)

    def test_a_future_schema_is_refused_rather_than_misread(self, sample_transcript):
        data = sample_transcript.to_dict()
        data["schema"] = "9.0"
        with pytest.raises(ValueError, match="schema"):
            Transcript.from_dict(data)

    def test_the_json_is_readable_and_unicode_is_not_escaped(self):
        transcript = Transcript(media=_media(), segments=[Segment(0, 0, 1, "مرحبا")])
        text = transcript.to_json()
        assert "مرحبا" in text
        assert json.loads(text)["schema"] == SCHEMA_VERSION

    def test_optional_fields_are_left_out_rather_than_written_as_null(self):
        segment = Segment(0, 0, 1, "text")
        assert "confidence" not in segment.to_dict()
        assert "words" not in segment.to_dict()


class TestHelpers:
    def test_renumber_makes_indexes_consecutive(self):
        segments = [Segment(7, 0, 1, "a"), Segment(9, 1, 2, "b")]
        assert [s.index for s in renumber(segments)] == [0, 1]

    def test_duration_prefers_the_longer_of_media_and_segments(self):
        transcript = Transcript(
            media=_media(duration=5.0), segments=[Segment(0, 0, 9.0, "long")]
        )
        assert transcript.duration == 9.0

    def test_a_correction_defaults_to_being_human(self):
        assert Correction(0, "x").origin == "human"


def _media(duration: float = 10.0) -> MediaIdentity:
    return MediaIdentity(
        name="a.wav", path="/a.wav", size_bytes=1, duration=duration, container="wav"
    )
