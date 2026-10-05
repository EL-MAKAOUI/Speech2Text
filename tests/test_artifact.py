"""The bundle on disk, and the contract other applications read."""

from __future__ import annotations

import json

import pytest

from speech2text import artifact
from speech2text.model import MediaIdentity, Segment, Transcript


class TestSlug:
    @pytest.mark.parametrize(
        "name,expected",
        [
            ("interview.mp4", "interview"),
            ("Réunion équipe.m4a", "reunion-equipe"),
            ("a  b__c!.wav", "a-b-c"),
            ("....", "recording"),
            ("محاضرة.mp3", "recording"),
            ("", "recording"),
        ],
    )
    def test_folder_names_stay_recognisable_and_safe(self, name, expected):
        assert artifact.slug(name) == expected

    def test_long_names_are_trimmed(self):
        assert len(artifact.slug("x" * 200)) <= 48


class TestBundle:
    def test_every_file_is_written(self, sample_transcript, tmp_path):
        bundle = artifact.write_bundle(sample_transcript, tmp_path / "b")
        for name in (
            artifact.TRANSCRIPT_JSON,
            artifact.TRANSCRIPT_TXT,
            artifact.TRANSCRIPT_RAW_TXT,
            artifact.CONSUMER_JSON,
            artifact.PROJECT_JSON,
        ):
            assert (bundle.directory / name).exists(), name

    def test_raw_and_corrected_text_differ_after_a_correction(self, sample_transcript, tmp_path):
        sample_transcript.correct(1, "Today we discuss testing.")
        bundle = artifact.write_bundle(sample_transcript, tmp_path / "b")
        assert "discus testing" in bundle.raw_text_path.read_text()
        assert "discuss testing" in bundle.text_path.read_text()

    def test_refresh_rewrites_the_derived_files_only(self, sample_transcript, tmp_path):
        bundle = artifact.write_bundle(sample_transcript, tmp_path / "b")
        raw_before = bundle.raw_text_path.read_text()

        transcript = artifact.load(bundle.directory)
        transcript.correct(0, "Hello, and a warm welcome.")
        bundle.json_path.write_text(transcript.to_json(), encoding="utf-8")
        refreshed = artifact.refresh(bundle.directory)

        assert refreshed.raw_text_path.read_text() == raw_before
        assert "warm welcome" in refreshed.text_path.read_text()

    def test_load_accepts_a_folder_or_the_json_itself(self, sample_transcript, tmp_path):
        bundle = artifact.write_bundle(sample_transcript, tmp_path / "b")
        assert artifact.load(bundle.directory).media.name == "interview.mp4"
        assert artifact.load(bundle.json_path).media.name == "interview.mp4"

    def test_loading_something_that_is_not_there_says_so(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            artifact.load(tmp_path)

    def test_the_partial_file_is_removed_when_the_run_finishes(self, sample_transcript, tmp_path):
        target = tmp_path / "b"
        artifact.write_partial(target, "half of it")
        assert (target / artifact.TRANSCRIPT_PARTIAL_TXT).exists()
        artifact.write_bundle(sample_transcript, target)
        assert not (target / artifact.TRANSCRIPT_PARTIAL_TXT).exists()

    def test_the_partial_file_explains_itself(self, tmp_path):
        path = artifact.write_partial(tmp_path / "b", "so far")
        body = path.read_text()
        assert "in progress" in body and "so far" in body


class TestConsumerContract:
    def test_it_holds_what_another_application_needs(self, sample_transcript):
        payload = artifact.consumer_payload(sample_transcript)
        assert payload["schema"] == artifact.CONSUMER_SCHEMA
        assert payload["kind"] == "speech-transcript"
        assert payload["source"]["name"] == "interview.mp4"
        assert payload["source"]["sha256"] == "a" * 64
        assert payload["language"] == "en"
        assert payload["unresolved_reviews"] == 1
        assert payload["text"].startswith("Hello and welcome.")
        assert len(payload["segments"]) == 3

    def test_it_reports_corrected_text_not_raw(self, sample_transcript):
        sample_transcript.correct(1, "Today we discuss testing.")
        payload = artifact.consumer_payload(sample_transcript)
        assert "discuss testing" in payload["text"]
        assert payload["unresolved_reviews"] == 0
        assert payload["corrections"] == 1

    def test_it_is_plain_json(self, sample_transcript):
        json.dumps(artifact.consumer_payload(sample_transcript))

    def test_empty_segments_are_left_out(self):
        transcript = Transcript(
            media=MediaIdentity("a", "/a", 1, 2.0, "wav"),
            segments=[Segment(0, 0, 1, "   "), Segment(1, 1, 2, "real")],
        )
        payload = artifact.consumer_payload(transcript)
        assert [s["text"] for s in payload["segments"]] == ["real"]


class TestFindingEarlierWork:
    def test_bundles_are_listed_most_recent_first(self, sample_transcript, tmp_path):
        for index, stamp in enumerate(["2026-01-01T00:00:00+00:00", "2026-06-01T00:00:00+00:00"]):
            sample_transcript.created_at = stamp
            artifact.write_bundle(sample_transcript, tmp_path / f"run{index}")
        found = artifact.find_previous(tmp_path)
        assert [run.created_at for run in found] == [
            "2026-06-01T00:00:00+00:00",
            "2026-01-01T00:00:00+00:00",
        ]

    def test_a_bundle_without_consumer_json_is_still_described(self, sample_transcript, tmp_path):
        bundle = artifact.write_bundle(sample_transcript, tmp_path / "b")
        bundle.consumer_path.unlink()
        found = artifact.find_previous(tmp_path)
        assert len(found) == 1
        assert found[0].name == "interview.mp4"
        assert found[0].unresolved_reviews == 1

    def test_an_unreadable_folder_is_skipped_not_fatal(self, sample_transcript, tmp_path):
        artifact.write_bundle(sample_transcript, tmp_path / "good")
        broken = tmp_path / "broken"
        broken.mkdir()
        (broken / artifact.TRANSCRIPT_JSON).write_text("{not json", encoding="utf-8")
        assert len(artifact.find_previous(tmp_path)) == 1

    def test_nothing_on_disk_is_not_an_error(self, tmp_path):
        assert artifact.find_previous(tmp_path / "nowhere") == []

    def test_the_label_says_what_is_worth_knowing(self, sample_transcript, tmp_path):
        artifact.write_bundle(sample_transcript, tmp_path / "b")
        label = artifact.find_previous(tmp_path)[0].label
        assert "interview.mp4" in label and "0:12" in label and "1 to check" in label


class TestProject:
    def test_review_position_is_remembered(self, sample_transcript, tmp_path):
        bundle = artifact.write_bundle(sample_transcript, tmp_path / "b")
        artifact.remember_review_position(bundle.directory, 7)
        assert artifact.load_project(bundle.directory)["review_position"] == 7

    def test_a_missing_project_file_reads_as_empty(self, tmp_path):
        assert artifact.load_project(tmp_path) == {}

    def test_allocate_numbers_a_batch_in_order(self, tmp_path):
        first = artifact.allocate(tmp_path, "a.wav", 1)
        second = artifact.allocate(tmp_path, "b.wav", 2)
        assert first.name == "001-a" and second.name == "002-b"
