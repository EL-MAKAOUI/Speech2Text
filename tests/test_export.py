"""Getting the text out, in every format offered."""

from __future__ import annotations

import csv
import io
import json
import re
import zipfile
from xml.etree import ElementTree

import pytest

from speech2text import export
from speech2text.model import MediaIdentity, Segment, Transcript

_W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"


def docx_parts(path) -> dict[str, bytes]:
    with zipfile.ZipFile(path) as archive:
        assert archive.testzip() is None, "the archive is corrupt"
        return {name: archive.read(name) for name in archive.namelist()}


def docx_paragraphs(path) -> list[str]:
    """Read the text back out of a .docx the way a reader would."""
    body = ElementTree.fromstring(docx_parts(path)["word/document.xml"])
    paragraphs = []
    for paragraph in body.iter(f"{_W}p"):
        text = "".join(node.text or "" for node in paragraph.iter(f"{_W}t"))
        if text:
            paragraphs.append(text)
    return paragraphs


class TestTimestamps:
    @pytest.mark.parametrize(
        "seconds,expected",
        [(0, "00:00:00,000"), (1.234, "00:00:01,234"), (61.5, "00:01:01,500"),
         (3661.007, "01:01:01,007"), (-1, "00:00:00,000")],
    )
    def test_subtitle_timestamps(self, seconds, expected):
        assert export.timestamp(seconds) == expected

    def test_rounding_up_a_whole_second_does_not_produce_1000_milliseconds(self):
        assert export.timestamp(1.9999) == "00:00:02,000"

    def test_webvtt_uses_a_full_stop(self):
        assert export.timestamp(1.5, millis_separator=".") == "00:00:01.500"


class TestTextFormats:
    def test_plain_text_is_one_segment_per_line(self, sample_transcript):
        lines = export.to_text(sample_transcript).strip().splitlines()
        assert lines == [
            "Hello and welcome.",
            "Today we discus testing.",
            "Thanks for listening.",
        ]

    def test_text_can_carry_times(self, sample_transcript):
        assert export.to_text(sample_transcript, timestamps=True).startswith("[0:00] ")

    def test_every_format_uses_the_corrected_text(self, sample_transcript):
        sample_transcript.correct(1, "Today we discuss testing.")
        for fmt in ("txt", "md", "srt", "vtt", "csv", "json"):
            rendered = export.render(sample_transcript, fmt)
            assert "discuss testing" in rendered, fmt
            if fmt != "json":
                assert "discus testing" not in rendered, fmt

    def test_the_raw_recognition_stays_in_json(self, sample_transcript):
        sample_transcript.correct(1, "Today we discuss testing.")
        data = json.loads(export.render(sample_transcript, "json"))
        assert data["segments"][1]["text"] == "Today we discus testing."
        assert data["corrections"][0]["text"] == "Today we discuss testing."

    def test_markdown_carries_the_provenance(self, sample_transcript):
        body = export.to_markdown(sample_transcript)
        assert body.startswith("# interview.mp4")
        assert "whisper" in body and "English" in body and "0:12" in body

    def test_an_unknown_format_is_refused(self, sample_transcript):
        with pytest.raises(export.ExportError, match="unknown format"):
            export.render(sample_transcript, "pdf")


class TestSubtitles:
    def test_srt_is_numbered_from_one_with_arrows(self, sample_transcript):
        blocks = export.to_srt(sample_transcript).strip().split("\n\n")
        assert len(blocks) == 3
        assert blocks[0].splitlines()[0] == "1"
        assert "-->" in blocks[0].splitlines()[1]

    def test_vtt_starts_with_its_header(self, sample_transcript):
        assert export.to_vtt(sample_transcript).startswith("WEBVTT\n")

    def test_a_zero_length_segment_still_gets_a_visible_duration(self):
        transcript = _transcript([Segment(0, 5.0, 5.0, "blink")])
        start, end = re.findall(r"(\d\d:\d\d:\d\d,\d\d\d)", export.to_srt(transcript))
        assert start != end, "a subtitle with no duration never appears"

    def test_empty_segments_are_skipped_and_numbering_stays_contiguous(self):
        transcript = _transcript([
            Segment(0, 0, 1, "one"), Segment(1, 1, 2, "   "), Segment(2, 2, 3, "two")
        ])
        numbers = [b.splitlines()[0] for b in export.to_srt(transcript).strip().split("\n\n")]
        assert numbers == ["1", "2"]


class TestCsv:
    def test_it_is_a_readable_table(self, sample_transcript):
        sample_transcript.correct(1, "Today we discuss testing.")
        rows = list(csv.DictReader(io.StringIO(export.to_csv(sample_transcript))))
        assert len(rows) == 3
        assert rows[1]["corrected"] == "yes"
        assert rows[0]["corrected"] == ""
        assert rows[1]["text"] == "Today we discuss testing."
        assert float(rows[2]["start"]) == 7.5


class TestWord:
    def test_the_file_is_a_valid_package(self, sample_transcript, tmp_path):
        path = export.write_docx(sample_transcript, tmp_path / "out.docx")
        parts = docx_parts(path)
        for required in (
            "[Content_Types].xml",
            "_rels/.rels",
            "word/document.xml",
            "word/styles.xml",
            "word/_rels/document.xml.rels",
            "docProps/core.xml",
        ):
            assert required in parts, required
        for name, blob in parts.items():
            ElementTree.fromstring(blob)  # every part must be well-formed XML

    def test_the_text_comes_back_out(self, sample_transcript, tmp_path):
        path = export.write_docx(sample_transcript, tmp_path / "out.docx", timestamps=False)
        paragraphs = docx_paragraphs(path)
        assert paragraphs[0] == "interview.mp4"
        assert "Hello and welcome." in paragraphs
        assert "Thanks for listening." in paragraphs

    def test_corrections_are_what_gets_written(self, sample_transcript, tmp_path):
        sample_transcript.correct(1, "Today we discuss testing.")
        path = export.write_docx(sample_transcript, tmp_path / "out.docx", timestamps=False)
        assert "Today we discuss testing." in docx_paragraphs(path)
        assert "Today we discus testing." not in docx_paragraphs(path)

    def test_timestamps_are_a_separate_run(self, sample_transcript, tmp_path):
        path = export.write_docx(sample_transcript, tmp_path / "out.docx", timestamps=True)
        assert any(p.startswith("[0:00] ") for p in docx_paragraphs(path))

    def test_special_characters_are_escaped(self, tmp_path):
        transcript = _transcript([Segment(0, 0, 1, 'he said "a < b & c > d"')])
        path = export.write_docx(transcript, tmp_path / "out.docx")
        assert any('he said "a < b & c > d"' in p for p in docx_paragraphs(path))

    def test_arabic_is_laid_out_right_to_left(self, tmp_path):
        transcript = _transcript([Segment(0, 0, 1, "مرحبا بالعالم")], language="ar")
        path = export.write_docx(transcript, tmp_path / "out.docx")
        document = docx_parts(path)["word/document.xml"].decode()
        assert "<w:bidi/>" in document
        assert "<w:rtl/>" in document
        assert any("مرحبا بالعالم" in p for p in docx_paragraphs(path))

    def test_english_is_not_marked_right_to_left(self, sample_transcript, tmp_path):
        path = export.write_docx(sample_transcript, tmp_path / "out.docx")
        assert "<w:bidi/>" not in docx_parts(path)["word/document.xml"].decode()

    def test_a_transcript_with_no_speech_still_produces_a_document(self, tmp_path):
        transcript = _transcript([])
        path = export.write_docx(transcript, tmp_path / "out.docx")
        assert "(no speech was recognized)" in docx_paragraphs(path)

    def test_the_output_is_reproducible(self, sample_transcript, tmp_path):
        one = export.write_docx(sample_transcript, tmp_path / "one.docx")
        two = export.write_docx(sample_transcript, tmp_path / "two.docx")
        # core.xml holds a timestamp; every other part must be byte-identical.
        parts_one = {k: v for k, v in docx_parts(one).items() if k != "docProps/core.xml"}
        parts_two = {k: v for k, v in docx_parts(two).items() if k != "docProps/core.xml"}
        assert parts_one == parts_two


class TestWrite:
    @pytest.mark.parametrize("suffix", ["txt", "md", "srt", "vtt", "csv", "json", "docx"])
    def test_the_suffix_chooses_the_format(self, sample_transcript, tmp_path, suffix):
        written = export.write(sample_transcript, tmp_path / f"out.{suffix}")
        assert written.exists() and written.stat().st_size > 0

    def test_folders_are_created(self, sample_transcript, tmp_path):
        written = export.write(sample_transcript, tmp_path / "a" / "b" / "out.txt")
        assert written.exists()

    def test_render_refuses_docx_because_it_is_binary(self, sample_transcript):
        with pytest.raises(export.ExportError, match="binary"):
            export.render(sample_transcript, "docx")


def _transcript(segments, language: str = "en") -> Transcript:
    return Transcript(
        media=MediaIdentity("rec.wav", "/rec.wav", 10, 12.0, "wav"),
        segments=segments,
        language_detected=language,
        engine="test",
    )
