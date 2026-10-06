"""The guide is checked against the code, so it cannot quietly go stale.

Guidance that is wrong is worse than guidance that is missing, because it is
trusted. Anything a user must know about is asserted here, so adding a
feature without documenting it fails the build.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from speech2text import export, languages, summarize
from speech2text.engines import ENGINE_NAMES, create
from speech2text.engines.whisper import MODEL_SIZES, _CACHE_ENV
from speech2text.ui.guide_window import (
    REVIEW_SHORTCUTS,
    SHORTCUTS,
    TOPICS,
    as_html,
    as_markdown,
)

GUIDE = as_markdown()
#: The guide with its line wrapping collapsed. Prose assertions use this, so
#: rewrapping a paragraph does not fail a test about what it says.
GUIDE_FLAT = " ".join(GUIDE.split())
REPOSITORY = Path(__file__).resolve().parents[1]
GUIDE_FILE = REPOSITORY / "docs" / "guide.md"
README = REPOSITORY / "README.md"


class TestCoverage:
    @pytest.mark.parametrize("name", ENGINE_NAMES)
    def test_every_engine_is_described(self, name):
        engine = create(name)
        assert engine.title in GUIDE, f"{name} is not in the guide"

    def test_the_cloud_engine_s_upload_is_stated_plainly(self):
        assert "uploaded" in GUIDE.lower()
        assert "CLOUD • AUDIO IS UPLOADED" in GUIDE

    def test_it_says_the_cloud_falls_back_so_nothing_is_lost(self):
        assert "transcribed locally instead" in GUIDE_FLAT

    @pytest.mark.parametrize("fmt", export.FORMATS)
    def test_every_export_format_is_listed(self, fmt):
        assert re.search(rf"`{re.escape(fmt)}`", GUIDE), f"{fmt} is not in the guide"

    @pytest.mark.parametrize("task", sorted(summarize.TASKS))
    def test_every_summary_task_is_listed(self, task):
        assert f"`{task}`" in GUIDE, f"the {task} task is not in the guide"

    @pytest.mark.parametrize("key", sorted(SHORTCUTS))
    def test_every_shortcut_is_listed(self, key):
        assert f"`{key}`" in GUIDE, f"{key} is not in the guide"

    @pytest.mark.parametrize("key", sorted(REVIEW_SHORTCUTS))
    def test_every_checking_shortcut_is_listed(self, key):
        assert f"`{key}`" in GUIDE, f"{key} is not in the guide"

    def test_checking_a_transcript_is_explained(self):
        assert "Check it…" in GUIDE_FLAT
        assert "Play as I move" in GUIDE_FLAT
        assert "It's correct" in GUIDE_FLAT
        assert "remembers where you were" in GUIDE_FLAT

    def test_the_default_model_in_the_guide_is_the_real_one(self):
        from speech2text.engines.whisper import DEFAULT_MODEL

        assert f"`{DEFAULT_MODEL}` is the default" in GUIDE_FLAT

    @pytest.mark.parametrize("variable", ["SPEECH2TEXT_MODEL", "SPEECH2TEXT_DEVICE"])
    def test_the_default_overrides_are_documented(self, variable):
        assert variable in GUIDE

    def test_the_model_cache_variable_is_documented(self):
        assert _CACHE_ENV in GUIDE

    def test_fetching_a_model_in_advance_is_documented(self):
        assert "speech2text models" in GUIDE

    def test_where_a_model_runs_is_explained(self):
        """A GPU too small for the model is the first thing a laptop hits."""
        assert "`--device cpu`" in GUIDE and "`--device cuda`" in GUIDE
        assert "laptop GPU" in GUIDE_FLAT

    def test_the_missing_gpu_library_is_explained(self):
        """libcublas is the other way a GPU fails, and it reads as our bug."""
        assert "libcublas" in GUIDE

    @pytest.mark.parametrize(
        "variable",
        ["SPEECH2TEXT_GROQ_API_KEY", "SPEECH2TEXT_GEMINI_API_KEY",
         "SPEECH2TEXT_OPENAI_API_KEY"],
    )
    def test_every_key_variable_is_documented(self, variable):
        assert variable in GUIDE

    def test_the_bundle_files_are_explained(self):
        for name in ("transcript.txt", "transcript.raw.txt", "transcript.json",
                     "consumer.json", "project.json", "transcript.partial.txt"):
            assert f"`{name}`" in GUIDE, f"{name} is not explained"

    def test_reading_a_long_recording_early_is_explained(self):
        assert "transcript.partial.txt" in GUIDE
        assert "while it runs" in GUIDE.lower()

    def test_the_promise_about_extensions_is_made(self):
        lowered = GUIDE_FLAT.lower()
        assert "extension" in lowered
        assert "name of the file does not matter" in lowered

    def test_automatic_language_detection_is_explained_with_its_risk(self):
        assert "Detect automatically" in GUIDE
        assert "music" in GUIDE_FLAT, "the guide must say when detection goes wrong"

    def test_the_number_of_languages_is_right(self):
        stated = re.search(r"knows (\d+) languages", GUIDE_FLAT)
        assert stated, "the guide must say how many languages are supported"
        # Whisper's own 99, which is what the Standard engine offers.
        assert int(stated.group(1)) == 99
        assert len(languages.LANGUAGES) == 100, "99 plus Cantonese"

    def test_whisper_model_sizes_mentioned_in_the_guide_are_real(self):
        for quoted in re.findall(r"`(large-v3|tiny|base|small|medium)`", GUIDE):
            assert quoted in MODEL_SIZES, f"{quoted} is not a real model size"

    def test_the_summarizer_says_it_sends_text_and_not_audio(self):
        assert "**text** (not the audio)" in GUIDE_FLAT

    def test_it_says_corrections_never_overwrite_the_recognition(self):
        assert "never overwrites" in GUIDE_FLAT or "never modified" in GUIDE_FLAT


class TestRendering:
    def test_every_topic_has_a_title_and_a_body(self):
        for topic in TOPICS:
            assert topic.title.strip() and topic.body.strip()

    def test_the_markdown_has_a_heading_for_each_topic(self):
        for topic in TOPICS:
            assert f"## {topic.title}" in GUIDE

    def test_the_html_is_balanced(self):
        html = as_html()
        for tag in ("table", "body", "html"):
            assert html.count(f"<{tag}>") == html.count(f"</{tag}>"), tag

    def test_the_html_escapes_content_rather_than_injecting_it(self, monkeypatch):
        from speech2text.ui import guide_window
        from speech2text.ui.guide_window import Topic

        monkeypatch.setattr(
            guide_window, "TOPICS",
            (Topic("Danger & <risk>", "A <script>alert(1)</script> and a & sign."),),
        )
        html = guide_window.as_html()
        assert "<script>" not in html
        assert "&lt;script&gt;" in html
        assert "Danger &amp; &lt;risk&gt;" in html

    def test_links_survive_into_the_html(self):
        assert 'href="https://console.groq.com/keys"' in as_html()


class TestTheWrittenCopy:
    def test_docs_guide_is_the_generated_guide(self):
        """``docs/guide.md`` is generated; regenerate it when the guide changes."""
        assert GUIDE_FILE.exists(), "docs/guide.md is missing"
        assert GUIDE_FILE.read_text(encoding="utf-8") == GUIDE, (
            "docs/guide.md is out of date — regenerate it with "
            "scripts/build-docs.py"
        )

    def test_the_readme_points_at_the_guide(self):
        assert "docs/guide.md" in README.read_text(encoding="utf-8")
