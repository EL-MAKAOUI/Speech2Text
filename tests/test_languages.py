"""Choosing a language, and the automatic option."""

from __future__ import annotations

import pytest

from speech2text import languages


class TestParsing:
    @pytest.mark.parametrize("value", ["auto", "AUTO", "detect", "automatic", "", "   ", None])
    def test_automatic_means_no_language_is_forced(self, value):
        assert languages.parse(value) == []

    @pytest.mark.parametrize(
        "value,expected",
        [
            ("en", ["en"]),
            ("French", ["fr"]),
            ("french", ["fr"]),
            ("Français", ["fr"]),
            ("العربية", ["ar"]),
            ("EN-US", ["en"]),
            ("pt_BR", ["pt"]),
            ("fr,ar,en", ["fr", "ar", "en"]),
            ("fr, ar ; en", ["fr", "ar", "en"]),
            ("en,en", ["en"]),
        ],
    )
    def test_codes_names_and_lists(self, value, expected):
        assert languages.parse(value) == expected

    def test_auto_anywhere_in_a_list_means_detect(self):
        assert languages.parse("fr,auto") == []

    def test_an_unknown_language_says_what_to_do(self):
        with pytest.raises(languages.UnknownLanguage, match="'auto'"):
            languages.parse("klingon")

    @pytest.mark.parametrize("alias,code", [("jv", "jw"), ("iw", "he"), ("nb", "no")])
    def test_aliases_map_to_the_code_the_engines_use(self, alias, code):
        assert languages.resolve(alias) == code


class TestCatalogue:
    def test_whisper_s_languages_are_all_present(self):
        # Whisper is trained on 99 languages plus Cantonese.
        assert len(languages.LANGUAGES) == 100
        for code in ("en", "fr", "ar", "zh", "yue", "sw", "haw"):
            assert code in languages.LANGUAGES

    def test_right_to_left_scripts_are_marked(self):
        assert languages.is_rtl("ar") and languages.is_rtl("he") and languages.is_rtl("fa")
        assert not languages.is_rtl("en")
        assert not languages.is_rtl(None)

    def test_rtl_handles_a_regional_code(self):
        assert languages.is_rtl("ar-EG")

    def test_common_languages_are_offered_first(self):
        first = [language.code for language in languages.choices()[:6]]
        assert first[0] == "en"
        assert "ar" in first and "fr" in first

    def test_every_language_appears_exactly_once_in_the_list(self):
        codes = [language.code for language in languages.choices()]
        assert len(codes) == len(set(codes)) == len(languages.LANGUAGES)

    def test_names_are_readable_and_native_names_are_shown_when_known(self):
        assert languages.LANGUAGES["ar"].label == "Arabic — العربية"
        assert languages.LANGUAGES["mi"].label == "Maori"

    def test_name_of_covers_auto_and_the_unknown(self):
        assert languages.name_of("auto") == "Detect automatically"
        assert languages.name_of(None) == "Unknown"
        assert languages.name_of("fr") == "French"
        assert languages.name_of("xx") == "xx"
