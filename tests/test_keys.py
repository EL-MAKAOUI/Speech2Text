"""Where keys come from, and that they stay out of everything else."""

from __future__ import annotations

import json
import stat

import pytest

from speech2text import keys

SECRET = "gsk_abcdefghijklmnopqrstuvwxyz0123456789"


@pytest.fixture(autouse=True)
def isolated_stores(tmp_path, monkeypatch):
    """Never read or write the real key stores during a test."""
    monkeypatch.setattr(keys, "_LLMKIT_STORE", tmp_path / "llmkit" / "keys.json")
    monkeypatch.setattr(keys, "_OWN_STORE", tmp_path / "own" / "keys.json")
    for name in (
        "GROQ_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY",
        "SPEECH2TEXT_GROQ_API_KEY", "SPEECH2TEXT_GEMINI_API_KEY",
        "SPEECH2TEXT_OPENAI_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)
    return tmp_path


class TestMasking:
    def test_a_key_never_appears_in_its_own_repr(self):
        key = keys.ApiKey("groq", SECRET, "test")
        assert SECRET not in repr(key)
        assert key.masked in repr(key)

    def test_the_mask_shows_just_enough_to_recognise_it(self):
        assert keys.ApiKey("groq", SECRET, "t").masked == "gsk…6789"

    def test_a_very_short_key_is_hidden_completely(self):
        assert keys.ApiKey("groq", "abc", "t").masked == "…"


class TestSources:
    def test_the_environment_is_read(self, monkeypatch):
        monkeypatch.setenv("GROQ_API_KEY", SECRET)
        found = keys.first_key("groq")
        assert found.value == SECRET
        assert found.source == "$GROQ_API_KEY"

    def test_the_application_s_own_variable_wins_over_the_vendor_s(self, monkeypatch):
        monkeypatch.setenv("GROQ_API_KEY", "vendor")
        monkeypatch.setenv("SPEECH2TEXT_GROQ_API_KEY", "ours")
        assert keys.first_key("groq").value == "ours"

    def test_the_environment_wins_over_a_stored_key(self, isolated_stores, monkeypatch):
        keys.save_key("groq", "stored-key")
        monkeypatch.setenv("GROQ_API_KEY", "from-environment")
        assert keys.first_key("groq").value == "from-environment"

    def test_nothing_anywhere_means_no_key(self):
        assert keys.first_key("groq") is None
        assert keys.keys_for("groq") == []

    def test_a_blank_variable_does_not_count_as_a_key(self, monkeypatch):
        monkeypatch.setenv("GROQ_API_KEY", "   ")
        assert keys.first_key("groq") is None


class TestSharedStore:
    @pytest.mark.parametrize(
        "payload",
        [
            {"keys": [{"provider": "gemini", "key": "AIza-one"}]},
            {"gemini": ["AIza-one"]},
            {"gemini": {"api_key": "AIza-one"}},
            {"providers": {"gemini": {"key": "AIza-one"}}},
            [{"provider": "gemini", "value": "AIza-one"}],
            {"keys": [{"provider": "gemini", "secret": "AIza-one"}]},
        ],
    )
    def test_plausible_layouts_of_the_shared_store_are_read(self, isolated_stores, payload):
        path = keys._LLMKIT_STORE.expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")
        assert [k.value for k in keys.keys_for("gemini")] == ["AIza-one"]

    def test_another_provider_s_keys_are_not_returned(self, isolated_stores):
        path = keys._LLMKIT_STORE.expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"keys": [{"provider": "openai", "key": "sk-other"}]}),
            encoding="utf-8",
        )
        assert keys.keys_for("gemini") == []

    def test_a_corrupt_store_is_ignored_rather_than_fatal(self, isolated_stores):
        path = keys._LLMKIT_STORE.expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{ this is not json", encoding="utf-8")
        assert keys.keys_for("gemini") == []

    def test_a_key_in_both_stores_is_offered_once(self, isolated_stores):
        for store in (keys._LLMKIT_STORE, keys._OWN_STORE):
            path = store.expanduser()
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps({"keys": [{"provider": "groq", "key": SECRET}]}),
                encoding="utf-8",
            )
        assert len(keys.keys_for("groq")) == 1


class TestOwnStore:
    def test_a_key_can_be_saved_and_read_back(self, isolated_stores):
        keys.save_key("groq", SECRET, label="laptop")
        found = keys.keys_for("groq")
        assert [k.value for k in found] == [SECRET]
        assert found[0].source == "speech2text"

    def test_the_file_is_readable_only_by_its_owner(self, isolated_stores):
        path = keys.save_key("groq", SECRET)
        assert stat.S_IMODE(path.stat().st_mode) == 0o600

    def test_an_empty_key_is_refused(self, isolated_stores):
        with pytest.raises(ValueError, match="empty"):
            keys.save_key("groq", "   ")

    def test_keys_can_be_forgotten(self, isolated_stores):
        keys.save_key("groq", SECRET)
        keys.save_key("gemini", "AIza-keep")
        assert keys.forget_keys("groq") == 1
        assert keys.keys_for("groq") == []
        assert len(keys.keys_for("gemini")) == 1

    def test_forgetting_what_is_not_there_is_not_an_error(self, isolated_stores):
        assert keys.forget_keys("groq") == 0

    def test_providers_with_keys_counts_them(self, isolated_stores):
        keys.save_key("groq", SECRET)
        keys.save_key("groq", SECRET + "2")
        keys.save_key("gemini", "AIza")
        assert keys.providers_with_keys() == {"gemini": 1, "groq": 2}
