"""The command line, driven the way a person drives it."""

from __future__ import annotations

import json

import pytest
from conftest import requires_ffmpeg, requires_sphinx, requires_whisper

from speech2text import artifact
from speech2text.cli import main


def run(capsys, *argv) -> tuple[int, str, str]:
    code = main(list(argv))
    captured = capsys.readouterr()
    return code, captured.out, captured.err


class TestBasics:
    def test_no_arguments_prints_help(self, capsys):
        code, out, _ = run(capsys)
        assert code == 0 and "transcribe" in out

    def test_version(self, capsys):
        with pytest.raises(SystemExit) as exit_info:
            main(["--version"])
        assert exit_info.value.code == 0

    def test_engines_lists_all_of_them(self, capsys):
        code, out, _ = run(capsys, "engines")
        assert code == 0
        for name in ("whisper", "sphinx", "cloud"):
            assert name in out
        assert "UPLOADS AUDIO" in out, "the cloud engine must be marked"

    def test_languages_lists_auto_first(self, capsys):
        code, out, _ = run(capsys, "languages")
        assert code == 0
        assert out.splitlines()[0].startswith("auto")
        assert "ar" in out and "Arabic" in out
        assert len(out.splitlines()) == 101


@requires_ffmpeg
class TestInfo:
    def test_it_describes_what_a_file_really_is(self, capsys, misnamed_video):
        code, out, _ = run(capsys, "info", str(misnamed_video))
        assert code == 0
        assert "video" in out and "h264" in out and "mov" in out

    def test_a_file_with_no_audio_is_reported_and_fails(self, capsys, silent_video):
        code, out, _ = run(capsys, "info", str(silent_video))
        assert code == 1
        assert "no sound track" in out


@requires_ffmpeg
@requires_sphinx
class TestTranscribe:
    def test_it_writes_a_bundle(self, capsys, tone_wav, tmp_path):
        code, _, err = run(
            capsys, "transcribe", str(tone_wav), "-e", "sphinx",
            "-o", str(tmp_path), "--quiet",
        )
        assert code == 0
        found = artifact.find_previous(tmp_path)
        assert len(found) == 1
        assert found[0].name == "tone.wav"

    def test_extra_formats_are_written_alongside(self, capsys, tone_wav, tmp_path):
        run(capsys, "transcribe", str(tone_wav), "-e", "sphinx", "-o", str(tmp_path),
            "-f", "srt", "-f", "docx", "--quiet")
        directory = artifact.find_previous(tmp_path)[0].directory
        assert (directory / "transcript.srt").exists()
        assert (directory / "transcript.docx").exists()

    def test_stdout_prints_the_text_itself(self, capsys, tone_wav, tmp_path):
        code, out, _ = run(capsys, "transcribe", str(tone_wav), "-e", "sphinx",
                           "-o", str(tmp_path), "--stdout", "--quiet")
        assert code == 0
        assert isinstance(out, str)

    def test_a_missing_file_is_refused_before_anything_runs(self, capsys, tmp_path):
        code, _, err = run(capsys, "transcribe", str(tmp_path / "ghost.mp3"), "--quiet")
        assert code == 1 and "no such file" in err

    def test_an_unknown_language_is_refused_with_advice(self, capsys, tone_wav, tmp_path):
        code, _, err = run(capsys, "transcribe", str(tone_wav), "-l", "klingon",
                           "-o", str(tmp_path), "--quiet")
        assert code == 1 and "'auto'" in err

    def test_a_bad_file_in_a_batch_does_not_stop_the_good_one(
        self, capsys, silent_video, tone_wav, tmp_path
    ):
        code, _, err = run(
            capsys, "transcribe", str(silent_video), str(tone_wav),
            "-e", "sphinx", "-o", str(tmp_path), "--quiet",
        )
        assert code == 1, "a failure must be reported in the exit code"
        assert "no sound track" in err
        assert len(artifact.find_previous(tmp_path)) == 1

    def test_the_language_is_recorded_in_the_bundle(self, capsys, tone_wav, tmp_path):
        run(capsys, "transcribe", str(tone_wav), "-e", "sphinx", "-l", "en",
            "-o", str(tmp_path), "--quiet")
        transcript = artifact.load(artifact.find_previous(tmp_path)[0].directory)
        assert transcript.languages_requested == ("en",)


class TestExportAndList:
    """Export renders a transcript, so it is tested against a known one."""

    @pytest.fixture
    def bundle_dir(self, sample_transcript, tmp_path):
        bundle = artifact.write_bundle(sample_transcript, tmp_path / "run" / "001-interview")
        return bundle.directory

    @pytest.mark.parametrize("suffix", ["txt", "md", "srt", "vtt", "csv", "json", "docx"])
    def test_every_format_can_be_exported(self, capsys, bundle_dir, tmp_path, suffix):
        target = tmp_path / f"out.{suffix}"
        code, out, _ = run(capsys, "export", str(bundle_dir), str(target))
        assert code == 0
        assert target.exists() and target.stat().st_size > 0
        assert str(target) in out

    def test_the_exported_text_is_the_transcript(self, capsys, bundle_dir, tmp_path):
        target = tmp_path / "out.txt"
        run(capsys, "export", str(bundle_dir), str(target))
        assert "Hello and welcome." in target.read_text()

    def test_an_explicit_format_overrides_the_suffix(self, capsys, bundle_dir, tmp_path):
        target = tmp_path / "subtitles.text"
        run(capsys, "export", str(bundle_dir), str(target), "-f", "srt")
        assert "-->" in target.read_text()

    def test_exporting_from_the_json_file_works_too(self, capsys, bundle_dir, tmp_path):
        target = tmp_path / "out.txt"
        code, _, _ = run(capsys, "export", str(bundle_dir / "transcript.json"), str(target))
        assert code == 0 and target.exists()

    def test_exporting_something_that_is_not_there_is_refused(self, capsys, tmp_path):
        code, _, err = run(capsys, "export", str(tmp_path / "nope"), str(tmp_path / "a.txt"))
        assert code == 1 and "no transcript" in err

    def test_list_shows_what_was_transcribed(self, capsys, bundle_dir, tmp_path):
        code, out, _ = run(capsys, "list", "-o", str(tmp_path))
        assert code == 0 and "interview.mp4" in out
        assert "1 to check" in out

    def test_list_on_an_empty_folder_says_so(self, capsys, tmp_path):
        code, out, _ = run(capsys, "list", "-o", str(tmp_path / "empty"))
        assert code == 0 and "nothing transcribed" in out


@requires_ffmpeg
@requires_sphinx
class TestExportingSilence:
    def test_audio_with_no_speech_exports_an_empty_text_file(
        self, capsys, tone_wav, tmp_path
    ):
        """Not an error: a tone has no words, so the text is empty."""
        run(capsys, "transcribe", str(tone_wav), "-e", "sphinx",
            "-o", str(tmp_path), "--quiet")
        directory = artifact.find_previous(tmp_path)[0].directory
        target = tmp_path / "silent.txt"
        code, _, _ = run(capsys, "export", str(directory), str(target))
        assert code == 0 and target.exists()
        assert target.read_text().strip() == ""


class TestKeys:
    @pytest.fixture(autouse=True)
    def isolated(self, tmp_path, monkeypatch):
        from speech2text import keys as keystore

        monkeypatch.setattr(keystore, "_LLMKIT_STORE", tmp_path / "llmkit.json")
        monkeypatch.setattr(keystore, "_OWN_STORE", tmp_path / "own.json")
        for name in ("GROQ_API_KEY", "GEMINI_API_KEY", "OPENAI_API_KEY",
                     "GOOGLE_API_KEY", "SPEECH2TEXT_GROQ_API_KEY"):
            monkeypatch.delenv(name, raising=False)

    def test_with_no_keys_it_says_they_are_only_for_the_cloud(self, capsys):
        code, out, _ = run(capsys, "keys", "list")
        assert code == 0 and "only needed for the cloud" in out

    def test_a_key_can_be_added_and_is_then_listed_masked(self, capsys):
        secret = "gsk_abcdefghijklmnop"
        code, _, _ = run(capsys, "keys", "add", "-p", "groq", "--value", secret)
        assert code == 0
        code, out, _ = run(capsys, "keys", "list")
        assert code == 0
        assert secret not in out, "a key must never be printed in full"
        assert "gsk…mnop" in out

    def test_a_key_can_be_forgotten(self, capsys):
        run(capsys, "keys", "add", "-p", "groq", "--value", "gsk_something_here")
        code, out, _ = run(capsys, "keys", "forget", "-p", "groq")
        assert code == 0 and "removed 1" in out


@requires_ffmpeg
@requires_sphinx
class TestSummarize:
    def test_without_a_key_it_explains_rather_than_crashing(
        self, capsys, tone_wav, tmp_path, monkeypatch
    ):
        from speech2text import summarize

        monkeypatch.setattr(summarize, "first_key", lambda provider: None)
        run(capsys, "transcribe", str(tone_wav), "-e", "sphinx",
            "-o", str(tmp_path), "--quiet")
        directory = artifact.find_previous(tmp_path)[0].directory
        code, _, err = run(capsys, "summarize", str(directory))
        assert code == 1 and "keys add" in err


class TestFallbackChoice:
    def test_the_cloud_cannot_be_its_own_fallback(self, capsys, tmp_path):
        """A cloud engine falling back to a cloud engine helps nobody."""
        with pytest.raises(SystemExit):
            main(["transcribe", "x.wav", "--engine", "cloud", "--fallback", "cloud"])
        assert "invalid choice" in capsys.readouterr().err

    def test_a_local_engine_is_accepted_as_a_fallback(self, capsys, tmp_path):
        from speech2text.engines import LOCAL_ENGINE_NAMES

        assert "cloud" not in LOCAL_ENGINE_NAMES
        code, _, err = run(capsys, "transcribe", str(tmp_path / "ghost.wav"),
                           "--engine", "cloud", "--fallback", "whisper", "--quiet")
        assert code == 1 and "no such file" in err, "the options parsed fine"


class TestModels:
    def test_it_lists_every_model_and_whether_it_is_here(self, capsys, tmp_path):
        from speech2text.engines.whisper import MODEL_SIZES

        code, out, _ = run(capsys, "models", "--cache", str(tmp_path))
        assert code == 0
        for size in MODEL_SIZES:
            assert size in out, f"{size} is not listed"
        assert out.count("not downloaded") == len(MODEL_SIZES)
        assert str(tmp_path) in out
        assert "SPEECH2TEXT_MODEL_CACHE" in out

    def test_a_downloaded_model_is_reported_as_present(self, capsys, tmp_path):
        snapshot = (
            tmp_path / "models--Systran--faster-whisper-base" / "snapshots" / "rev"
        )
        snapshot.mkdir(parents=True)
        (snapshot / "model.bin").write_bytes(b"weights")
        code, out, _ = run(capsys, "models", "--cache", str(tmp_path))
        base_line = next(line for line in out.splitlines() if line.startswith("base "))
        assert "downloaded" in base_line
        assert "not downloaded" not in base_line

    @requires_whisper
    def test_fetching_one_already_here_does_not_download_again(self, capsys, tmp_path):
        snapshot = (
            tmp_path / "models--Systran--faster-whisper-tiny" / "snapshots" / "rev"
        )
        snapshot.mkdir(parents=True)
        (snapshot / "model.bin").write_bytes(b"x" * 1024)
        code, out, _ = run(capsys, "models", "get", "tiny", "--cache", str(tmp_path))
        assert code == 0 and "already in" in out
        assert "1 KB" in out, "it should say how much disk it is using"

    def test_the_listing_says_how_big_a_downloaded_model_is(self, capsys, tmp_path):
        snapshot = (
            tmp_path / "models--Systran--faster-whisper-base" / "snapshots" / "rev"
        )
        snapshot.mkdir(parents=True)
        (snapshot / "model.bin").write_bytes(b"x" * (3 * 1024 ** 2))
        code, out, _ = run(capsys, "models", "--cache", str(tmp_path))
        base_line = next(line for line in out.splitlines() if line.startswith("base "))
        assert "downloaded, 3 MB" in base_line

    @requires_whisper
    def test_a_download_announces_its_size_so_it_does_not_look_stalled(
        self, capsys, tmp_path, monkeypatch
    ):
        """A silent multi-gigabyte fetch is indistinguishable from a hang."""
        from speech2text.engines.whisper import WhisperEngine

        monkeypatch.setattr(WhisperEngine, "load", lambda self: None)
        code, out, err = run(capsys, "models", "get", "large-v3", "--cache", str(tmp_path))
        assert code == 0
        assert "about ~3.1 GB" in err
        assert "takes a while" in err

    def test_an_unknown_model_is_refused_with_the_real_list(self, capsys):
        with pytest.raises(SystemExit):
            main(["models", "get", "enormous"])
        assert "invalid choice" in capsys.readouterr().err

    @requires_whisper
    def test_a_failed_download_explains_itself_rather_than_crashing(
        self, capsys, tmp_path, monkeypatch
    ):
        from speech2text.engines.whisper import WhisperEngine
        from speech2text.engines.base import EngineUnavailable

        def refuse(self):
            raise EngineUnavailable("the weights come from huggingface.co")

        monkeypatch.setattr(WhisperEngine, "load", refuse)
        code, _, err = run(capsys, "models", "get", "tiny", "--cache", str(tmp_path))
        assert code == 1 and "huggingface.co" in err

    def test_the_quoted_sizes_cover_every_model_offered(self):
        from speech2text.cli import APPROXIMATE_SIZES
        from speech2text.engines.whisper import MODEL_SIZES

        assert set(APPROXIMATE_SIZES) == set(MODEL_SIZES)
