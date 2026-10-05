"""A file is what is inside it, not what it is called."""

from __future__ import annotations

import wave

import pytest
from conftest import requires_ffmpeg

from speech2text import media


@requires_ffmpeg
class TestProbe:
    def test_reads_a_plain_audio_file(self, tone_wav):
        info = media.probe(tone_wav)
        assert info.has_audio and not info.has_video
        assert info.kind == "audio"
        assert info.sample_rate == 16000
        assert info.channels == 1
        assert 2.8 < info.duration < 3.2

    def test_a_video_renamed_dat_is_still_a_video(self, misnamed_video):
        info = media.probe(misnamed_video)
        assert info.has_video and info.has_audio
        assert info.kind == "video"
        assert info.video_codec == "h264"

    def test_a_file_with_no_extension_is_read_normally(self, extensionless_audio):
        info = media.probe(extensionless_audio)
        assert info.has_audio
        assert info.container == "wav"
        assert info.path.suffix == "", "the fixture must really have no suffix"

    def test_a_wav_called_mp3_is_reported_as_a_wav(self, wav_called_mp3):
        # Trusting the extension here would hand the decoder the wrong format.
        assert media.probe(wav_called_mp3).container == "wav"

    def test_cover_art_does_not_make_an_mp3_a_video(self, mp3_with_cover_art):
        info = media.probe(mp3_with_cover_art)
        assert info.has_audio
        assert not info.has_video, "an attached picture is not a video stream"
        assert info.kind == "audio"

    def test_stereo_and_other_rates_are_described_as_they_are(self, stereo_audio):
        info = media.probe(stereo_audio)
        assert info.channels == 2
        assert info.sample_rate == 44100

    def test_describe_mentions_the_codecs(self, misnamed_video):
        described = media.probe(misnamed_video).describe()
        assert "h264" in described and "aac" in described


@requires_ffmpeg
class TestProbeRefusals:
    def test_a_video_with_no_sound_says_so(self, silent_video):
        with pytest.raises(media.MediaError, match="no sound track"):
            media.probe(silent_video)

    def test_a_text_file_is_refused(self, not_media):
        with pytest.raises(media.MediaError, match="not audio or video"):
            media.probe(not_media)

    def test_an_empty_file_is_refused(self, empty_file):
        with pytest.raises(media.MediaError, match="empty"):
            media.probe(empty_file)

    def test_a_missing_file_is_refused(self, tmp_path):
        with pytest.raises(media.MediaError, match="does not exist"):
            media.probe(tmp_path / "nothing-here.wav")

    def test_a_folder_is_refused(self, tmp_path):
        with pytest.raises(media.MediaError, match="folder"):
            media.probe(tmp_path)


@requires_ffmpeg
class TestDecode:
    def test_decodes_to_sixteen_kilohertz_mono(self, stereo_audio, tmp_path):
        target = media.decode_to_wav(stereo_audio, tmp_path / "out.wav")
        with wave.open(str(target), "rb") as handle:
            assert handle.getframerate() == media.TARGET_SAMPLE_RATE
            assert handle.getnchannels() == media.TARGET_CHANNELS
            assert handle.getsampwidth() == 2

    def test_a_videos_picture_is_discarded(self, misnamed_video, tmp_path):
        target = media.decode_to_wav(misnamed_video, tmp_path / "out.wav")
        assert media.probe(target).has_video is False

    def test_progress_runs_from_zero_to_one(self, tone_wav, tmp_path):
        seen: list[float] = []
        media.decode_to_wav(
            tone_wav, tmp_path / "out.wav", duration_hint=3.0, progress=seen.append
        )
        assert seen and seen[-1] == 1.0
        assert all(0.0 <= value <= 1.0 for value in seen)
        assert seen == sorted(seen), "progress must not go backwards"

    def test_decoding_something_unreadable_fails_clearly(self, not_media, tmp_path):
        with pytest.raises(media.MediaError, match="could not decode|no audio"):
            media.decode_to_wav(not_media, tmp_path / "out.wav")

    def test_creates_the_destination_folder(self, tone_wav, tmp_path):
        target = media.decode_to_wav(tone_wav, tmp_path / "deep" / "nested" / "a.wav")
        assert target.exists()


class TestHelpers:
    def test_sha256_identifies_the_file(self, tmp_path):
        one = tmp_path / "a.bin"
        two = tmp_path / "b.bin"
        one.write_bytes(b"same")
        two.write_bytes(b"same")
        assert media.sha256(one) == media.sha256(two)
        two.write_bytes(b"different")
        assert media.sha256(one) != media.sha256(two)

    @pytest.mark.parametrize(
        "seconds,expected",
        [(0, "0:00"), (9, "0:09"), (61, "1:01"), (600, "10:00"),
         (3600, "1:00:00"), (3725, "1:02:05"), (-5, "0:00")],
    )
    def test_durations_read_like_a_clock(self, seconds, expected):
        assert media.format_duration(seconds) == expected


class TestSizes:
    @pytest.mark.parametrize(
        "num_bytes,expected",
        [
            (0, "0 B"),
            (512, "512 B"),
            (1024, "1 KB"),
            (150 * 1024 ** 2, "150 MB"),
            (int(3.1 * 1024 ** 3), "3.1 GB"),
            (-10, "0 B"),
        ],
    )
    def test_sizes_read_the_way_a_person_writes_them(self, num_bytes, expected):
        assert media.format_size(num_bytes) == expected
