"""Unit tests for FileHandler: validation, sanitization, verification, save, cleanup."""

from io import BytesIO
from pathlib import Path

import pytest

from amadeus_api.errors import FileValidationError
from amadeus_api.files import FileHandler, secure_filename


@pytest.fixture
def file_handler(tmp_path):
    return FileHandler(
        upload_dir=tmp_path / "uploads",
        max_file_size=50 * 1024 * 1024,
        allowed_extensions={".wav", ".mp3", ".flac"},
    )


WAV = b"RIFF" + b"\x00" * 4 + b"WAVE" + b"\x00" * 100


@pytest.mark.unit
class TestValidate:
    def test_accepts_valid_wav(self, file_handler):
        file_handler.validate("test.wav", BytesIO(WAV))  # no raise

    def test_rejects_invalid_extension(self, file_handler):
        with pytest.raises(FileValidationError) as exc:
            file_handler.validate("test.txt", BytesIO(b"data"))
        assert exc.value.error_code == "INVALID_FORMAT"

    def test_rejects_empty(self, file_handler):
        with pytest.raises(FileValidationError) as exc:
            file_handler.validate("test.wav", BytesIO(b""))
        assert exc.value.error_code == "EMPTY_FILE"

    def test_rejects_oversized(self, file_handler):
        file_handler.max_file_size = 1024
        with pytest.raises(FileValidationError) as exc:
            file_handler.validate("test.wav", BytesIO(b"\x00" * 4096))
        assert exc.value.error_code == "FILE_TOO_LARGE"

    def test_rejects_no_filename(self, file_handler):
        with pytest.raises(FileValidationError) as exc:
            file_handler.validate("", BytesIO(WAV))
        assert exc.value.error_code == "NO_FILE"


@pytest.mark.unit
class TestSecureFilename:
    @pytest.mark.parametrize(
        "name",
        [
            "../../../etc/passwd",
            "..\\..\\windows\\system32",
            "test/../../../file.wav",
            "....//....//file.wav",
        ],
    )
    def test_removes_path_traversal(self, name):
        s = secure_filename(name)
        assert ".." not in s
        assert "/" not in s
        assert "\\" not in s

    def test_preserves_valid_filename(self):
        assert secure_filename("my_song_2024.wav") == "my_song_2024.wav"

    def test_handles_empty(self):
        assert secure_filename("") == "unnamed_file"

    def test_removes_leading_dots(self):
        assert not secure_filename("...hidden.wav").startswith(".")

    def test_limits_length(self):
        s = secure_filename("a" * 300 + ".wav")
        assert len(s) <= 255
        assert s.endswith(".wav")


@pytest.mark.unit
class TestVerifyAudio:
    def test_valid_wav(self, file_handler, tmp_path):
        p = tmp_path / "test.wav"
        p.write_bytes(WAV)
        assert file_handler.verify_audio_file(str(p)) is True

    def test_valid_mp3_mpeg_header(self, file_handler, tmp_path):
        p = tmp_path / "test.mp3"
        p.write_bytes(b"\xff\xfb" + b"\x00" * 100)
        assert file_handler.verify_audio_file(str(p)) is True

    def test_rejects_mismatched_extension(self, file_handler, tmp_path):
        p = tmp_path / "fake.mp3"
        p.write_bytes(WAV)  # WAV content, mp3 extension
        assert file_handler.verify_audio_file(str(p)) is False

    def test_rejects_nonexistent(self, file_handler):
        assert file_handler.verify_audio_file("/nonexistent/file.wav") is False


@pytest.mark.unit
class TestSaveUpload:
    def test_creates_job_directory_and_sanitizes(self, file_handler):
        path, name = file_handler.save_upload(
            BytesIO(WAV), "../../../etc/passwd.wav", "job-123"
        )
        assert Path(path).parent.name == "job-123"
        assert ".." not in name and "/" not in name
        assert Path(path).exists()

    def test_rejects_empty(self, file_handler):
        with pytest.raises(FileValidationError) as exc:
            file_handler.save_upload(BytesIO(b""), "empty.wav", "job")
        assert exc.value.error_code == "EMPTY_FILE"

    def test_rejects_content_mismatch(self, file_handler):
        with pytest.raises(FileValidationError) as exc:
            file_handler.save_upload(BytesIO(b"not audio data"), "fake.wav", "job")
        assert exc.value.error_code == "INVALID_CONTENT"


@pytest.mark.unit
class TestCleanup:
    def test_deletes_file_and_empty_dir(self, file_handler):
        job_dir = file_handler.upload_dir / "job-1"
        job_dir.mkdir(parents=True)
        f = job_dir / "test.wav"
        f.write_bytes(b"data")
        file_handler.cleanup_file(str(f))
        assert not f.exists()
        assert not job_dir.exists()

    def test_preserves_nonempty_dir(self, file_handler):
        job_dir = file_handler.upload_dir / "job-2"
        job_dir.mkdir(parents=True)
        a, b = job_dir / "a.wav", job_dir / "b.wav"
        a.write_bytes(b"1")
        b.write_bytes(b"2")
        file_handler.cleanup_file(str(a))
        assert not a.exists()
        assert b.exists() and job_dir.exists()

    def test_handles_nonexistent(self, file_handler):
        file_handler.cleanup_file("/nonexistent/file.wav")  # no raise
