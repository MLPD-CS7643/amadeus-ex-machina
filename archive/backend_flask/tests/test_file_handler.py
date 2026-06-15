"""
Unit tests for FileHandler class.

Tests file validation, sanitization, and cleanup functionality.
"""
import os
import pytest
from pathlib import Path
from io import BytesIO
from werkzeug.datastructures import FileStorage
from app.file_handler import FileHandler, FileValidationError


@pytest.fixture
def temp_upload_dir(tmp_path):
    """Create temporary upload directory for tests."""
    upload_dir = tmp_path / "uploads"
    upload_dir.mkdir()
    return upload_dir


@pytest.fixture
def file_handler(temp_upload_dir):
    """Create FileHandler instance with test configuration."""
    return FileHandler(
        upload_dir=temp_upload_dir,
        max_file_size=50 * 1024 * 1024,  # 50MB
        allowed_extensions={'.wav', '.mp3', '.flac'}
    )


@pytest.mark.unit
class TestFileValidation:
    """Test file validation functionality."""
    
    def test_validate_valid_wav_file(self, file_handler):
        """Test validation accepts valid WAV file."""
        # Create mock WAV file
        wav_data = b'RIFF' + b'\x00' * 4 + b'WAVE' + b'\x00' * 100
        file = FileStorage(
            stream=BytesIO(wav_data),
            filename='test.wav',
            content_type='audio/wav'
        )
        
        assert file_handler.validate_file(file) is True

    
    def test_validate_valid_mp3_file(self, file_handler):
        """Test validation accepts valid MP3 file."""
        mp3_data = b'ID3' + b'\x00' * 100
        file = FileStorage(
            stream=BytesIO(mp3_data),
            filename='test.mp3',
            content_type='audio/mpeg'
        )
        
        assert file_handler.validate_file(file) is True
    
    def test_validate_valid_flac_file(self, file_handler):
        """Test validation accepts valid FLAC file."""
        flac_data = b'fLaC' + b'\x00' * 100
        file = FileStorage(
            stream=BytesIO(flac_data),
            filename='test.flac',
            content_type='audio/flac'
        )
        
        assert file_handler.validate_file(file) is True
    
    def test_validate_rejects_invalid_extension(self, file_handler):
        """Test validation rejects unsupported file extensions."""
        file = FileStorage(
            stream=BytesIO(b'data'),
            filename='test.txt',
            content_type='text/plain'
        )
        
        assert file_handler.validate_file(file) is False
    
    def test_validate_rejects_empty_file(self, file_handler):
        """Test validation rejects empty files."""
        file = FileStorage(
            stream=BytesIO(b''),
            filename='test.wav',
            content_type='audio/wav'
        )
        
        assert file_handler.validate_file(file) is False
    
    def test_validate_rejects_oversized_file(self, file_handler):
        """Test validation rejects files exceeding size limit."""
        # Create file larger than 50MB
        large_data = b'\x00' * (51 * 1024 * 1024)
        file = FileStorage(
            stream=BytesIO(large_data),
            filename='test.wav',
            content_type='audio/wav'
        )
        
        assert file_handler.validate_file(file) is False
    
    def test_validate_rejects_no_filename(self, file_handler):
        """Test validation rejects file with no filename."""
        file = FileStorage(
            stream=BytesIO(b'data'),
            filename='',
            content_type='audio/wav'
        )
        
        assert file_handler.validate_file(file) is False


@pytest.mark.unit
class TestFilenameSanitization:
    """Test filename sanitization functionality."""

    
    def test_sanitize_removes_path_traversal(self, file_handler):
        """Test sanitization removes path traversal sequences."""
        dangerous_names = [
            '../../../etc/passwd',
            '..\\..\\windows\\system32',
            'test/../../../file.wav',
            '....//....//file.wav'
        ]
        
        for name in dangerous_names:
            sanitized = file_handler.sanitize_filename(name)
            assert '..' not in sanitized
            assert '/' not in sanitized
            assert '\\' not in sanitized
    
    def test_sanitize_removes_absolute_paths(self, file_handler):
        """Test sanitization handles absolute paths."""
        sanitized = file_handler.sanitize_filename('/etc/passwd')
        assert not sanitized.startswith('/')
        
        sanitized = file_handler.sanitize_filename('C:\\Windows\\system32\\file.wav')
        assert '\\' not in sanitized
    
    def test_sanitize_preserves_valid_filename(self, file_handler):
        """Test sanitization preserves valid filenames."""
        valid_name = 'my_song_2024.wav'
        sanitized = file_handler.sanitize_filename(valid_name)
        assert sanitized == valid_name
    
    def test_sanitize_handles_empty_filename(self, file_handler):
        """Test sanitization provides default for empty filename."""
        sanitized = file_handler.sanitize_filename('')
        assert sanitized == 'unnamed_file'
    
    def test_sanitize_removes_leading_dots(self, file_handler):
        """Test sanitization removes leading dots (hidden files)."""
        sanitized = file_handler.sanitize_filename('...hidden.wav')
        assert not sanitized.startswith('.')
    
    def test_sanitize_limits_filename_length(self, file_handler):
        """Test sanitization limits filename length."""
        long_name = 'a' * 300 + '.wav'
        sanitized = file_handler.sanitize_filename(long_name)
        assert len(sanitized) <= 255
        assert sanitized.endswith('.wav')


@pytest.mark.unit
class TestAudioFileVerification:
    """Test audio file magic number verification."""
    
    def test_verify_valid_wav_file(self, file_handler, tmp_path):
        """Test verification accepts valid WAV file."""
        wav_file = tmp_path / "test.wav"
        wav_data = b'RIFF' + b'\x00' * 4 + b'WAVE' + b'\x00' * 100
        wav_file.write_bytes(wav_data)
        
        assert file_handler.verify_audio_file(str(wav_file)) is True

    
    def test_verify_valid_mp3_file(self, file_handler, tmp_path):
        """Test verification accepts valid MP3 file with ID3 tag."""
        mp3_file = tmp_path / "test.mp3"
        mp3_data = b'ID3' + b'\x00' * 100
        mp3_file.write_bytes(mp3_data)
        
        assert file_handler.verify_audio_file(str(mp3_file)) is True
    
    def test_verify_valid_mp3_file_mpeg_header(self, file_handler, tmp_path):
        """Test verification accepts valid MP3 file with MPEG header."""
        mp3_file = tmp_path / "test.mp3"
        mp3_data = b'\xff\xfb' + b'\x00' * 100
        mp3_file.write_bytes(mp3_data)
        
        assert file_handler.verify_audio_file(str(mp3_file)) is True
    
    def test_verify_valid_flac_file(self, file_handler, tmp_path):
        """Test verification accepts valid FLAC file."""
        flac_file = tmp_path / "test.flac"
        flac_data = b'fLaC' + b'\x00' * 100
        flac_file.write_bytes(flac_data)
        
        assert file_handler.verify_audio_file(str(flac_file)) is True
    
    def test_verify_rejects_mismatched_extension(self, file_handler, tmp_path):
        """Test verification rejects file with mismatched extension and content."""
        # WAV content with MP3 extension
        fake_mp3 = tmp_path / "fake.mp3"
        fake_mp3.write_bytes(b'RIFF' + b'\x00' * 4 + b'WAVE' + b'\x00' * 100)
        
        assert file_handler.verify_audio_file(str(fake_mp3)) is False
    
    def test_verify_rejects_invalid_content(self, file_handler, tmp_path):
        """Test verification rejects file with invalid content."""
        invalid_file = tmp_path / "invalid.wav"
        invalid_file.write_bytes(b'not audio data')
        
        assert file_handler.verify_audio_file(str(invalid_file)) is False
    
    def test_verify_rejects_empty_file(self, file_handler, tmp_path):
        """Test verification rejects empty file."""
        empty_file = tmp_path / "empty.wav"
        empty_file.write_bytes(b'')
        
        assert file_handler.verify_audio_file(str(empty_file)) is False
    
    def test_verify_handles_nonexistent_file(self, file_handler):
        """Test verification handles nonexistent file gracefully."""
        assert file_handler.verify_audio_file('/nonexistent/file.wav') is False


@pytest.mark.unit
class TestSaveUpload:
    """Test file upload saving functionality."""

    
    def test_save_upload_creates_job_directory(self, file_handler):
        """Test save_upload creates job-specific directory."""
        wav_data = b'RIFF' + b'\x00' * 4 + b'WAVE' + b'\x00' * 100
        file = FileStorage(
            stream=BytesIO(wav_data),
            filename='test.wav',
            content_type='audio/wav'
        )
        
        job_id = 'test-job-123'
        file_path, sanitized_name = file_handler.save_upload(file, job_id)
        
        assert Path(file_path).parent.name == job_id
        assert Path(file_path).exists()
    
    def test_save_upload_sanitizes_filename(self, file_handler):
        """Test save_upload sanitizes dangerous filenames."""
        wav_data = b'RIFF' + b'\x00' * 4 + b'WAVE' + b'\x00' * 100
        file = FileStorage(
            stream=BytesIO(wav_data),
            filename='../../../etc/passwd.wav',
            content_type='audio/wav'
        )
        
        job_id = 'test-job-456'
        file_path, sanitized_name = file_handler.save_upload(file, job_id)
        
        assert '..' not in sanitized_name
        assert '/' not in sanitized_name
        assert Path(file_path).exists()
    
    def test_save_upload_rejects_invalid_file(self, file_handler):
        """Test save_upload raises error for invalid file."""
        file = FileStorage(
            stream=BytesIO(b''),
            filename='empty.wav',
            content_type='audio/wav'
        )
        
        with pytest.raises(FileValidationError) as exc_info:
            file_handler.save_upload(file, 'test-job')
        
        assert exc_info.value.error_code == 'VALIDATION_FAILED'
    
    def test_save_upload_rejects_content_mismatch(self, file_handler):
        """Test save_upload rejects file with mismatched content."""
        # Text content with WAV extension
        file = FileStorage(
            stream=BytesIO(b'not audio data'),
            filename='fake.wav',
            content_type='audio/wav'
        )
        
        with pytest.raises(FileValidationError) as exc_info:
            file_handler.save_upload(file, 'test-job')
        
        assert exc_info.value.error_code == 'INVALID_CONTENT'


@pytest.mark.unit
class TestCleanupFile:
    """Test file cleanup functionality."""
    
    def test_cleanup_deletes_file(self, file_handler, tmp_path):
        """Test cleanup_file deletes the specified file."""
        test_file = tmp_path / "test.wav"
        test_file.write_bytes(b'test data')
        
        assert test_file.exists()
        file_handler.cleanup_file(str(test_file))
        assert not test_file.exists()

    
    def test_cleanup_removes_empty_directory(self, file_handler, temp_upload_dir):
        """Test cleanup_file removes empty parent directory."""
        job_dir = temp_upload_dir / "job-123"
        job_dir.mkdir()
        test_file = job_dir / "test.wav"
        test_file.write_bytes(b'test data')
        
        file_handler.cleanup_file(str(test_file))
        
        assert not test_file.exists()
        assert not job_dir.exists()
    
    def test_cleanup_preserves_nonempty_directory(self, file_handler, temp_upload_dir):
        """Test cleanup_file preserves directory with other files."""
        job_dir = temp_upload_dir / "job-456"
        job_dir.mkdir()
        test_file1 = job_dir / "test1.wav"
        test_file2 = job_dir / "test2.wav"
        test_file1.write_bytes(b'test data 1')
        test_file2.write_bytes(b'test data 2')
        
        file_handler.cleanup_file(str(test_file1))
        
        assert not test_file1.exists()
        assert test_file2.exists()
        assert job_dir.exists()
    
    def test_cleanup_handles_nonexistent_file(self, file_handler):
        """Test cleanup_file handles nonexistent file gracefully."""
        # Should not raise exception
        file_handler.cleanup_file('/nonexistent/file.wav')
    
    def test_cleanup_does_not_remove_upload_dir(self, file_handler, temp_upload_dir):
        """Test cleanup_file does not remove the main upload directory."""
        test_file = temp_upload_dir / "test.wav"
        test_file.write_bytes(b'test data')
        
        file_handler.cleanup_file(str(test_file))
        
        assert not test_file.exists()
        assert temp_upload_dir.exists()
