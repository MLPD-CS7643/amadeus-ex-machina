"""
FileHandler class for secure file operations.

Provides validation, sanitization, and cleanup of uploaded audio files.
"""
import os
from pathlib import Path
from typing import Tuple
from werkzeug.datastructures import FileStorage
from werkzeug.utils import secure_filename

# Canonical definition lives in app.errors; re-exported here for backward
# compatibility (existing tests import FileValidationError from this module).
from app.errors import FileValidationError

__all__ = ["FileHandler", "FileValidationError"]


class FileHandler:
    """
    Handles secure file operations for uploaded audio files.
    
    Provides validation, sanitization, and cleanup functionality to ensure
    secure handling of user-uploaded files.
    """
    
    # Magic number signatures for audio file formats
    AUDIO_SIGNATURES = {
        '.wav': [b'RIFF', b'WAVE'],
        '.mp3': [b'ID3', b'\xff\xfb', b'\xff\xf3', b'\xff\xf2'],
        '.flac': [b'fLaC']
    }
    
    def __init__(self, upload_dir: Path, max_file_size: int, allowed_extensions: set):
        """
        Initialize FileHandler with configuration.
        
        Args:
            upload_dir: Directory for storing uploaded files
            max_file_size: Maximum file size in bytes
            allowed_extensions: Set of allowed file extensions (e.g., {'.wav', '.mp3', '.flac'})
        """
        self.upload_dir = upload_dir
        self.max_file_size = max_file_size
        self.allowed_extensions = allowed_extensions
        
        # Ensure upload directory exists
        self.upload_dir.mkdir(parents=True, exist_ok=True)
    
    def save_upload(self, file: FileStorage, job_id: str) -> Tuple[str, str]:
        """
        Save uploaded file with sanitized name.
        
        Args:
            file: Uploaded file from Flask request
            job_id: Unique job identifier for organizing files
            
        Returns:
            Tuple of (file_path, sanitized_filename)
            
        Raises:
            FileValidationError: If file validation fails
        """
        # Validate file first
        if not self.validate_file(file):
            raise FileValidationError(
                "File validation failed",
                "VALIDATION_FAILED"
            )
        
        # Sanitize filename
        sanitized_name = self.sanitize_filename(file.filename)
        
        # Create job-specific directory
        job_dir = self.upload_dir / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        
        # Save file
        file_path = job_dir / sanitized_name
        file.save(str(file_path))
        
        # Verify audio file after saving
        if not self.verify_audio_file(str(file_path)):
            # Clean up invalid file
            self.cleanup_file(str(file_path))
            raise FileValidationError(
                "File content does not match declared format",
                "INVALID_CONTENT"
            )
        
        return str(file_path), sanitized_name
    
    def validate_file(self, file: FileStorage) -> bool:
        """
        Validate file extension and size.
        
        Args:
            file: Uploaded file from Flask request
            
        Returns:
            True if file is valid, False otherwise
        """
        # Check if file exists
        if not file or not file.filename:
            return False
        
        # Check file extension
        file_ext = Path(file.filename).suffix.lower()
        if file_ext not in self.allowed_extensions:
            return False
        
        # Check file size by seeking to end
        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        file.seek(0)  # Reset to beginning
        
        if file_size == 0:
            return False
        
        if file_size > self.max_file_size:
            return False
        
        return True
    
    def sanitize_filename(self, filename: str) -> str:
        """
        Remove path traversal sequences and dangerous characters from filename.
        
        Args:
            filename: Original filename from upload
            
        Returns:
            Sanitized filename safe for filesystem operations
        """
        # Use werkzeug's secure_filename as base
        safe_name = secure_filename(filename)
        
        # Additional sanitization for path traversal
        # Remove any remaining path separators
        safe_name = safe_name.replace('/', '_').replace('\\', '_')
        
        # Remove parent directory references
        safe_name = safe_name.replace('..', '')
        
        # Remove leading dots (hidden files)
        safe_name = safe_name.lstrip('.')
        
        # Ensure filename is not empty after sanitization
        if not safe_name:
            safe_name = 'unnamed_file'
        
        # Limit filename length (keep extension)
        max_length = 255
        if len(safe_name) > max_length:
            name_part = Path(safe_name).stem
            ext_part = Path(safe_name).suffix
            name_part = name_part[:max_length - len(ext_part)]
            safe_name = name_part + ext_part
        
        return safe_name
    
    def verify_audio_file(self, file_path: str) -> bool:
        """
        Verify file is valid audio using magic number verification.
        
        Checks that file content matches the declared file extension by
        examining the file's magic number (header bytes).
        
        Args:
            file_path: Path to file to verify
            
        Returns:
            True if file content matches extension, False otherwise
        """
        file_ext = Path(file_path).suffix.lower()
        
        # Check if extension is in our allowed list
        if file_ext not in self.AUDIO_SIGNATURES:
            return False
        
        try:
            # Read first 12 bytes for magic number check
            with open(file_path, 'rb') as f:
                header = f.read(12)
            
            if not header:
                return False
            
            # Check against known signatures for this extension
            signatures = self.AUDIO_SIGNATURES[file_ext]
            
            for signature in signatures:
                if file_ext == '.wav':
                    # WAV files have RIFF at start and WAVE at offset 8
                    if header.startswith(b'RIFF') and b'WAVE' in header:
                        return True
                elif header.startswith(signature):
                    return True
            
            return False
            
        except (IOError, OSError):
            return False
    
    def cleanup_file(self, file_path: str) -> None:
        """
        Securely delete file and its parent directory if empty.
        
        Args:
            file_path: Path to file to delete
        """
        try:
            path = Path(file_path)
            
            # Delete the file
            if path.exists() and path.is_file():
                path.unlink()
            
            # Try to remove parent directory if empty
            parent_dir = path.parent
            if parent_dir.exists() and parent_dir != self.upload_dir:
                try:
                    parent_dir.rmdir()  # Only removes if empty
                except OSError:
                    # Directory not empty, that's fine
                    pass
                    
        except (IOError, OSError) as e:
            # Log error but don't raise - cleanup is best effort
            pass
