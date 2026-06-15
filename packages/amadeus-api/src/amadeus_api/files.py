"""
FileHandler: secure handling of uploaded audio files.

Validates extension + size, sanitizes the filename, verifies the magic bytes match the
declared format, and cleans up afterwards. Framework-neutral: it works on a binary
stream + filename (FastAPI passes ``UploadFile.file`` / ``UploadFile.filename``), so it
carries no werkzeug dependency.
"""

import os
import re
import shutil
from pathlib import Path
from typing import BinaryIO

from amadeus_api.errors import OVERSIZE_CODE, FileValidationError

_SAFE_CHARS = re.compile(r"[^A-Za-z0-9._-]+")


def secure_filename(filename: str) -> str:
    """
    Reduce an arbitrary filename to a safe basename.

    Strips directory components and path-traversal sequences, replaces unsafe
    characters with underscores, and guarantees a non-empty result.
    """
    name = Path(filename).name  # drop any directory components
    name = name.replace("..", "")
    name = _SAFE_CHARS.sub("_", name).strip("._")
    if not name:
        name = "unnamed_file"
    if len(name) > 255:
        stem = Path(name).stem[: 255 - len(Path(name).suffix)]
        name = stem + Path(name).suffix
    return name


class FileHandler:
    # Magic-number signatures for supported audio formats.
    AUDIO_SIGNATURES = {
        ".wav": [b"RIFF", b"WAVE"],
        ".mp3": [b"ID3", b"\xff\xfb", b"\xff\xf3", b"\xff\xf2"],
        ".flac": [b"fLaC"],
    }

    def __init__(self, upload_dir: Path, max_file_size: int, allowed_extensions: set):
        self.upload_dir = Path(upload_dir)
        self.max_file_size = max_file_size
        self.allowed_extensions = allowed_extensions
        self.upload_dir.mkdir(parents=True, exist_ok=True)

    def validate(self, filename: str, file_obj: BinaryIO) -> None:
        """Validate extension + size, raising :class:`FileValidationError` on failure."""
        if not filename:
            raise FileValidationError("No file was provided.", "NO_FILE")

        ext = Path(filename).suffix.lower()
        if ext not in self.allowed_extensions:
            allowed = ", ".join(sorted(self.allowed_extensions))
            raise FileValidationError(
                f"Unsupported file format. Supported formats: {allowed}.",
                "INVALID_FORMAT",
            )

        file_obj.seek(0, os.SEEK_END)
        size = file_obj.tell()
        file_obj.seek(0)
        if size == 0:
            raise FileValidationError("Uploaded file is empty.", "EMPTY_FILE")
        if size > self.max_file_size:
            raise FileValidationError(
                "File exceeds the maximum allowed size.", OVERSIZE_CODE
            )

    def save_upload(
        self, file_obj: BinaryIO, filename: str, job_id: str
    ) -> tuple[str, str]:
        """
        Validate + persist an uploaded audio stream under a job-specific directory.

        Returns ``(file_path, sanitized_filename)``. Raises ``FileValidationError`` if
        validation fails or the content does not match the declared format.
        """
        self.validate(filename, file_obj)

        sanitized_name = secure_filename(filename)
        job_dir = self.upload_dir / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        file_path = job_dir / sanitized_name

        file_obj.seek(0)
        with open(file_path, "wb") as out:
            shutil.copyfileobj(file_obj, out)

        if not self.verify_audio_file(str(file_path)):
            self.cleanup_file(str(file_path))
            raise FileValidationError(
                "File content does not match declared format.", "INVALID_CONTENT"
            )
        return str(file_path), sanitized_name

    def verify_audio_file(self, file_path: str) -> bool:
        """Verify the file's magic bytes match its declared extension."""
        ext = Path(file_path).suffix.lower()
        if ext not in self.AUDIO_SIGNATURES:
            return False
        try:
            with open(file_path, "rb") as f:
                header = f.read(12)
        except OSError:
            return False
        if not header:
            return False
        for signature in self.AUDIO_SIGNATURES[ext]:
            if ext == ".wav":
                if header.startswith(b"RIFF") and b"WAVE" in header:
                    return True
            elif header.startswith(signature):
                return True
        return False

    def cleanup_file(self, file_path: str) -> None:
        """Best-effort delete of a file and its (now-empty) parent directory."""
        try:
            path = Path(file_path)
            if path.exists() and path.is_file():
                path.unlink()
            parent_dir = path.parent
            if parent_dir.exists() and parent_dir != self.upload_dir:
                try:
                    parent_dir.rmdir()  # only if empty
                except OSError:
                    pass
        except OSError:
            pass
