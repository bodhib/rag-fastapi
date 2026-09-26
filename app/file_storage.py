from pathlib import Path
from typing import BinaryIO


class UploadedFileStorage:
    SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".docx"}

    def __init__(self, storage_directory: Path) -> None:
        self.storage_directory = storage_directory
        self.storage_directory.mkdir(parents=True, exist_ok=True)

    def save(
        self,
        source_file: BinaryIO,
        filename: str,
    ) -> Path:
        safe_filename = Path(filename).name

        if not safe_filename:
            raise ValueError("Uploaded file must have a filename")

        extension = Path(safe_filename).suffix.lower()

        if extension not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported file format: {extension}"
            )

        destination = self.storage_directory / safe_filename

        resolved_directory = self.storage_directory.resolve()
        resolved_destination = destination.resolve()

        if not self._is_within_directory(
            resolved_destination,
            resolved_directory,
        ):
            raise ValueError(
                "Uploaded file path escapes the storage directory"
            )

        content = source_file.read()

        if not content:
            raise ValueError("Uploaded file is empty")

        destination.write_bytes(content)

        return destination

    @staticmethod
    def _is_within_directory(
        path: Path,
        directory: Path,
    ) -> bool:
        try:
            path.relative_to(directory)
            return True
        except ValueError:
            return False