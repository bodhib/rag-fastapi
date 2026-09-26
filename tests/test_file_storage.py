from io import BytesIO
from pathlib import Path

import pytest

from app.file_storage import UploadedFileStorage


def test_save_upload(
    tmp_path: Path,
) -> None:
    """Verify an uploaded file is saved correctly."""

    storage = UploadedFileStorage(
        storage_directory=tmp_path / "uploads"
    )

    source = BytesIO(
        b"Refund policy allows refunds within thirty days."
    )

    saved_path = storage.save(
        source_file=source,
        filename="policy.txt",
    )

    assert saved_path.exists()
    assert saved_path.name == "policy.txt"

    assert (
        saved_path.read_bytes()
        == b"Refund policy allows refunds within thirty days."
    )


def test_save_strips_directory_from_filename(
    tmp_path: Path,
) -> None:
    """Verify directory traversal is not preserved."""

    storage = UploadedFileStorage(
        storage_directory=tmp_path / "uploads"
    )

    source = BytesIO(b"safe content")

    saved_path = storage.save(
        source_file=source,
        filename="../../policy.txt",
    )

    assert saved_path.name == "policy.txt"
    assert saved_path.parent == tmp_path / "uploads"


def test_save_rejects_unsupported_extension(
    tmp_path: Path,
) -> None:
    """Verify unsupported file formats are rejected."""

    storage = UploadedFileStorage(
        storage_directory=tmp_path / "uploads"
    )

    source = BytesIO(b"some content")

    with pytest.raises(ValueError):
        storage.save(
            source_file=source,
            filename="malware.exe",
        )


def test_save_rejects_empty_file(
    tmp_path: Path,
) -> None:
    """Verify empty uploads are rejected."""

    storage = UploadedFileStorage(
        storage_directory=tmp_path / "uploads"
    )

    source = BytesIO(b"")

    with pytest.raises(ValueError):
        storage.save(
            source_file=source,
            filename="empty.txt",
        )