from pathlib import Path

import pytest

from app.document_processor import DocumentProcessor

from reportlab.pdfgen import canvas


def test_load_txt_file(tmp_path: Path) -> None:
    """Verify that a TXT document can be loaded."""

    file_path = tmp_path / "sample.txt"

    file_path.write_text(
        "This is a sample document.",
        encoding="utf-8",
    )

    processor = DocumentProcessor()

    result = processor.load(file_path)

    assert len(result) == 1
    assert result[0].text == "This is a sample document."
    assert result[0].source == "sample.txt"
    assert result[0].page_number == 1


def test_empty_file_is_rejected(tmp_path: Path) -> None:
    """Verify that empty files are rejected."""

    file_path = tmp_path / "empty.txt"

    file_path.write_bytes(b"")

    processor = DocumentProcessor()

    with pytest.raises(ValueError, match="empty"):
        processor.load(file_path)


def test_missing_file_is_rejected(tmp_path: Path) -> None:
    """Verify that missing files are rejected."""

    file_path = tmp_path / "missing.txt"

    processor = DocumentProcessor()

    with pytest.raises(FileNotFoundError):
        processor.load(file_path)


def test_unsupported_file_is_rejected(tmp_path: Path) -> None:
    """Verify that unsupported file formats are rejected."""

    file_path = tmp_path / "sample.csv"

    file_path.write_text(
        "name,price",
        encoding="utf-8",
    )

    processor = DocumentProcessor()

    with pytest.raises(ValueError, match="Unsupported"):
        processor.load(file_path)


def test_clean_text_normalizes_whitespace() -> None:
    """Verify whitespace normalization."""

    processor = DocumentProcessor()

    raw_text = (
        "Hello     world.\n"
        "\n"
        "\n"
        "\n"
        "This   is   another line."
    )

    result = processor.clean_text(raw_text)

    assert result == (
        "Hello world.\n\n"
        "This is another line."
    )


def test_clean_text_normalizes_unicode() -> None:
    """Verify Unicode normalization."""

    processor = DocumentProcessor()

    raw_text = "Cafe\u0301"

    result = processor.clean_text(raw_text)

    assert result == "Café"


def test_clean_text_empty_string() -> None:
    """Verify empty input produces empty output."""

    processor = DocumentProcessor()

    assert processor.clean_text("") == ""


def test_process_returns_cleaned_page(
    tmp_path: Path,
) -> None:
    """Verify process() cleans text and preserves metadata."""

    file_path = tmp_path / "sample.txt"

    file_path.write_text(
        "Hello     world.\n\n\nThis   is   a test.",
        encoding="utf-8",
    )

    processor = DocumentProcessor()

    result = processor.process(file_path)

    assert len(result) == 1

    assert result[0].text == (
        "Hello world.\n\n"
        "This is a test."
    )

    assert result[0].source == "sample.txt"
    assert result[0].page_number == 1


def test_load_pdf_file(tmp_path: Path) -> None: 
    """Verify that PDF text can be extracted.""" 
    
    file_path = tmp_path / "sample.pdf" 
    
    pdf = canvas.Canvas(str(file_path)) 
    
    pdf.drawString( 100, 750, 
                   "This is text inside the PDF.", 
                   ) 
    pdf.save() 
    processor = DocumentProcessor() 
    result = processor.load(file_path) 
    assert "This is text inside the PDF." in result[0].text