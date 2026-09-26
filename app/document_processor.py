"""
Document loading and cleaning utilities.

This module is responsible for converting supported document formats
into clean, page-aware text that can later be passed to the chunking
and retrieval stages.
"""

from dataclasses import dataclass
from pathlib import Path
import re
import unicodedata

from docx import Document as DocxDocument
from pypdf import PdfReader


@dataclass
class ProcessedPage:
    """
    Represent one logical page of a processed document.

    Attributes:
        text: Cleaned text belonging to the page.
        source: Original document filename.
        page_number: One-based page number.
    """

    text: str
    source: str
    page_number: int


class DocumentProcessor:
    """Load and clean PDF, TXT, and DOCX documents."""

    SUPPORTED_EXTENSIONS = {
        ".pdf",
        ".txt",
        ".docx",
    }

    def load(self, file_path: Path) -> list[ProcessedPage]:
        """
        Load a document and return page-aware textual content.

        Args:
            file_path: Path to the document.

        Returns:
            A list of processed pages.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file is empty or unsupported.
        """

        self._validate_file(file_path)

        extension = file_path.suffix.lower()

        if extension == ".pdf":
            return self._load_pdf(file_path)

        if extension == ".txt":
            return self._load_txt(file_path)

        if extension == ".docx":
            return self._load_docx(file_path)

        raise ValueError(
            f"Unsupported file format: {extension}"
        )

    def clean_text(self, text: str) -> str:
        """
        Normalize and clean extracted document text.

        Cleaning includes:

        - Unicode normalization
        - Removal of non-printable characters
        - Newline normalization
        - Whitespace normalization
        - Removal of excessive blank lines

        Args:
            text: Raw document text.

        Returns:
            Cleaned text.
        """

        if not text:
            return ""

        # Normalize Unicode characters.
        text = unicodedata.normalize("NFKC", text)

        # Remove non-printable characters while preserving
        # newline and tab characters.
        text = "".join(
            character
            for character in text
            if character.isprintable()
            or character in {"\n", "\t"}
        )

        # Normalize Windows/Mac line endings.
        text = text.replace("\r\n", "\n")
        text = text.replace("\r", "\n")

        # Normalize spaces and tabs.
        text = re.sub(r"[ \t]+", " ", text)

        # Prevent excessive blank lines.
        text = re.sub(r"\n{3,}", "\n\n", text)

        return text.strip()

    def process(self, file_path: Path) -> list[ProcessedPage]:
        """
        Load and clean a document while preserving page metadata.

        Args:
            file_path: Path to the document.

        Returns:
            A list of processed pages containing cleaned text
            and source metadata.

        Raises:
            ValueError: If the document contains no usable text.
        """

        pages = self.load(file_path)

        processed_pages = []

        for page in pages:
            cleaned_text = self.clean_text(page.text)

            if not cleaned_text:
                continue

            processed_pages.append(
                ProcessedPage(
                    text=cleaned_text,
                    source=page.source,
                    page_number=page.page_number,
                )
            )

        if not processed_pages:
            raise ValueError(
                "Document contains no extractable text"
            )

        return processed_pages

    def _validate_file(self, file_path: Path) -> None:
        """
        Validate that the supplied file exists and is supported.

        Args:
            file_path: Path to the document.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file is empty or unsupported.
        """

        if not file_path.exists():
            raise FileNotFoundError(
                f"Document not found: {file_path}"
            )

        if not file_path.is_file():
            raise ValueError(
                f"Path is not a file: {file_path}"
            )

        if file_path.stat().st_size == 0:
            raise ValueError(
                "Document file is empty"
            )

        if file_path.suffix.lower() not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported file format: {file_path.suffix}"
            )

    def _load_txt(self, file_path: Path) -> list[ProcessedPage]:
        """
        Load a text file.

        A TXT file does not have native pages, so the complete
        document is treated as one logical page.

        Args:
            file_path: Path to the text file.

        Returns:
            A list containing one logical page.
        """

        raw_bytes = file_path.read_bytes()

        encodings = (
            "utf-8",
            "utf-8-sig",
            "cp1252",
            "latin-1",
        )

        for encoding in encodings:
            try:
                text = raw_bytes.decode(encoding)

                return [
                    ProcessedPage(
                        text=text,
                        source=file_path.name,
                        page_number=1,
                    )
                ]

            except UnicodeDecodeError:
                continue

        raise ValueError(
            "Unable to decode text file using supported encodings"
        )

    def _load_pdf(self, file_path: Path) -> list[ProcessedPage]:
        """
        Extract text from a PDF while preserving page numbers.

        Args:
            file_path: Path to the PDF.

        Returns:
            A list of pages with their original page numbers.
        """

        reader = PdfReader(str(file_path))

        pages = []

        for page_number, page in enumerate(
            reader.pages,
            start=1,
        ):
            text = page.extract_text()

            if text:
                pages.append(
                    ProcessedPage(
                        text=text,
                        source=file_path.name,
                        page_number=page_number,
                    )
                )

        return pages

    def _load_docx(self, file_path: Path) -> list[ProcessedPage]:
        """
        Extract paragraph text from a DOCX document.

        DOCX files do not expose reliable PDF-like page boundaries
        through python-docx, so the document is treated as one
        logical page.

        Args:
            file_path: Path to the DOCX document.

        Returns:
            A list containing one logical page.
        """

        document = DocxDocument(str(file_path))

        paragraphs = []

        for paragraph in document.paragraphs:
            if paragraph.text.strip():
                paragraphs.append(paragraph.text)

        text = "\n".join(paragraphs)

        return [
            ProcessedPage(
                text=text,
                source=file_path.name,
                page_number=1,
            )
        ]