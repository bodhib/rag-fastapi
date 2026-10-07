"""
Recursive document chunking.

This module converts processed document pages into smaller chunks
while preserving document, knowledge-base, and page-level metadata.
"""

from dataclasses import dataclass

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import Settings, settings
from app.document_processor import ProcessedPage


@dataclass
class DocumentChunk:
    """
    Represent one chunk of a document.

    Attributes:
        doc_id: Unique identifier of the source document.
        text: Chunk text.
        source: Original document filename.
        page_number: One-based source page number.
        chunk_index: Zero-based index across the entire document.
        knowledge_base_id: Knowledge base containing the document.
    """

    doc_id: str
    text: str
    source: str
    page_number: int
    chunk_index: int
    knowledge_base_id: str = "kb_default"


class RecursiveChunker:
    """
    Split processed document pages into recursive text chunks.
    """

    def __init__(
        self,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        config: Settings | None = None,
    ) -> None:
        self.config = config or settings

        resolved_chunk_size = (
            chunk_size
            if chunk_size is not None
            else self.config.chunk_size
        )

        resolved_chunk_overlap = (
            chunk_overlap
            if chunk_overlap is not None
            else self.config.chunk_overlap
        )

        if resolved_chunk_size <= 0:
            raise ValueError(
                "chunk_size must be greater than zero"
            )

        if resolved_chunk_overlap < 0:
            raise ValueError(
                "chunk_overlap cannot be negative"
            )

        if resolved_chunk_overlap >= resolved_chunk_size:
            raise ValueError(
                "chunk_overlap must be smaller than chunk_size"
            )

        self.chunk_size = resolved_chunk_size
        self.chunk_overlap = resolved_chunk_overlap

        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=resolved_chunk_size,
            chunk_overlap=resolved_chunk_overlap,
            separators=[
                "\n\n",
                "\n",
                ". ",
                " ",
                "",
            ],
        )

    def chunk(
        self,
        doc_id: str,
        pages: list[ProcessedPage],
        knowledge_base_id: str,
    ) -> list[DocumentChunk]:
        """
        Split processed pages into document chunks.

        All generated chunks inherit the supplied knowledge base ID.
        """

        if not doc_id.strip():
            raise ValueError(
                "doc_id cannot be empty"
            )

        if not pages:
            raise ValueError(
                "Cannot create chunks from an empty page list"
            )

        if not knowledge_base_id.strip():
            raise ValueError(
                "knowledge_base_id cannot be empty"
            )

        chunks: list[DocumentChunk] = []

        chunk_index = 0

        for page in pages:
            page_chunks = self._splitter.split_text(
                page.text
            )

            for chunk_text in page_chunks:
                cleaned_chunk = chunk_text.strip()

                if not cleaned_chunk:
                    continue

                chunks.append(
                    DocumentChunk(
                        doc_id=doc_id,
                        text=cleaned_chunk,
                        source=page.source,
                        page_number=page.page_number,
                        chunk_index=chunk_index,
                        knowledge_base_id=knowledge_base_id,
                    )
                )

                chunk_index += 1

        if not chunks:
            raise ValueError(
                "No chunks could be created from the supplied pages"
            )

        return chunks