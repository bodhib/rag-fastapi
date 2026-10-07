import pytest

from app.chunker import DocumentChunk, RecursiveChunker
from app.config import Settings
from app.document_processor import ProcessedPage


def create_pages() -> list[ProcessedPage]:
    return [
        ProcessedPage(
            text=(
                "This is the first page. "
                "It contains information about "
                "retrieval augmented generation."
            ),
            source="document.pdf",
            page_number=1,
        ),
        ProcessedPage(
            text=(
                "This is the second page. "
                "It contains additional information."
            ),
            source="document.pdf",
            page_number=2,
        ),
    ]


def test_chunker_creates_chunks():
    chunker = RecursiveChunker(
        chunk_size=50,
        chunk_overlap=10,
    )

    chunks = chunker.chunk(
        doc_id="doc_001",
        pages=create_pages(),
        knowledge_base_id="kb_default",
    )

    assert chunks
    assert all(
        isinstance(chunk, DocumentChunk)
        for chunk in chunks
    )


def test_chunk_metadata_is_preserved():
    chunker = RecursiveChunker(
        chunk_size=100,
        chunk_overlap=10,
    )

    chunks = chunker.chunk(
        doc_id="doc_001",
        pages=[
            ProcessedPage(
                text="Information about the document.",
                source="policy.pdf",
                page_number=3,
            )
        ],
        knowledge_base_id="kb_company",
    )

    assert len(chunks) == 1

    chunk = chunks[0]

    assert chunk.doc_id == "doc_001"
    assert chunk.source == "policy.pdf"
    assert chunk.page_number == 3
    assert chunk.chunk_index == 0
    assert chunk.knowledge_base_id == "kb_company"


def test_chunk_indexes_are_sequential():
    chunker = RecursiveChunker(
        chunk_size=30,
        chunk_overlap=5,
    )

    chunks = chunker.chunk(
        doc_id="doc_001",
        pages=create_pages(),
        knowledge_base_id="kb_default",
    )

    indexes = [
        chunk.chunk_index
        for chunk in chunks
    ]

    assert indexes == list(range(len(chunks)))


def test_doc_id_is_preserved_on_every_chunk():
    chunker = RecursiveChunker(
        chunk_size=30,
        chunk_overlap=5,
    )

    chunks = chunker.chunk(
        doc_id="doc_123",
        pages=create_pages(),
        knowledge_base_id="kb_default",
    )

    assert all(
        chunk.doc_id == "doc_123"
        for chunk in chunks
    )


def test_knowledge_base_id_is_preserved_on_every_chunk():
    chunker = RecursiveChunker(
        chunk_size=30,
        chunk_overlap=5,
    )

    chunks = chunker.chunk(
        doc_id="doc_123",
        pages=create_pages(),
        knowledge_base_id="kb_company",
    )

    assert chunks

    assert all(
        chunk.knowledge_base_id == "kb_company"
        for chunk in chunks
    )


def test_empty_pages_are_rejected():
    chunker = RecursiveChunker()

    with pytest.raises(
        ValueError,
        match="empty page list",
    ):
        chunker.chunk(
            doc_id="doc_001",
            pages=[],
            knowledge_base_id="kb_default",
        )


def test_empty_doc_id_is_rejected():
    chunker = RecursiveChunker()

    with pytest.raises(
        ValueError,
        match="doc_id",
    ):
        chunker.chunk(
            doc_id="",
            pages=create_pages(),
            knowledge_base_id="kb_default",
        )


def test_whitespace_doc_id_is_rejected():
    chunker = RecursiveChunker()

    with pytest.raises(
        ValueError,
        match="doc_id",
    ):
        chunker.chunk(
            doc_id="   ",
            pages=create_pages(),
            knowledge_base_id="kb_default",
        )


def test_empty_knowledge_base_id_is_rejected():
    chunker = RecursiveChunker()

    with pytest.raises(
        ValueError,
        match="knowledge_base_id",
    ):
        chunker.chunk(
            doc_id="doc_001",
            pages=create_pages(),
            knowledge_base_id="",
        )


def test_whitespace_knowledge_base_id_is_rejected():
    chunker = RecursiveChunker()

    with pytest.raises(
        ValueError,
        match="knowledge_base_id",
    ):
        chunker.chunk(
            doc_id="doc_001",
            pages=create_pages(),
            knowledge_base_id="   ",
        )


def test_invalid_chunk_size_is_rejected():
    with pytest.raises(
        ValueError,
        match="chunk_size",
    ):
        RecursiveChunker(chunk_size=0)


def test_negative_overlap_is_rejected():
    with pytest.raises(
        ValueError,
        match="chunk_overlap",
    ):
        RecursiveChunker(
            chunk_size=100,
            chunk_overlap=-1,
        )


def test_overlap_equal_to_size_is_rejected():
    with pytest.raises(
        ValueError,
        match="chunk_overlap",
    ):
        RecursiveChunker(
            chunk_size=100,
            chunk_overlap=100,
        )


def test_chunker_uses_settings():
    config = Settings(
        chunk_size=500,
        chunk_overlap=100,
    )

    chunker = RecursiveChunker(
        config=config,
    )

    assert chunker.chunk_size == 500
    assert chunker.chunk_overlap == 100


def test_chunker_explicit_values_override_settings():
    config = Settings(
        chunk_size=500,
        chunk_overlap=100,
    )

    chunker = RecursiveChunker(
        chunk_size=800,
        chunk_overlap=150,
        config=config,
    )

    assert chunker.chunk_size == 800
    assert chunker.chunk_overlap == 150