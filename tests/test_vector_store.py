from pathlib import Path

import pytest

from app.chunker import DocumentChunk
from app.embeddings import EmbeddedChunk
from app.vector_store import ChromaVectorStore


def create_embedded_chunk(
    doc_id: str,    
    text: str,
    source: str,
    page_number: int,
    chunk_index: int,
    embedding: list[float],
) -> EmbeddedChunk:
    """Create a test embedded chunk."""

    chunk = DocumentChunk(
        doc_id=doc_id,
        text=text,
        source=source,
        page_number=page_number,
        chunk_index=chunk_index,
    )

    return EmbeddedChunk(
        chunk=chunk,
        embedding=embedding,
    )


def test_empty_collection(
    tmp_path: Path,
) -> None:
    """Verify a newly created collection is empty."""

    store = ChromaVectorStore(
        persist_directory=tmp_path / "chroma"
    )

    assert store.count == 0


def test_add_chunks(
    tmp_path: Path,
) -> None:
    """Verify chunks can be stored."""

    store = ChromaVectorStore(
        persist_directory=tmp_path / "chroma"
    )

    chunks = [
        create_embedded_chunk(
            doc_id="doc_001",
            text="Refunds are available within 30 days.",
            source="policy.pdf",
            page_number=4,
            chunk_index=0,
            embedding=[1.0, 0.0, 0.0],
        ),
        create_embedded_chunk(
            doc_id="doc_002",
            text="Shipping takes five business days.",
            source="shipping.pdf",
            page_number=2,
            chunk_index=1,
            embedding=[0.0, 1.0, 0.0],
        ),
    ]

    store.add_chunks(chunks)

    assert store.count == 2


def test_search_returns_similar_chunk(
    tmp_path: Path,
) -> None:
    """Verify vector similarity search works."""

    store = ChromaVectorStore(
        persist_directory=tmp_path / "chroma"
    )

    chunks = [
        create_embedded_chunk(
            doc_id="doc_001",
            text="Refunds are available within 30 days.",
            source="policy.pdf",
            page_number=4,
            chunk_index=0,
            embedding=[1.0, 0.0, 0.0],
        ),
        create_embedded_chunk(
            doc_id="doc_002",
            text="Shipping takes five business days.",
            source="shipping.pdf",
            page_number=2,
            chunk_index=1,
            embedding=[0.0, 1.0, 0.0],
        ),
    ]

    store.add_chunks(chunks)

    results = store.search(
        query_embedding=[0.99, 0.01, 0.0],
        top_k=1,
    )

    assert len(results) == 1

    assert (
        results[0]["text"]
        == "Refunds are available within 30 days."
    )

    assert results[0]["metadata"]["source"] == "policy.pdf"
    assert results[0]["metadata"]["page_number"] == 4
    assert results[0]["metadata"]["doc_id"] == "doc_001"


def test_search_empty_store(
    tmp_path: Path,
) -> None:
    """Verify searching an empty store returns no results."""

    store = ChromaVectorStore(
        persist_directory=tmp_path / "chroma"
    )

    results = store.search(
        query_embedding=[1.0, 0.0, 0.0],
        top_k=5,
    )

    assert results == []


def test_empty_chunks_are_rejected(
    tmp_path: Path,
) -> None:
    """Verify empty chunk insertion is rejected."""

    store = ChromaVectorStore(
        persist_directory=tmp_path / "chroma"
    )

    with pytest.raises(
        ValueError,
        match="empty chunk list",
    ):
        store.add_chunks([])


def test_invalid_top_k_is_rejected(
    tmp_path: Path,
) -> None:
    """Verify invalid top_k is rejected."""

    store = ChromaVectorStore(
        persist_directory=tmp_path / "chroma"
    )

    with pytest.raises(
        ValueError,
        match="top_k",
    ):
        store.search(
            query_embedding=[1.0, 0.0, 0.0],
            top_k=0,
        )


def test_delete_source(
    tmp_path: Path,
) -> None:
    """Verify all chunks for a source can be deleted."""

    store = ChromaVectorStore(
        persist_directory=tmp_path / "chroma"
    )

    chunks = [
        create_embedded_chunk(
            doc_id="doc_001",
            text="First policy chunk.",
            source="policy.pdf",
            page_number=1,
            chunk_index=0,
            embedding=[1.0, 0.0, 0.0],
        ),
        create_embedded_chunk(
            doc_id="doc_001",
            text="Second policy chunk.",
            source="policy.pdf",
            page_number=2,
            chunk_index=1,
            embedding=[0.9, 0.1, 0.0],
        ),
        create_embedded_chunk(
            doc_id="doc_002",
            text="Shipping information.",
            source="shipping.pdf",
            page_number=1,
            chunk_index=2,
            embedding=[0.0, 1.0, 0.0],
        ),
    ]

    store.add_chunks(chunks)

    assert store.count == 3

    store.delete_source("policy.pdf")

    assert store.count == 1

def test_same_filename_with_different_doc_ids_is_stored_separately(
    tmp_path: Path,
) -> None:
    """Verify doc_id prevents filename-based chunk collisions."""

    store = ChromaVectorStore(
        persist_directory=tmp_path / "chroma"
    )

    chunks = [
        create_embedded_chunk(
            doc_id="doc_001",
            text="First document content.",
            source="policy.pdf",
            page_number=1,
            chunk_index=0,
            embedding=[1.0, 0.0, 0.0],
        ),
        create_embedded_chunk(
            doc_id="doc_002",
            text="Second document content.",
            source="policy.pdf",
            page_number=1,
            chunk_index=0,
            embedding=[0.0, 1.0, 0.0],
        ),
    ]

    store.add_chunks(chunks)

    assert store.count == 2

def test_delete_document_removes_only_that_document(
    tmp_path: Path,
) -> None:
    """Verify deletion by doc_id removes only one document."""

    store = ChromaVectorStore(
        persist_directory=tmp_path / "chroma"
    )

    chunks = [
        create_embedded_chunk(
            doc_id="doc_001",
            text="First policy chunk.",
            source="policy.pdf",
            page_number=1,
            chunk_index=0,
            embedding=[1.0, 0.0, 0.0],
        ),
        create_embedded_chunk(
            doc_id="doc_001",
            text="Second policy chunk.",
            source="policy.pdf",
            page_number=2,
            chunk_index=1,
            embedding=[0.9, 0.1, 0.0],
        ),
        create_embedded_chunk(
            doc_id="doc_002",
            text="Another document.",
            source="manual.pdf",
            page_number=1,
            chunk_index=0,
            embedding=[0.0, 1.0, 0.0],
        ),
    ]

    store.add_chunks(chunks)

    assert store.count == 3

    store.delete_document("doc_001")

    assert store.count == 1

    results = store.search(
        query_embedding=[0.0, 1.0, 0.0],
        top_k=5,
    )

    assert len(results) == 1
    assert results[0]["metadata"]["doc_id"] == "doc_002" 

def test_delete_document_rejects_empty_doc_id(
    tmp_path: Path,
) -> None:
    """Verify empty document IDs are rejected."""

    store = ChromaVectorStore(
        persist_directory=tmp_path / "chroma"
    )

    with pytest.raises(
        ValueError,
        match="doc_id",
    ):
        store.delete_document("")

def test_get_all_chunks_returns_persisted_chunks(
    tmp_path: Path,
) -> None:
    """Verify all stored chunks can be reconstructed."""

    store = ChromaVectorStore(
        persist_directory=tmp_path / "chroma"
    )

    chunks = [
        create_embedded_chunk(
            doc_id="doc_001",
            text="First document chunk.",
            source="policy.pdf",
            page_number=1,
            chunk_index=0,
            embedding=[1.0, 0.0, 0.0],
        ),
        create_embedded_chunk(
            doc_id="doc_001",
            text="Second document chunk.",
            source="policy.pdf",
            page_number=2,
            chunk_index=1,
            embedding=[0.9, 0.1, 0.0],
        ),
        create_embedded_chunk(
            doc_id="doc_002",
            text="Another document.",
            source="manual.pdf",
            page_number=1,
            chunk_index=0,
            embedding=[0.0, 1.0, 0.0],
        ),
    ]

    store.add_chunks(chunks)

    result = store.get_all_chunks()

    assert len(result) == 3

    assert result[0].doc_id == "doc_001"
    assert result[0].text == "First document chunk."
    assert result[0].source == "policy.pdf"
    assert result[0].page_number == 1
    assert result[0].chunk_index == 0

    assert result[1].doc_id == "doc_001"
    assert result[1].chunk_index == 1

    assert result[2].doc_id == "doc_002"
    assert result[2].chunk_index == 0        


def test_get_all_chunks_returns_empty_list_for_empty_store(
    tmp_path: Path,
) -> None:
    """Verify an empty Chroma store returns no chunks."""

    store = ChromaVectorStore(
        persist_directory=tmp_path / "chroma"
    )

    assert store.get_all_chunks() == []              