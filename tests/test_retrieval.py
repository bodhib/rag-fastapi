import pytest

from app.embeddings import EmbeddingService
from app.retrieval import DenseRetriever
from app.vector_store import ChromaVectorStore

from pathlib import Path
from app.chunker import DocumentChunk
from app.embeddings import EmbeddedChunk


class FakeEmbeddingService:
    """Fake embedding service for retrieval unit tests."""

    def embed_text(
        self,
        text: str,
    ) -> list[float]:
        """Return a deterministic query embedding."""

        return [1.0, 0.0, 0.0]


class FakeVectorStore:
    """Fake vector store for retrieval unit tests."""

    def search(
        self,
        query_embedding: list[float],
        top_k: int,
    ) -> list[dict]:
        """Return deterministic vector-search results."""

        assert query_embedding == [
            1.0,
            0.0,
            0.0,
        ]

        return [
            {
                "text": (
                    "Refunds are available "
                    "within 30 days."
                ),
                "metadata": {
                    "source": "policy.pdf",
                    "page_number": 4,
                    "chunk_index": 17,
                },
                "distance": 0.1,
            },
            {
                "text": (
                    "Refund requests require "
                    "an order number."
                ),
                "metadata": {
                    "source": "policy.pdf",
                    "page_number": 4,
                    "chunk_index": 18,
                },
                "distance": 0.2,
            },
        ]


def create_retriever() -> DenseRetriever:
    """Create a retriever using fake dependencies."""

    return DenseRetriever(
        embedding_service=FakeEmbeddingService(),
        vector_store=FakeVectorStore(),
    )


def test_retrieve_returns_results() -> None:
    """Verify dense retrieval returns structured results."""

    retriever = create_retriever()

    results = retriever.retrieve(
        "What is the refund period?"
    )

    assert len(results) == 2

    assert (
        results[0].text
        == "Refunds are available within 30 days."
    )

    assert results[0].source == "policy.pdf"
    assert results[0].page_number == 4
    assert results[0].chunk_index == 17


def test_similarity_score_is_calculated() -> None:
    """Verify distance is converted to similarity."""

    retriever = create_retriever()

    results = retriever.retrieve(
        "What is the refund period?"
    )

    assert results[0].score == pytest.approx(0.9)
    assert results[1].score == pytest.approx(0.8)


def test_empty_query_is_rejected() -> None:
    """Verify empty queries are rejected."""

    retriever = create_retriever()

    with pytest.raises(
        ValueError,
        match="query cannot be empty",
    ):
        retriever.retrieve("")


def test_whitespace_query_is_rejected() -> None:
    """Verify whitespace-only queries are rejected."""

    retriever = create_retriever()

    with pytest.raises(
        ValueError,
        match="query cannot be empty",
    ):
        retriever.retrieve("   ")


def test_invalid_top_k_is_rejected() -> None:
    """Verify invalid top_k is rejected."""

    retriever = create_retriever()

    with pytest.raises(
        ValueError,
        match="top_k",
    ):
        retriever.retrieve(
            "What is the refund period?",
            top_k=0,
        )

@pytest.mark.integration
def test_real_dense_retrieval(
    tmp_path: Path,
) -> None:
    """Verify real embeddings and Chroma work together."""

    embedding_service = EmbeddingService()

    vector_store = ChromaVectorStore(
        persist_directory=tmp_path / "chroma"
    )

    chunks = [
        DocumentChunk(
            doc_id="doc_001",
            text=(
                "Customers can request a refund "
                "within 30 days of purchase."
            ),
            source="policy.pdf",
            page_number=4,
            chunk_index=0,
        ),
        DocumentChunk(
            doc_id="doc_002",
            text=(
                "The office is open from "
                "nine in the morning until five."
            ),
            source="office.pdf",
            page_number=2,
            chunk_index=1,
        ),
    ]

    # Generate embeddings for the DocumentChunk objects.
    embedded_chunks = embedding_service.embed_chunks(
        chunks
    )

    # Store the embedded chunks in Chroma.
    vector_store.add_chunks(
        embedded_chunks
    )

    retriever = DenseRetriever(
        embedding_service=embedding_service,
        vector_store=vector_store,
    )

    results = retriever.retrieve(
        "How long do I have to get a refund?",
        top_k=1,
    )

    assert len(results) == 1
    assert results[0].source == "policy.pdf"
    assert results[0].page_number == 4
    assert "refund" in results[0].text.lower()     