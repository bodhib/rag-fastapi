"""
Dense semantic retrieval.

This module combines query embedding generation with vector-store
search to retrieve semantically relevant document chunks.
"""

from dataclasses import dataclass

from app.embeddings import EmbeddingService
from app.vector_store import VectorStore


@dataclass
class RetrievalResult:
    """
    Represent one dense retrieval result.

    Attributes:
        text: Retrieved document chunk text.
        source: Original document filename.
        page_number: Original page number.
        chunk_index: Position of the chunk within the document.
        score: Similarity score where higher is better.
    """

    text: str
    source: str
    page_number: int
    chunk_index: int
    score: float


class DenseRetriever:
    """
    Retrieve document chunks using dense vector similarity.

    The retriever coordinates query embedding generation and vector
    database search while keeping those implementation details away
    from the API layer.
    """

    def __init__(
        self,
        embedding_service: EmbeddingService,
        vector_store: VectorStore,
    ) -> None:
        """
        Initialize the dense retriever.

        Args:
            embedding_service: Service used to embed user queries.
            vector_store: Vector store used for similarity search.
        """

        self.embedding_service = embedding_service
        self.vector_store = vector_store

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[RetrievalResult]:
        """
        Retrieve the most semantically similar document chunks.

        Args:
            query: Natural-language user question.
            top_k: Maximum number of results to retrieve.

        Returns:
            Dense retrieval results ordered by similarity.

        Raises:
            ValueError: If the query is empty or top_k is invalid.
        """

        if not query.strip():
            raise ValueError(
                "query cannot be empty"
            )

        if top_k <= 0:
            raise ValueError(
                "top_k must be greater than zero"
            )

        query_embedding = (
            self.embedding_service.embed_text(
                query
            )
        )

        raw_results = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k,
        )

        return [
            self._convert_result(result)
            for result in raw_results
        ]

    @staticmethod
    def _convert_result(
        result: dict,
    ) -> RetrievalResult:
        """
        Convert a vector-store result into a retrieval result.

        Args:
            result: Raw result returned by the vector store.

        Returns:
            Structured retrieval result.
        """

        metadata = result["metadata"]

        distance = float(
            result["distance"]
        )

        similarity_score = 1.0 - distance

        return RetrievalResult(
            text=result["text"],
            source=str(
                metadata["source"]
            ),
            page_number=int(
                metadata["page_number"]
            ),
            chunk_index=int(
                metadata["chunk_index"]
            ),
            score=similarity_score,
        )