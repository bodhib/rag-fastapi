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
    doc_id: str = ""


class DenseRetriever:
    """
    Retrieve document chunks using dense vector similarity.
    """

    def __init__(
        self,
        embedding_service: EmbeddingService,
        vector_store: VectorStore,
    ) -> None:
        self.embedding_service = embedding_service
        self.vector_store = vector_store

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        knowledge_base_id: str = "kb_default",
        allowed_doc_ids: set[str] | None = None,
    ) -> list[RetrievalResult]:
            if not query.strip():
                raise ValueError(
                    "query cannot be empty"
                )

            if top_k <= 0:
                raise ValueError(
                    "top_k must be greater than zero"
                )

            if (
                knowledge_base_id is not None
                and not knowledge_base_id.strip()
            ):
                raise ValueError(
                    "knowledge_base_id cannot be empty"
                )

            if allowed_doc_ids is not None:
                allowed_doc_ids = {
                    doc_id.strip()
                    for doc_id in allowed_doc_ids
                    if doc_id and doc_id.strip()
                }

                if not allowed_doc_ids:
                    return []

            query_embedding = (
                self.embedding_service.embed_text(
                    query
                )
            )

            raw_results = self.vector_store.search(
                query_embedding=query_embedding,
                top_k=top_k,
                knowledge_base_id=knowledge_base_id,
                allowed_doc_ids=allowed_doc_ids,
            )

            return [
                self._convert_result(result)
                for result in raw_results
            ]
    
    @staticmethod
    def _convert_result(
        result: dict,
    ) -> RetrievalResult:
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
            doc_id=metadata["doc_id"],
        )