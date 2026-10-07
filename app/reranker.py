"""
Cross-encoder reranking.

This module reranks candidate documents produced by the hybrid
retrieval stage using a cross-encoder model.
"""

from dataclasses import dataclass

from sentence_transformers import CrossEncoder

from app.hybrid_retriever import HybridResult

from app.config import Settings, settings


@dataclass
class RerankedResult:
    """
    Represent a document after cross-encoder reranking.

    Attributes:
        text: Retrieved document chunk text.
        source: Original document filename.
        page_number: Original page number.
        chunk_index: Position of the chunk within the document.
        dense_score: Normalized dense retrieval score.
        bm25_score: Normalized BM25 score.
        hybrid_score: Score produced by hybrid retrieval.
        rerank_score: Score produced by the cross-encoder.
    """

    text: str
    source: str
    page_number: int
    chunk_index: int
    dense_score: float
    bm25_score: float
    hybrid_score: float
    rerank_score: float
    doc_id: str = ""


class CrossEncoderReranker:
    """
    Rerank retrieved candidates using a cross-encoder model.

    The reranker receives a relatively small candidate set from the
    retrieval layer and applies a more expensive relevance model to
    determine the final ordering.
    """

    # DEFAULT_MODEL = (
    #     "cross-encoder/ms-marco-MiniLM-L-6-v2"
    # )

    def __init__(
    self,
    model_name: str | None = None,
    config: Settings | None = None,
) -> None:
        self.config = config or settings

        resolved_model_name = (
            model_name
            if model_name is not None
            else self.config.reranker_model
        )

        if not resolved_model_name.strip():
            raise ValueError(
                "Reranker model name cannot be empty"
            )

        self.model_name = resolved_model_name

        self._model = CrossEncoder(
            resolved_model_name
        )

    def rerank(
        self,
        query: str,
        candidates: list[HybridResult],
        top_k: int = 5,
    ) -> list[RerankedResult]:
        """
        Rerank hybrid retrieval candidates.

        Args:
            query: Natural-language user query.
            candidates: Candidate results from hybrid retrieval.
            top_k: Maximum number of reranked results to return.

        Returns:
            Reranked results ordered by descending cross-encoder score.

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

        if not candidates:
            return []

        pairs = [
            [
                query,
                candidate.text,
            ]
            for candidate in candidates
        ]

        scores = self._model.predict(
            pairs
        )

        results = [
            RerankedResult(
                text=candidate.text,
                source=candidate.source,
                page_number=candidate.page_number,
                chunk_index=candidate.chunk_index,
                dense_score=candidate.dense_score,
                bm25_score=candidate.bm25_score,
                hybrid_score=candidate.score,
                rerank_score=float(score),
                doc_id=candidate.doc_id,

            )
            for candidate, score
            in zip(candidates, scores)
        ]

        results.sort(
            key=lambda result: result.rerank_score,
            reverse=True,
        )

        return results[:top_k]