"""
Hybrid retrieval combining dense semantic retrieval
and lexical retrieval.
"""

from dataclasses import dataclass

from app.bm25_retriever import LexicalIndex
from app.retrieval import DenseRetriever, RetrievalResult


@dataclass
class HybridResult:
    """
    Represent one result after combining dense and lexical retrieval.

    Attributes:
        text: Retrieved document chunk text.
        source: Original document filename.
        page_number: Original page number.
        chunk_index: Position of the chunk within the document.
        dense_score: Score produced by dense retrieval.
        bm25_score: Score produced by lexical retrieval.
        score: Final hybrid retrieval score.
    """

    text: str
    source: str
    page_number: int
    chunk_index: int
    dense_score: float
    bm25_score: float
    score: float


class HybridRetriever:
    """
    Combine dense and lexical retrieval results.

    The hybrid retriever depends on retrieval capabilities,
    not on a specific lexical retrieval technology such as BM25.
    """

    def __init__(
        self,
        dense_retriever: DenseRetriever,
        lexical_retriever: LexicalIndex,
    ) -> None:
        self.dense_retriever = dense_retriever
        self.lexical_retriever = lexical_retriever

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[HybridResult]:
        dense_results = self.dense_retriever.retrieve(
            query,
            top_k=top_k,
        )

        lexical_results = self.lexical_retriever.retrieve(
            query,
            top_k=top_k,
        )

        return self._merge_results(
            dense_results,
            lexical_results,
            top_k=top_k,
        )

    @staticmethod
    def _merge_results(
        dense_results: list[RetrievalResult],
        lexical_results: list[RetrievalResult],
        top_k: int,
    ) -> list[HybridResult]:
        merged: dict[
            tuple[str, int, int],
            HybridResult,
        ] = {}

        # Add dense retrieval results.
        for result in dense_results:
            key = (
                result.source,
                result.page_number,
                result.chunk_index,
            )

            merged[key] = HybridResult(
                text=result.text,
                source=result.source,
                page_number=result.page_number,
                chunk_index=result.chunk_index,
                dense_score=result.score,
                bm25_score=0.0,
                score=result.score,
            )

        # Add lexical retrieval results or combine with
        # an existing dense result.
        for result in lexical_results:
            key = (
                result.source,
                result.page_number,
                result.chunk_index,
            )

            if key not in merged:
                merged[key] = HybridResult(
                    text=result.text,
                    source=result.source,
                    page_number=result.page_number,
                    chunk_index=result.chunk_index,
                    dense_score=0.0,
                    bm25_score=result.score,
                    score=result.score,
                )

            else:
                existing = merged[key]

                merged[key] = HybridResult(
                    text=existing.text,
                    source=existing.source,
                    page_number=existing.page_number,
                    chunk_index=existing.chunk_index,
                    dense_score=existing.dense_score,
                    bm25_score=result.score,
                    score=max(
                        existing.dense_score,
                        result.score,
                    ),
                )

        return sorted(
            merged.values(),
            key=lambda result: result.score,
            reverse=True,
        )[:top_k]
