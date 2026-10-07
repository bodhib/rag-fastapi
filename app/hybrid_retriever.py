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
    """

    text: str
    source: str
    page_number: int
    chunk_index: int
    dense_score: float
    bm25_score: float
    score: float
    doc_id: str = ""


class HybridRetriever:
    """
    Combine dense and lexical retrieval results.

    Both retrieval branches receive the same knowledge base ID,
    ensuring that hybrid retrieval never combines results from
    different knowledge bases.
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
        knowledge_base_id: str | None = None,
        allowed_doc_ids: set[str] | None = None,
    ) -> list[HybridResult]:

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

            dense_results = (
                self.dense_retriever.retrieve(
                    query=query,
                    top_k=top_k,
                    knowledge_base_id=knowledge_base_id,
                    allowed_doc_ids=allowed_doc_ids,
                )
            )

            lexical_results = (
                self.lexical_retriever.retrieve(
                    query=query,
                    top_k=top_k,
                    knowledge_base_id=knowledge_base_id,
                    allowed_doc_ids=allowed_doc_ids,
                )
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

        for result in dense_results:
            key = (
                result.doc_id,
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
                doc_id=result.doc_id,
            )

        for result in lexical_results:
            key = (
                result.doc_id,
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
                    doc_id=result.doc_id,
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
                    doc_id=result.doc_id,
                )

        return sorted(
            merged.values(),
            key=lambda result: result.score,
            reverse=True,
        )[:top_k]