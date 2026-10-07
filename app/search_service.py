"""
Enterprise knowledge search service.

SearchService provides ranked document/chunk search without LLM
generation. It reuses the existing hybrid retrieval and reranking
pipeline so search and Q&A share the same retrieval behavior.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.config import Settings
from app.hybrid_retriever import HybridRetriever
from app.reranker import RerankedResult


@dataclass(frozen=True)
class SearchResult:
    """
    A ranked enterprise-search result.

    The result intentionally contains both the original retrieval
    scores and the CrossEncoder reranking score so the API can later
    expose richer search diagnostics if required.
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

    @property
    def snippet(self) -> str:
        """
        Return the searchable text as the result snippet.

        Keeping this as a property allows us to introduce real
        snippet extraction/highlighting later without changing
        the API/domain model.
        """
        return self.text


class SearchService:
    """
    Enterprise knowledge search.

    Pipeline:

        query
          ↓
        hybrid retrieval
          ↓
        CrossEncoder reranking
          ↓
        ranked search results

    No LLM generation is performed here.
    """

    DEFAULT_TOP_K = 10
    MAX_TOP_K = 50

    def __init__(
        self,
        hybrid_retriever: HybridRetriever,
        reranker,
        config: Settings,
    ) -> None:
        self.hybrid_retriever = hybrid_retriever
        self.reranker = reranker
        self.config = config

    def search(
        self,
        query: str,
        knowledge_base_id: str = "kb_default",
        top_k: int = DEFAULT_TOP_K,
        allowed_doc_ids: set[str] | None = None,
    ) -> list[SearchResult]:
        """
        Search a knowledge base and return ranked evidence.

        `top_k` controls the number of final reranked results.

        Retrieval uses at least the configured retrieval_top_k
        candidates, or top_k when the caller requests more results.
        """

        if not query or not query.strip():
            raise ValueError("query cannot be empty")

        if not knowledge_base_id or not knowledge_base_id.strip():
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

        if top_k <= 0:
            raise ValueError(
                "top_k must be greater than zero"
            )

        if top_k > self.MAX_TOP_K:
            raise ValueError(
                f"top_k cannot be greater than {self.MAX_TOP_K}"
            )

        retrieval_top_k = max(
            self.config.retrieval_top_k,
            top_k,
        )

        candidates = self.hybrid_retriever.retrieve(
            query=query,
            top_k=retrieval_top_k,
            knowledge_base_id=knowledge_base_id,
            allowed_doc_ids=allowed_doc_ids,

        )

        if not candidates:
            return []

        reranked = self.reranker.rerank(
            query=query,
            candidates=candidates,
            top_k=top_k,
        )

        if not reranked:
            return []

        return [
            self._to_search_result(result)
            for result in reranked
        ]

    @staticmethod
    def _to_search_result(
        result: RerankedResult,
    ) -> SearchResult:
        return SearchResult(
            text=result.text,
            source=result.source,
            page_number=result.page_number,
            chunk_index=result.chunk_index,
            dense_score=result.dense_score,
            bm25_score=result.bm25_score,
            hybrid_score=result.hybrid_score,
            rerank_score=result.rerank_score,
            doc_id=result.doc_id,
        )