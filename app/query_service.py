"""
RAG query orchestration.

This module coordinates retrieval, reranking, context compression,
and LLM generation for a single user query.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from app.config import Settings, settings


if TYPE_CHECKING:
    from app.compressor import ContextCompressor
    from app.generator import GenerationResult, GeminiGenerator
    from app.hybrid_retriever import HybridRetriever
    from app.reranker import CrossEncoderReranker, RerankedResult


@dataclass
class QueryResult:
    """
    Represent the result of the complete RAG query pipeline.
    """

    answer: str
    contexts: list[RerankedResult]


class QueryService:
    """
    Coordinate the complete RAG query pipeline.

    Pipeline:

        Query
          ↓
        KB-aware Hybrid Retrieval
          ↓
        Cross-Encoder Reranking
          ↓
        Context Compression
          ↓
        Gemini Generation
    """

    def __init__(
        self,
        hybrid_retriever,
        reranker,
        compressor,
        generator,
        config: Settings | None = None,
    ) -> None:
        self.hybrid_retriever = hybrid_retriever
        self.reranker = reranker
        self.compressor = compressor
        self.generator = generator
        self.config = config or settings

    def query(
        self,
        question: str,
        knowledge_base_id: str = "kb_default",
        retrieval_top_k: int | None = None,
        rerank_top_k: int | None = None,
        context_top_k: int | None = None,
    ) -> QueryResult:

        if not question.strip():
            raise ValueError(
                "question cannot be empty"
            )

        if not knowledge_base_id.strip():
            raise ValueError(
                "knowledge_base_id cannot be empty"
            )

        resolved_retrieval_top_k = (
            retrieval_top_k
            if retrieval_top_k is not None
            else self.config.retrieval_top_k
        )

        resolved_rerank_top_k = (
            rerank_top_k
            if rerank_top_k is not None
            else self.config.rerank_top_k
        )

        resolved_context_top_k = (
            context_top_k
            if context_top_k is not None
            else self.config.context_top_k
        )

        if resolved_retrieval_top_k <= 0:
            raise ValueError(
                "retrieval_top_k must be greater than zero"
            )

        if resolved_rerank_top_k <= 0:
            raise ValueError(
                "rerank_top_k must be greater than zero"
            )

        if resolved_context_top_k <= 0:
            raise ValueError(
                "context_top_k must be greater than zero"
            )

        hybrid_results = (
            self.hybrid_retriever.retrieve(
                query=question,
                top_k=resolved_retrieval_top_k,
                knowledge_base_id=knowledge_base_id,
            )
        )

        if not hybrid_results:
            raise ValueError(
                "No relevant documents found"
            )

        reranked_results = (
            self.reranker.rerank(
                query=question,
                candidates=hybrid_results,
                top_k=resolved_rerank_top_k,
            )
        )

        if not reranked_results:
            raise ValueError(
                "No relevant documents remained after reranking"
            )

        contexts = self.compressor.compress(
            results=reranked_results,
            top_k=resolved_context_top_k,
        )

        if not contexts:
            raise ValueError(
                "No context available for generation"
            )

        generation_result: GenerationResult = (
            self.generator.generate(
                query=question,
                contexts=contexts,
            )
        )

        return QueryResult(
            answer=generation_result.answer,
            contexts=generation_result.contexts,
        )