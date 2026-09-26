"""
Context compression.

This module reduces reranked retrieval results to a smaller set of
highly relevant context chunks while preserving their original text
and citation metadata.
"""

from app.reranker import RerankedResult


class ContextCompressor:
    """
    Compress reranked retrieval results to a smaller context set.

    The current implementation uses extractive compression: it keeps
    the highest-ranked chunks unchanged rather than rewriting or
    summarizing their contents.
    """

    def compress(
        self,
        results: list[RerankedResult],
        top_k: int = 3,
    ) -> list[RerankedResult]:
        """
        Select the highest-ranked results as final context.

        Args:
            results: Reranked retrieval results.
            top_k: Maximum number of context chunks to retain.

        Returns:
            The highest-ranked results, preserving their original
            ordering and metadata.

        Raises:
            ValueError: If top_k is not greater than zero.
        """

        if top_k <= 0:
            raise ValueError(
                "top_k must be greater than zero"
            )

        if not results:
            return []

        return results[:top_k]