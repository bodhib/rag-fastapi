"""
BM25 lexical retrieval.

This module provides keyword-based document retrieval using the
BM25 ranking algorithm.
"""

import re
from typing import Protocol, runtime_checkable

from rank_bm25 import BM25Okapi

from app.chunker import DocumentChunk
from app.retrieval import RetrievalResult


@runtime_checkable
class LexicalIndex(Protocol):
    """
    Contract for lexical retrieval backends.

    The application only requires lexical retrieval.
    How the underlying index is built or maintained is an
    implementation detail of the concrete backend.
    """

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[RetrievalResult]:
        ...


@runtime_checkable
class RebuildableLexicalIndex(LexicalIndex, Protocol):
    """
    Contract for lexical backends that require an explicit
    rebuild from document chunks.

    BM25 is currently implemented this way because its index
    is maintained in application memory.

    A future persistent lexical backend, such as PostgreSQL
    Full-Text Search, may implement only LexicalIndex and
    therefore not require this operation.
    """

    def build(
        self,
        chunks: list[DocumentChunk],
    ) -> None:
        ...

    def remove_document(
        self,
        doc_id: str,
    ) -> None:
        ...   


class BM25Retriever:
    """
    Retrieve document chunks using BM25 lexical matching.

    BM25 is complementary to dense semantic retrieval. It is
    particularly useful when exact terms, product names, SKUs,
    identifiers, or keywords matter.
    """

    def __init__(
        self,
        chunks: list[DocumentChunk] | None = None,
    ) -> None:
        self._chunks: list[DocumentChunk] = []
        self._bm25: BM25Okapi | None = None

        if chunks is not None:
            self.build(chunks)

    def build(
        self,
        chunks: list[DocumentChunk],
    ) -> None:
        if not chunks:
            raise ValueError(
                "Cannot build BM25 index from empty chunk list"
            )

        valid_chunks = [
            chunk
            for chunk in chunks
            if chunk.text.strip()
        ]

        if not valid_chunks:
            raise ValueError(
                "Cannot build BM25 index from empty chunks"
            )

        tokenized_corpus = [
            self._tokenize(chunk.text)
            for chunk in valid_chunks
        ]

        self._chunks = valid_chunks
        self._bm25 = BM25Okapi(tokenized_corpus)

    def retrieve(
    self,
    query: str,
    top_k: int = 5,
) -> list[RetrievalResult]:
        if not query.strip():
            raise ValueError("query cannot be empty")

        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")

        if self._bm25 is None:
            raise ValueError("BM25 index has not been built")

        query_tokens = self._tokenize(query)

        if not query_tokens:
            return []

        scores = self._bm25.get_scores(query_tokens)

        ranked_indexes = sorted(
            range(len(scores)),
            key=lambda index: scores[index],
            reverse=True,
        )

        results: list[RetrievalResult] = []

        for index in ranked_indexes[:top_k]:
            chunk = self._chunks[index]

            chunk_tokens = set(
                self._tokenize(chunk.text)
            )

            has_query_term = any(
                token in chunk_tokens
                for token in query_tokens
            )

            score = float(scores[index])

            if score <= 0 and not has_query_term:
                continue

            results.append(
                RetrievalResult(
                    text=chunk.text,
                    source=chunk.source,
                    page_number=chunk.page_number,
                    chunk_index=chunk.chunk_index,
                    score=score,
                )
            )

        return results
    
    def remove_document(
    self,
    doc_id: str,
) -> None:
        if not doc_id.strip():
            raise ValueError("doc_id cannot be empty")

        remaining_chunks = [
            chunk
            for chunk in self._chunks
            if chunk.doc_id != doc_id
        ]

        if not remaining_chunks:
            self._chunks = []
            self._bm25 = None
            return

        self.build(remaining_chunks)
    
    @staticmethod
    def _tokenize(
        text: str,
    ) -> list[str]:
        return re.findall(
            r"\w+",
            text.lower(),
            flags=re.UNICODE,
        )
