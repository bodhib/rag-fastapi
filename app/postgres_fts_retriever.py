### `app/postgres_fts_retriever.py`

"""
PostgreSQL Full-Text Search retrieval.

Chunks are persisted in PostgreSQL so lexical retrieval is shared
across application instances.

PostgreSQL uses its native full-text search implementation through
to_tsvector/plainto_tsquery.
"""

from collections.abc import Callable

from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session

from app.chunker import DocumentChunk
from app.models.chunk import ChunkModel
from app.retrieval import RetrievalResult


class PostgresFTSRetriever:
    """
    PostgreSQL-backed lexical retrieval implementation.

    Unlike BM25Retriever, this implementation does not maintain
    an in-memory lexical index. The searchable corpus lives in
    PostgreSQL and can therefore be shared by multiple API instances.
    """

    def __init__(
        self,
        session_factory: Callable[[], Session],
    ) -> None:
        self.session_factory = session_factory

    def build(
        self,
        chunks: list[DocumentChunk],
    ) -> int:
        """
        Persist chunks into PostgreSQL for lexical retrieval.

        Existing chunks belonging to the same documents are replaced.
        """

        if not chunks:
            return 0

        document_ids = {
            chunk.doc_id
            for chunk in chunks
        }

        with self.session_factory() as session:
            session.execute(
                delete(ChunkModel).where(
                    ChunkModel.doc_id.in_(document_ids)
                )
            )

            for chunk in chunks:
                session.add(
                    ChunkModel(
                        chunk_id=(
                            f"{chunk.doc_id}:"
                            f"{chunk.chunk_index}"
                        ),
                        doc_id=chunk.doc_id,
                        text=chunk.text,
                        source=chunk.source,
                        page_number=chunk.page_number,
                        chunk_index=chunk.chunk_index,
                    )
                )

            session.commit()

        return len(chunks)

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[RetrievalResult]:
        """
        Retrieve chunks using PostgreSQL full-text search.
        """

        if not query.strip():
            raise ValueError("query cannot be empty")

        if top_k <= 0:
            raise ValueError(
                "top_k must be greater than zero"
            )

        with self.session_factory() as session:
            search_vector = func.to_tsvector(
                "english",
                ChunkModel.text,
            )

            search_query = func.plainto_tsquery(
                "english",
                query,
            )

            score = func.ts_rank_cd(
                search_vector,
                search_query,
            )

            statement = (
                select(
                    ChunkModel,
                    score.label("score"),
                )
                .where(
                    search_vector.op("@@")(search_query)
                )
                .order_by(
                    score.desc()
                )
                .limit(top_k)
            )

            rows = session.execute(statement).all()

            return [
                RetrievalResult(
                    text=chunk.text,
                    source=chunk.source,
                    page_number=chunk.page_number,
                    chunk_index=chunk.chunk_index,
                    score=float(score_value),
                )
                for chunk, score_value in rows
            ]

    def remove_document(
        self,
        doc_id: str,
    ) -> None:
        if not doc_id.strip():
            raise ValueError("doc_id cannot be empty")

        with self.session_factory() as session:
            session.execute(
                delete(ChunkModel).where(
                    ChunkModel.doc_id == doc_id
                )
            )
            session.commit()