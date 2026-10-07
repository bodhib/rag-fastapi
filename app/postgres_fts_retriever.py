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

    The searchable corpus lives in PostgreSQL and can therefore be
    shared by multiple API instances.
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
                    ChunkModel.doc_id.in_(
                        document_ids
                    )
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
                        knowledge_base_id=(
                            chunk.knowledge_base_id
                        ),
                    )
                )

            session.commit()

        return len(chunks)

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        knowledge_base_id: str = "kb_default",
        allowed_doc_ids: set[str] | None = None,
    ) -> list[RetrievalResult]:
        """
        Retrieve chunks using PostgreSQL full-text search.

        When a knowledge base ID is supplied, only chunks belonging
        to that knowledge base are eligible for retrieval.

        When allowed_doc_ids is supplied, retrieval is restricted
        to chunks belonging to those specific documents.
        """

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

            conditions = [
                search_vector.op("@@")(
                    search_query
                )
            ]

            if knowledge_base_id is not None:
                conditions.append(
                    ChunkModel.knowledge_base_id
                    == knowledge_base_id
                )

            # Restrict lexical retrieval to the documents
            # belonging to the eligible candidates.
            if allowed_doc_ids is not None:
                conditions.append(
                    ChunkModel.doc_id.in_(
                        allowed_doc_ids
                    )
                )

            statement = (
                select(
                    ChunkModel,
                    score.label("score"),
                )
                .where(*conditions)
                .order_by(
                    score.desc()
                )
                .limit(top_k)
            )

            rows = session.execute(
                statement
            ).all()

            return [
                RetrievalResult(
                    text=chunk.text,
                    source=chunk.source,
                    page_number=chunk.page_number,
                    chunk_index=chunk.chunk_index,
                    score=float(
                        score_value
                    ),
                    doc_id=chunk.doc_id,
                )
                for chunk, score_value in rows
            ]

    def remove_document(
        self,
        doc_id: str,
    ) -> None:
        if not doc_id.strip():
            raise ValueError(
                "doc_id cannot be empty"
            )

        with self.session_factory() as session:
            session.execute(
                delete(ChunkModel).where(
                    ChunkModel.doc_id == doc_id
                )
            )

            session.commit()