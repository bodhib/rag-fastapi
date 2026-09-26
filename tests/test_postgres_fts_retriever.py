### `tests/test_postgres_fts_retriever.py`

from datetime import datetime

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.chunker import DocumentChunk
from app.database import Base
from app.models.chunk import ChunkModel
from app.postgres_fts_retriever import PostgresFTSRetriever


def create_test_session_factory(tmp_path):
    database_url = (
        f"sqlite:///{tmp_path / 'lexical.db'}"
    )

    engine = create_engine(database_url)

    Base.metadata.create_all(
        bind=engine,
        tables=[ChunkModel.__table__],
    )

    return sessionmaker(
        bind=engine,
        autocommit=False,
        autoflush=False,
    )


def make_chunk(
    doc_id: str,
    chunk_index: int,
    text: str,
) -> DocumentChunk:
    return DocumentChunk(
        doc_id=doc_id,
        text=text,
        source="test.txt",
        page_number=1,
        chunk_index=chunk_index,
    )


def test_build_persists_chunks(tmp_path):
    session_factory = create_test_session_factory(
        tmp_path
    )

    retriever = PostgresFTSRetriever(
        session_factory=session_factory
    )

    chunks = [
        make_chunk(
            doc_id="doc_1",
            chunk_index=0,
            text="Annual leave policy allows employees to carry unused leave.",
        ),
        make_chunk(
            doc_id="doc_1",
            chunk_index=1,
            text="Employees must submit leave requests through the HR system.",
        ),
    ]

    count = retriever.build(chunks)

    assert count == 2

    with session_factory() as session:
        stored_chunks = (
            session.query(ChunkModel)
            .order_by(ChunkModel.chunk_index)
            .all()
        )

        assert len(stored_chunks) == 2
        assert stored_chunks[0].doc_id == "doc_1"
        assert stored_chunks[0].chunk_index == 0
        assert "Annual leave" in stored_chunks[0].text


def test_build_replaces_existing_document_chunks(tmp_path):
    session_factory = create_test_session_factory(
        tmp_path
    )

    retriever = PostgresFTSRetriever(
        session_factory=session_factory
    )

    retriever.build(
        [
            make_chunk(
                doc_id="doc_1",
                chunk_index=0,
                text="Old content",
            ),
        ]
    )

    retriever.build(
        [
            make_chunk(
                doc_id="doc_1",
                chunk_index=0,
                text="New content",
            ),
            make_chunk(
                doc_id="doc_1",
                chunk_index=1,
                text="Additional content",
            ),
        ]
    )

    with session_factory() as session:
        stored_chunks = (
            session.query(ChunkModel)
            .order_by(ChunkModel.chunk_index)
            .all()
        )

        assert len(stored_chunks) == 2
        assert stored_chunks[0].text == "New content"
        assert stored_chunks[1].text == "Additional content"


def test_retrieve_rejects_empty_query(tmp_path):
    session_factory = create_test_session_factory(
        tmp_path
    )

    retriever = PostgresFTSRetriever(
        session_factory=session_factory
    )

    try:
        retriever.retrieve("")
        assert False
    except ValueError as exc:
        assert str(exc) == "query cannot be empty"


def test_retrieve_rejects_invalid_top_k(tmp_path):
    session_factory = create_test_session_factory(
        tmp_path
    )

    retriever = PostgresFTSRetriever(
        session_factory=session_factory
    )

    try:
        retriever.retrieve(
            query="leave",
            top_k=0,
        )
        assert False
    except ValueError as exc:
        assert str(exc) == (
            "top_k must be greater than zero"
        )

def test_remove_document_removes_only_target_document(
    postgres_session_factory,
) -> None:
    retriever = PostgresFTSRetriever(
        session_factory=postgres_session_factory,
    )

    chunks = [
        DocumentChunk(
            doc_id="doc_a",
            text="annual leave policy for employees",
            source="doc_a.txt",
            page_number=1,
            chunk_index=0,
        ),
        DocumentChunk(
            doc_id="doc_b",
            text="medical insurance policy for employees",
            source="doc_b.txt",
            page_number=1,
            chunk_index=0,
        ),
    ]

    retriever.build(chunks)

    retriever.remove_document("doc_a")

    results = retriever.retrieve("annual leave")

    assert results == []

    results = retriever.retrieve("medical insurance")

    assert len(results) == 1
    assert results[0].source == "doc_b.txt"        
