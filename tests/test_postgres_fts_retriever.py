from datetime import datetime

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.chunker import DocumentChunk
from app.database import Base
from app.models.chunk import ChunkModel
from app.models.knowledge_base import KnowledgeBaseModel
from app.postgres_fts_retriever import PostgresFTSRetriever


def create_test_session_factory(tmp_path):
    database_url = (
        f"sqlite:///{tmp_path / 'lexical.db'}"
    )

    engine = create_engine(database_url)

    Base.metadata.create_all(
        bind=engine,
        tables=[
            KnowledgeBaseModel.__table__,
            ChunkModel.__table__,
        ],
    )

    session_factory = sessionmaker(
        bind=engine,
        autocommit=False,
        autoflush=False,
    )

    with session_factory() as session:
        session.add_all(
            [
                KnowledgeBaseModel(
                    knowledge_base_id="kb_default",
                    name="My Documents",
                    slug="my-documents",
                    description=None,
                    active=True,
                    created_at=datetime.now(),
                ),
                KnowledgeBaseModel(
                    knowledge_base_id="kb_company",
                    name="Company Knowledge",
                    slug="company",
                    description=None,
                    active=True,
                    created_at=datetime.now(),
                ),
            ]
        )
        session.commit()

    return session_factory


def make_chunk(
    doc_id: str,
    chunk_index: int,
    text: str,
    knowledge_base_id: str = "kb_default",
) -> DocumentChunk:
    return DocumentChunk(
        doc_id=doc_id,
        text=text,
        source="test.txt",
        page_number=1,
        chunk_index=chunk_index,
        knowledge_base_id=knowledge_base_id,
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
            text=(
                "Annual leave policy allows employees "
                "to carry unused leave."
            ),
        ),
        make_chunk(
            doc_id="doc_1",
            chunk_index=1,
            text=(
                "Employees must submit leave requests "
                "through the HR system."
            ),
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
        assert (
            stored_chunks[0].knowledge_base_id
            == "kb_default"
        )


def test_build_persists_knowledge_base_id(tmp_path):
    session_factory = create_test_session_factory(
        tmp_path
    )

    retriever = PostgresFTSRetriever(
        session_factory=session_factory
    )

    retriever.build(
        [
            make_chunk(
                doc_id="doc_company",
                chunk_index=0,
                text="Company leave policy.",
                knowledge_base_id="kb_company",
            )
        ]
    )

    with session_factory() as session:
        stored = session.query(ChunkModel).one()

        assert (
            stored.knowledge_base_id
            == "kb_company"
        )


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


def test_retrieve_filters_by_knowledge_base(
    postgres_session_factory,
) -> None:
    retriever = PostgresFTSRetriever(
        session_factory=postgres_session_factory,
    )

    retriever.build(
        [
            DocumentChunk(
                doc_id="doc_default_1",
                text=(
                    "Annual leave policy for "
                    "default knowledge base."
                ),
                source="default.txt",
                page_number=1,
                chunk_index=0,
                knowledge_base_id="kb_default",
            ),
            DocumentChunk(
                doc_id="doc_default_2",
                text=(
                    "Employees may carry forward "
                    "unused annual leave."
                ),
                source="default.txt",
                page_number=2,
                chunk_index=0,
                knowledge_base_id="kb_default",
            ),
            DocumentChunk(
                doc_id="doc_company_1",
                text=(
                    "Annual leave policy for "
                    "company knowledge base."
                ),
                source="company.txt",
                page_number=1,
                chunk_index=0,
                knowledge_base_id="kb_company",
            ),
            DocumentChunk(
                doc_id="doc_company_2",
                text=(
                    "Company employees may carry "
                    "forward annual leave."
                ),
                source="company.txt",
                page_number=2,
                chunk_index=0,
                knowledge_base_id="kb_company",
            ),
        ]
    )

    results = retriever.retrieve(
        query="annual leave",
        top_k=10,
        knowledge_base_id="kb_company",
    )

    assert len(results) == 2

    assert all(
        result.source == "company.txt"
        for result in results
    )

def test_empty_knowledge_base_id_is_rejected(tmp_path):
    session_factory = create_test_session_factory(
        tmp_path
    )

    retriever = PostgresFTSRetriever(
        session_factory=session_factory
    )

    try:
        retriever.retrieve(
            "leave",
            knowledge_base_id="   ",
        )
        assert False
    except ValueError as exc:
        assert str(exc) == (
            "knowledge_base_id cannot be empty"
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
            knowledge_base_id="kb_default",
        ),
        DocumentChunk(
            doc_id="doc_b",
            text="medical insurance policy for employees",
            source="doc_b.txt",
            page_number=1,
            chunk_index=0,
            knowledge_base_id="kb_default",
        ),
    ]

    retriever.build(chunks)

    retriever.remove_document("doc_a")

    results = retriever.retrieve("annual leave")

    assert results == []

    results = retriever.retrieve("medical insurance")

    assert len(results) == 1
    assert results[0].source == "doc_b.txt"

def test_retrieve_uses_default_knowledge_base(
    postgres_session_factory,
) -> None:
    retriever = PostgresFTSRetriever(
        session_factory=postgres_session_factory,
    )

    retriever.build(
        [
            DocumentChunk(
                doc_id="doc_default",
                text=(
                    "Annual leave policy for "
                    "default knowledge base."
                ),
                source="default.txt",
                page_number=1,
                chunk_index=0,
                knowledge_base_id="kb_default",
            ),
            DocumentChunk(
                doc_id="doc_company",
                text=(
                    "Annual leave policy for "
                    "company knowledge base."
                ),
                source="company.txt",
                page_number=1,
                chunk_index=0,
                knowledge_base_id="kb_company",
            ),
        ]
    )

    results = retriever.retrieve(
        query="annual leave",
    )

    assert len(results) == 1
    assert results[0].source == "default.txt"    

def test_retrieve_top_k_is_scoped_to_knowledge_base(
    postgres_session_factory,
) -> None:
    retriever = PostgresFTSRetriever(
        session_factory=postgres_session_factory,
    )

    retriever.build(
        [
            DocumentChunk(
                doc_id="doc_default_1",
                text="Annual leave policy one.",
                source="default-1.txt",
                page_number=1,
                chunk_index=0,
                knowledge_base_id="kb_default",
            ),
            DocumentChunk(
                doc_id="doc_default_2",
                text="Annual leave policy two.",
                source="default-2.txt",
                page_number=1,
                chunk_index=0,
                knowledge_base_id="kb_default",
            ),
            DocumentChunk(
                doc_id="doc_company_1",
                text="Annual leave company policy one.",
                source="company-1.txt",
                page_number=1,
                chunk_index=0,
                knowledge_base_id="kb_company",
            ),
            DocumentChunk(
                doc_id="doc_company_2",
                text="Annual leave company policy two.",
                source="company-2.txt",
                page_number=1,
                chunk_index=0,
                knowledge_base_id="kb_company",
            ),
        ]
    )

    results = retriever.retrieve(
        query="annual leave",
        top_k=1,
        knowledge_base_id="kb_company",
    )

    assert len(results) == 1
    assert results[0].source.startswith(
        "company-"
    )    