from datetime import datetime

import pytest
from sqlalchemy import create_engine
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.database import Base
from app.models.document import DocumentModel
from app.models.knowledge_base import KnowledgeBaseModel


@pytest.fixture
def db_session():
    engine = create_engine(
        "sqlite:///:memory:",
    )

    Base.metadata.create_all(
        bind=engine,
    )

    with Session(engine) as session:
        knowledge_base = KnowledgeBaseModel(
            knowledge_base_id="kb_default",
            name="My Documents",
            slug="my-documents",
            description=None,
            active=True,
            created_at=datetime.now(),
        )

        session.add(knowledge_base)
        session.commit()

        yield session

    Base.metadata.drop_all(
        bind=engine,
    )

    engine.dispose()


def test_document_model_can_be_persisted(
    db_session,
):
    document = DocumentModel(
        doc_id="doc_test_001",
        filename="test.txt",
        stored_path="/tmp/test.txt",
        created_at=datetime.now(),
        content_hash="a" * 64,
        status="completed",
        knowledge_base_id="kb_default",
    )

    db_session.add(document)
    db_session.commit()

    stored = db_session.get(
        DocumentModel,
        "doc_test_001",
    )

    assert stored is not None

    assert stored.doc_id == "doc_test_001"

    assert stored.filename == "test.txt"

    assert stored.stored_path == "/tmp/test.txt"

    assert stored.content_hash == "a" * 64

    assert stored.status == "completed"

    assert stored.knowledge_base_id == "kb_default"


def test_document_model_rejects_duplicate_doc_id(
    db_session,
):
    first_document = DocumentModel(
        doc_id="doc_duplicate",
        filename="first.txt",
        stored_path="/tmp/first.txt",
        created_at=datetime.now(),
        content_hash="a" * 64,
        status="completed",
        knowledge_base_id="kb_default",
    )

    second_document = DocumentModel(
        doc_id="doc_duplicate",
        filename="second.txt",
        stored_path="/tmp/second.txt",
        created_at=datetime.now(),
        content_hash="b" * 64,
        status="completed",
        knowledge_base_id="kb_default",
    )

    db_session.add(first_document)
    db_session.commit()

    db_session.add(second_document)

    with pytest.raises(IntegrityError):
        db_session.commit()

    db_session.rollback()


def test_document_model_rejects_duplicate_content_hash(
    db_session,
):
    first_document = DocumentModel(
        doc_id="doc_hash_001",
        filename="first.txt",
        stored_path="/tmp/first.txt",
        created_at=datetime.now(),
        content_hash="c" * 64,
        status="completed",
        knowledge_base_id="kb_default",
    )

    second_document = DocumentModel(
        doc_id="doc_hash_002",
        filename="second.txt",
        stored_path="/tmp/second.txt",
        created_at=datetime.now(),
        content_hash="c" * 64,
        status="completed",
        knowledge_base_id="kb_default",
    )

    db_session.add(first_document)
    db_session.commit()

    db_session.add(second_document)

    with pytest.raises(IntegrityError):
        db_session.commit()

    db_session.rollback()


def test_document_model_requires_knowledge_base(
    db_session,
):
    document = DocumentModel(
        doc_id="doc_without_kb",
        filename="test.txt",
        stored_path="/tmp/test.txt",
        created_at=datetime.now(),
        content_hash="d" * 64,
        status="completed",
        knowledge_base_id=None,
    )

    db_session.add(document)

    with pytest.raises(IntegrityError):
        db_session.commit()

    db_session.rollback()

