from datetime import datetime
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.database import Base
from app.models.document import DocumentModel

import pytest
from sqlalchemy.exc import IntegrityError


def test_document_model_can_be_persisted(
    tmp_path: Path,
) -> None:
    database_url = f"sqlite:///{tmp_path / 'documents.db'}"

    engine = create_engine(database_url)

    Base.metadata.create_all(bind=engine)

    created_at = datetime.now()

    with Session(engine) as session:
        document = DocumentModel(
            doc_id="doc_001",
            filename="leave-policy.pdf",
            stored_path="data/uploads/leave-policy.pdf",
            created_at=created_at,
        )

        session.add(document)
        session.commit()

    with Session(engine) as session:
        saved_document = session.get(
            DocumentModel,
            "doc_001",
        )

        assert saved_document is not None
        assert saved_document.doc_id == "doc_001"
        assert saved_document.filename == "leave-policy.pdf"
        assert (
            saved_document.stored_path
            == "data/uploads/leave-policy.pdf"
        )
        assert saved_document.created_at == created_at

def test_document_model_requires_doc_id(
    tmp_path: Path,
) -> None:
    database_url = f"sqlite:///{tmp_path / 'documents.db'}"

    engine = create_engine(database_url)

    Base.metadata.create_all(bind=engine)

    with Session(engine) as session:
        document = DocumentModel(
            filename="leave-policy.pdf",
            stored_path="data/uploads/leave-policy.pdf",
            created_at=datetime.now(),
        )

        session.add(document)

        with pytest.raises(IntegrityError):
            session.commit()

def test_document_model_rejects_duplicate_doc_id(
    tmp_path: Path,
) -> None:
    database_url = f"sqlite:///{tmp_path / 'documents.db'}"

    engine = create_engine(database_url)

    Base.metadata.create_all(bind=engine)

    created_at = datetime.now()

    with Session(engine) as session:
        first_document = DocumentModel(
            doc_id="doc_001",
            filename="first.pdf",
            stored_path="data/uploads/first.pdf",
            created_at=created_at,
        )

        second_document = DocumentModel(
            doc_id="doc_001",
            filename="second.pdf",
            stored_path="data/uploads/second.pdf",
            created_at=created_at,
        )

        session.add(first_document)
        session.commit()

        session.add(second_document)

        with pytest.raises(IntegrityError):
            session.commit()                    