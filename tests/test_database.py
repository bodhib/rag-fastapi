from datetime import datetime

from sqlalchemy import inspect, text
from sqlalchemy.orm import Session

from app.database import (
    SessionLocal,
    engine,
    get_db_session,
)
from app.document_store import DocumentStore, StoredDocument


def test_database_engine_can_execute_query() -> None:
    with Session(engine) as session:
        result = session.execute(
            text("SELECT 1")
        )

        assert result.scalar_one() == 1


def test_get_db_session_provides_session() -> None:
    generator = get_db_session()

    session = next(generator)

    try:
        assert isinstance(session, Session)

        result = session.execute(
            text("SELECT 1")
        )

        assert result.scalar_one() == 1
    finally:
        generator.close()


def test_documents_table_exists() -> None:
    inspector = inspect(engine)

    table_names = inspector.get_table_names()

    assert "documents" in table_names


def test_document_store_works_with_application_database() -> None:
    store = DocumentStore(
        session_factory=SessionLocal,
    )

    document = StoredDocument(
        doc_id="db_test_001",
        filename="database-test.txt",
        stored_path="data/uploads/database-test.txt",
        created_at=datetime.now(),
    )

    store.save(document)

    saved_document = store.get("db_test_001")

    assert saved_document is not None
    assert saved_document.doc_id == "db_test_001"
    assert saved_document.filename == "database-test.txt"

    store.delete("db_test_001")


def test_ingestion_jobs_table_exists() -> None:
    inspector = inspect(engine)

    table_names = inspector.get_table_names()

    assert "ingestion_jobs" in table_names