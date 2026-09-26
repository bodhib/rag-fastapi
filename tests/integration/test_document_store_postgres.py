import os
from uuid import uuid4

import pytest
from sqlalchemy import create_engine
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import sessionmaker

from app.database import Base
from app.document_store import DocumentStore, StoredDocument

from concurrent.futures import ThreadPoolExecutor

from app.document_store import (
    DuplicateDocumentInProgressError,
    StoredDocument,
)


DATABASE_URL = os.environ["DATABASE_URL"]


def create_postgres_store() -> DocumentStore:
    engine = create_engine(
        DATABASE_URL,
        pool_pre_ping=True,
    )

    session_factory = sessionmaker(
        bind=engine,
        autocommit=False,
        autoflush=False,
    )

    return DocumentStore(session_factory=session_factory)


@pytest.mark.postgres
def test_postgres_enforces_unique_content_hash():
    store = create_postgres_store()

    content_hash = uuid4().hex

    first = StoredDocument(
        doc_id=f"doc_{uuid4().hex}",
        filename="first.txt",
        stored_path="data/uploads/first.txt",
        created_at=store.create_timestamp(),
        content_hash=content_hash,
        status="completed",
    )

    second = StoredDocument(
        doc_id=f"doc_{uuid4().hex}",
        filename="second.txt",
        stored_path="data/uploads/second.txt",
        created_at=store.create_timestamp(),
        content_hash=content_hash,
        status="completed",
    )

    store.save(first)

    with pytest.raises(IntegrityError):
        store.save(second)

    store.delete(first.doc_id)

@pytest.mark.postgres
def test_postgres_concurrent_claim_allows_only_one_owner():
    content_hash = uuid4().hex

    document_a = StoredDocument(
        doc_id=f"doc_{uuid4().hex}",
        filename="same.txt",
        stored_path="data/uploads/a.txt",
        created_at=create_postgres_store().create_timestamp(),
        content_hash=content_hash,
        status="processing",
    )

    document_b = StoredDocument(
        doc_id=f"doc_{uuid4().hex}",
        filename="same.txt",
        stored_path="data/uploads/b.txt",
        created_at=create_postgres_store().create_timestamp(),
        content_hash=content_hash,
        status="processing",
    )

    def attempt_claim(document: StoredDocument):
        store = create_postgres_store()

        try:
            result = store.claim_document(document)

            return {
                "success": True,
                "doc_id": result.doc_id,
            }

        except DuplicateDocumentInProgressError:
            return {
                "success": False,
                "duplicate": True,
            }

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(attempt_claim, document_a),
            executor.submit(attempt_claim, document_b),
        ]

        results = [future.result() for future in futures]

    successful_claims = [
        result
        for result in results
        if result["success"]
    ]

    duplicate_results = [
        result
        for result in results
        if result.get("duplicate")
    ]

    assert len(successful_claims) == 1
    assert len(duplicate_results) == 1

    winner_doc_id = successful_claims[0]["doc_id"]

    store = create_postgres_store()
    store.release_claim(winner_doc_id)    