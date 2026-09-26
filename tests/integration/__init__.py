import os
from concurrent.futures import ThreadPoolExecutor

import pytest
from sqlalchemy import create_engine
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import sessionmaker

from app.database import Base
from app.document_store import (
    DocumentStore,
    DuplicateDocumentInProgressError,
    StoredDocument,
)


POSTGRES_DATABASE_URL = os.getenv("DATABASE_URL")


pytestmark = pytest.mark.postgres


def create_postgres_store():
    if not POSTGRES_DATABASE_URL:
        pytest.skip(
            "DATABASE_URL is not configured"
        )

    engine = create_engine(
        POSTGRES_DATABASE_URL,
        pool_pre_ping=True,
    )

    session_factory = sessionmaker(
        bind=engine,
        autocommit=False,
        autoflush=False,
    )

    return DocumentStore(
        session_factory=session_factory
    )


def test_postgres_enforces_unique_content_hash():
    store = create_postgres_store()

    document_a = StoredDocument(
        doc_id="test_unique_doc_a",
        filename="unique-a.txt",
        stored_path="data/unique-a.txt",
        created_at=store.create_timestamp(),
        content_hash="postgres-unique-test-hash",
        status="completed",
    )

    document_b = StoredDocument(
        doc_id="test_unique_doc_b",
        filename="unique-b.txt",
        stored_path="data/unique-b.txt",
        created_at=store.create_timestamp(),
        content_hash="postgres-unique-test-hash",
        status="completed",
    )

    store.delete(document_a.doc_id)
    store.delete(document_b.doc_id)

    store.save(document_a)

    with pytest.raises(IntegrityError):
        store.save(document_b)

    store.delete(document_a.doc_id)
    store.delete(document_b.doc_id)


def test_postgres_concurrent_claim_allows_only_one_owner():
    store_a = create_postgres_store()
    store_b = create_postgres_store()

    content_hash = (
        "postgres-concurrent-claim-test-hash"
    )

    document_a = StoredDocument(
        doc_id="concurrent_doc_a",
        filename="concurrent-a.txt",
        stored_path="data/concurrent-a.txt",
        created_at=store_a.create_timestamp(),
        content_hash=content_hash,
        status="processing",
    )

    document_b = StoredDocument(
        doc_id="concurrent_doc_b",
        filename="concurrent-b.txt",
        stored_path="data/concurrent-b.txt",
        created_at=store_b.create_timestamp(),
        content_hash=content_hash,
        status="processing",
    )

    store_a.delete(document_a.doc_id)
    store_a.delete(document_b.doc_id)

    def claim_a():
        try:
            result = store_a.claim_document(document_a)
            return ("success", result.doc_id)
        except Exception as exc:
            return ("error", type(exc).__name__)

    def claim_b():
        try:
            result = store_b.claim_document(document_b)
            return ("success", result.doc_id)
        except Exception as exc:
            return ("error", type(exc).__name__)

    with ThreadPoolExecutor(
        max_workers=2
    ) as executor:
        future_a = executor.submit(claim_a)
        future_b = executor.submit(claim_b)

        result_a = future_a.result()
        result_b = future_b.result()

    results = [result_a, result_b]

    successful_claims = [
        result
        for result in results
        if result[0] == "success"
    ]

    duplicate_errors = [
        result
        for result in results
        if result[1]
        == "DuplicateDocumentInProgressError"
    ]

    assert len(successful_claims) == 1
    assert len(duplicate_errors) == 1

    stored = store_a.get_by_content_hash(
        content_hash
    )

    assert stored is not None
    assert stored.status == "processing"

    assert stored.doc_id in {
        "concurrent_doc_a",
        "concurrent_doc_b",
    }

    store_a.release_claim(stored.doc_id)