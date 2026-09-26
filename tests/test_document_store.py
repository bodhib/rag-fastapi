from datetime import datetime

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database import Base
from app.document_store import (
    DocumentStore,
    DuplicateDocumentInProgressError,
    StoredDocument,
)
from app.models.document import DocumentModel


def create_test_store(tmp_path):
    database_path = tmp_path / "test.db"

    engine = create_engine(
        f"sqlite:///{database_path}"
    )

    Base.metadata.create_all(engine)

    session_factory = sessionmaker(
        bind=engine,
        autocommit=False,
        autoflush=False,
    )

    return DocumentStore(
        session_factory=session_factory
    )


def create_document(
    *,
    doc_id="doc_123",
    filename="policy.txt",
    stored_path="data/policy.txt",
    content_hash="abc123",
    status="completed",
):
    return StoredDocument(
        doc_id=doc_id,
        filename=filename,
        stored_path=stored_path,
        created_at=datetime.now(),
        content_hash=content_hash,
        status=status,
    )


def test_save_and_get_document(tmp_path):
    store = create_test_store(tmp_path)

    document = create_document()

    store.save(document)

    result = store.get("doc_123")

    assert result is not None
    assert result.doc_id == "doc_123"
    assert result.filename == "policy.txt"
    assert result.content_hash == "abc123"
    assert result.status == "completed"


def test_get_missing_document_returns_none(tmp_path):
    store = create_test_store(tmp_path)

    result = store.get("does_not_exist")

    assert result is None


def test_get_all_documents(tmp_path):
    store = create_test_store(tmp_path)

    store.save(
        create_document(
            doc_id="doc_1",
            content_hash="hash_1",
        )
    )

    store.save(
        create_document(
            doc_id="doc_2",
            content_hash="hash_2",
        )
    )

    result = store.get_all()

    assert len(result) == 2
    assert result[0].doc_id == "doc_1"
    assert result[1].doc_id == "doc_2"


def test_update_document(tmp_path):
    store = create_test_store(tmp_path)

    document = create_document()

    store.save(document)

    updated = create_document(
        filename="updated.txt",
        stored_path="data/updated.txt",
        content_hash="new_hash",
        status="processing",
    )

    store.update(updated)

    result = store.get("doc_123")

    assert result is not None
    assert result.filename == "updated.txt"
    assert result.stored_path == "data/updated.txt"
    assert result.content_hash == "new_hash"
    assert result.status == "processing"


def test_delete_document(tmp_path):
    store = create_test_store(tmp_path)

    store.save(create_document())

    store.delete("doc_123")

    assert store.get("doc_123") is None


def test_get_by_content_hash(tmp_path):
    store = create_test_store(tmp_path)

    store.save(
        create_document(
            content_hash="abc123",
        )
    )

    result = store.get_by_content_hash("abc123")

    assert result is not None
    assert result.doc_id == "doc_123"
    assert result.content_hash == "abc123"
    assert result.status == "completed"


def test_get_by_content_hash_returns_none_for_unknown_hash(
    tmp_path,
):
    store = create_test_store(tmp_path)

    result = store.get_by_content_hash(
        "does_not_exist"
    )

    assert result is None


def test_get_by_content_hash_rejects_empty_hash(tmp_path):
    store = create_test_store(tmp_path)

    with pytest.raises(ValueError, match="content_hash"):
        store.get_by_content_hash("")


# ------------------------------------------------------------------
# Claim tests
# ------------------------------------------------------------------


def test_claim_creates_processing_document(tmp_path):
    store = create_test_store(tmp_path)

    document = create_document(
        doc_id="doc_claimed",
        content_hash="hash_claim",
        status="processing",
    )

    result = store.claim_document(document)

    assert result.doc_id == "doc_claimed"
    assert result.content_hash == "hash_claim"
    assert result.status == "processing"

    stored = store.get("doc_claimed")

    assert stored is not None
    assert stored.status == "processing"


def test_claim_existing_completed_document_returns_existing_document(
    tmp_path,
):
    store = create_test_store(tmp_path)

    existing = create_document(
        doc_id="doc_existing",
        filename="original.txt",
        content_hash="same_hash",
        status="completed",
    )

    store.save(existing)

    duplicate = create_document(
        doc_id="doc_new",
        filename="duplicate.txt",
        content_hash="same_hash",
        status="processing",
    )

    result = store.claim_document(duplicate)

    assert result.doc_id == "doc_existing"
    assert result.filename == "original.txt"
    assert result.content_hash == "same_hash"
    assert result.status == "completed"

    assert store.get("doc_new") is None


def test_claim_existing_processing_document_raises(
    tmp_path,
):
    store = create_test_store(tmp_path)

    existing = create_document(
        doc_id="doc_processing",
        content_hash="same_hash",
        status="processing",
    )

    store.save(existing)

    duplicate = create_document(
        doc_id="doc_new",
        content_hash="same_hash",
        status="processing",
    )

    with pytest.raises(
        DuplicateDocumentInProgressError,
        match="already being processed",
    ):
        store.claim_document(duplicate)

    assert store.get("doc_processing") is not None
    assert store.get("doc_new") is None


def test_mark_completed_changes_processing_to_completed(
    tmp_path,
):
    store = create_test_store(tmp_path)

    store.save(
        create_document(
            doc_id="doc_processing",
            content_hash="hash_123",
            status="processing",
        )
    )

    store.mark_completed("doc_processing")

    result = store.get("doc_processing")

    assert result is not None
    assert result.status == "completed"


def test_mark_completed_rejects_missing_document(
    tmp_path,
):
    store = create_test_store(tmp_path)

    with pytest.raises(
        ValueError,
        match="Document not found",
    ):
        store.mark_completed("does_not_exist")


def test_mark_completed_rejects_already_completed_document(
    tmp_path,
):
    store = create_test_store(tmp_path)

    store.save(
        create_document(
            doc_id="doc_completed",
            content_hash="hash_123",
            status="completed",
        )
    )

    with pytest.raises(
        ValueError,
        match="not processing",
    ):
        store.mark_completed("doc_completed")


def test_release_claim_removes_processing_document(
    tmp_path,
):
    store = create_test_store(tmp_path)

    store.save(
        create_document(
            doc_id="doc_processing",
            content_hash="hash_123",
            status="processing",
        )
    )

    store.release_claim("doc_processing")

    assert store.get("doc_processing") is None
    assert store.get_by_content_hash("hash_123") is None


def test_release_claim_does_nothing_for_missing_document(
    tmp_path,
):
    store = create_test_store(tmp_path)

    store.release_claim("does_not_exist")

    assert store.get("does_not_exist") is None


def test_release_claim_rejects_completed_document(
    tmp_path,
):
    store = create_test_store(tmp_path)

    store.save(
        create_document(
            doc_id="doc_completed",
            content_hash="hash_123",
            status="completed",
        )
    )

    with pytest.raises(
        ValueError,
        match="not processing",
    ):
        store.release_claim("doc_completed")