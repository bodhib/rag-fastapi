from datetime import datetime

import pytest

from app.chunker import DocumentChunk
from app.document_store import (
    DuplicateDocumentInProgressError,
    StoredDocument,
)
from app.ingestion_service import (
    DocumentIngestionService,
)


class FakeDocumentProcessor:
    def __init__(self):
        self.process_count = 0
        self.should_fail = False

    def process(self, file_path):
        self.process_count += 1

        if self.should_fail:
            raise RuntimeError("processing failed")

        return [
            {
                "page_number": 1,
                "text": "test content",
            }
        ]


class FakeChunker:
    def __init__(self):
        self.chunk_count = 0

    def chunk(self, doc_id, pages):
        self.chunk_count += 1

        return [
            DocumentChunk(
                doc_id=doc_id,
                text="test chunk",
                source="test.txt",
                page_number=1,
                chunk_index=0,
            )
        ]


class FakeEmbeddingService:
    def __init__(self):
        self.embed_count = 0
        self.should_fail = False

    def embed_chunks(self, chunks):
        self.embed_count += 1

        if self.should_fail:
            raise RuntimeError("embedding failed")

        return chunks


class FakeVectorStore:
    def __init__(self):
        self.add_count = 0
        self.chunks = []
        self.delete_count = 0
        self.deleted_doc_ids = []

    def add_chunks(self, chunks):
        self.add_count += 1
        self.chunks.extend(chunks)

    def delete_document(self, doc_id):
        self.delete_count += 1
        self.deleted_doc_ids.append(doc_id)
        self.chunks = [
            chunk
            for chunk in self.chunks
            if chunk.doc_id != doc_id
        ]


class FakeRetrievalIndexManager:
    def __init__(self):
        self.rebuild_count = 0
        self.remove_count = 0
        self.removed_doc_ids = []
        self.should_fail = False

    def rebuild(self, chunks):
        self.rebuild_count += 1

        if self.should_fail:
            raise RuntimeError("lexical build failed")

        return len(chunks)

    def remove_document(self, doc_id):
        self.remove_count += 1
        self.removed_doc_ids.append(doc_id)


class FakeDocumentStore:
    def __init__(self):
        self.saved_documents = []
        self.documents_by_hash = {}

        self.claim_count = 0
        self.mark_completed_count = 0
        self.release_claim_count = 0

        self.claimed_documents = {}

    @staticmethod
    def create_timestamp():
        return datetime.now()

    def save(self, document):
        self.saved_documents.append(document)
        self.documents_by_hash[document.content_hash] = document

    def get_by_content_hash(self, content_hash):
        return self.documents_by_hash.get(content_hash)

    def claim_document(self, document):
        self.claim_count += 1

        existing = self.documents_by_hash.get(
            document.content_hash
        )

        if existing is not None:
            if existing.status == "completed":
                return existing

            if existing.status == "processing":
                raise DuplicateDocumentInProgressError(
                    "Document with the same content is already "
                    "being processed."
                )

        document.status = "processing"

        self.documents_by_hash[
            document.content_hash
        ] = document

        self.claimed_documents[
            document.doc_id
        ] = document

        return document

    def mark_completed(self, doc_id):
        self.mark_completed_count += 1

        document = self.claimed_documents[doc_id]
        document.status = "completed"

    def release_claim(self, doc_id):
        self.release_claim_count += 1

        document = self.claimed_documents.pop(doc_id)

        self.documents_by_hash.pop(
            document.content_hash,
            None,
        )


def create_service(
    vector_store=None,
    retrieval_index_manager=None,
    document_store=None,
):
    processor = FakeDocumentProcessor()
    chunker = FakeChunker()
    embedding_service = FakeEmbeddingService()

    vector_store = vector_store or FakeVectorStore()
    retrieval_index_manager = (
        retrieval_index_manager
        or FakeRetrievalIndexManager()
    )
    document_store = (
        document_store
        or FakeDocumentStore()
    )

    service = DocumentIngestionService(
        document_processor=processor,
        chunker=chunker,
        embedding_service=embedding_service,
        vector_store=vector_store,
        retrieval_index_manager=retrieval_index_manager,
        document_store=document_store,
    )

    return (
        service,
        processor,
        embedding_service,
        document_store,
    )


def test_ingest_runs_complete_pipeline(tmp_path):
    file_path = tmp_path / "test.txt"
    file_path.write_text("hello world")

    (
        service,
        processor,
        embedding_service,
        document_store,
    ) = create_service()

    result = service.ingest(
        file_path=file_path,
        filename="test.txt",
    )

    assert result.filename == "test.txt"
    assert result.page_count == 1
    assert result.chunk_count == 1

    assert processor.process_count == 1
    assert embedding_service.embed_count == 1

    assert document_store.claim_count == 1
    assert document_store.mark_completed_count == 1
    assert document_store.release_claim_count == 0


def test_ingest_uses_supplied_filename(tmp_path):
    file_path = tmp_path / "actual.txt"
    file_path.write_text("hello")

    (
        service,
        _,
        _,
        document_store,
    ) = create_service()

    result = service.ingest(
        file_path=file_path,
        filename="supplied-name.txt",
    )

    assert result.filename == "supplied-name.txt"

    stored = document_store.claimed_documents[
        result.doc_id
    ]

    assert stored.filename == "supplied-name.txt"
    assert stored.status == "completed"

def test_ingest_rejects_missing_file(tmp_path):
    service, _, _, document_store = create_service()

    file_path = tmp_path / "missing.txt"

    with pytest.raises(
        FileNotFoundError,
        match="Document not found",
    ):
        service.ingest(file_path)

    assert document_store.claim_count == 0


def test_ingest_does_not_leave_claim_when_pipeline_fails(
    tmp_path,
):
    file_path = tmp_path / "test.txt"
    file_path.write_text("hello world")

    service, processor, _, document_store = create_service()

    processor.should_fail = True

    with pytest.raises(
        RuntimeError,
        match="processing failed",
    ):
        service.ingest(
            file_path=file_path,
            filename="test.txt",
        )

    assert processor.process_count == 1
    assert document_store.claim_count == 1
    assert document_store.mark_completed_count == 0
    assert document_store.release_claim_count == 1

    assert len(document_store.documents_by_hash) == 0


def test_ingest_calculates_and_persists_content_hash(
    tmp_path,
):
    file_path = tmp_path / "test.txt"
    file_path.write_text("hello world")

    service, _, _, document_store = create_service()

    service.ingest(
        file_path=file_path,
        filename="test.txt",
    )

    assert len(document_store.documents_by_hash) == 1

    stored = next(
        iter(document_store.documents_by_hash.values())
    )

    assert stored.content_hash is not None
    assert len(stored.content_hash) == 64
    assert stored.status == "completed"


def test_ingest_duplicate_returns_existing_document(
    tmp_path,
):
    file_path = tmp_path / "test.txt"
    file_path.write_text("same content")

    service, processor, _, document_store = create_service()

    first_result = service.ingest(
        file_path=file_path,
        filename="first.txt",
    )

    process_count_after_first = (
        processor.process_count
    )

    second_result = service.ingest(
        file_path=file_path,
        filename="second.txt",
    )

    assert second_result.doc_id == first_result.doc_id
    assert second_result.filename == "first.txt"

    assert (
        processor.process_count
        == process_count_after_first
    )

    assert document_store.claim_count == 2
    assert document_store.mark_completed_count == 1


def test_ingest_duplicate_skips_pipeline(
    tmp_path,
):
    file_path = tmp_path / "test.txt"
    file_path.write_text("duplicate content")

    service, processor, embedding_service, _ = (
        create_service()
    )

    service.ingest(file_path)

    processor_count = processor.process_count
    embedding_count = embedding_service.embed_count

    service.ingest(file_path)

    assert processor.process_count == processor_count
    assert embedding_service.embed_count == embedding_count


def test_ingest_same_filename_with_different_content_creates_new_document(
    tmp_path,
):
    file_path = tmp_path / "test.txt"

    file_path.write_text("content A")

    service, processor, _, document_store = (
        create_service()
    )

    first_result = service.ingest(
        file_path=file_path,
        filename="test.txt",
    )

    file_path.write_text("content B")

    second_result = service.ingest(
        file_path=file_path,
        filename="test.txt",
    )

    assert first_result.doc_id != second_result.doc_id
    assert processor.process_count == 2
    assert len(document_store.documents_by_hash) == 2


def test_ingest_duplicate_in_progress_is_rejected(
    tmp_path,
):
    file_path = tmp_path / "test.txt"
    file_path.write_text("same content")

    document_store = FakeDocumentStore()

    (
        service,
        processor,
        _,
        _,
    ) = create_service(
        document_store=document_store,
    )

    # Simulate another ingestion request already owning
    # the claim.
    content_hash = (
        service._calculate_content_hash(file_path)
    )

    existing = StoredDocument(
        doc_id="doc_existing",
        filename="existing.txt",
        stored_path=str(file_path),
        created_at=datetime.now(),
        content_hash=content_hash,
        status="processing",
    )

    document_store.documents_by_hash[
        content_hash
    ] = existing

    with pytest.raises(
        DuplicateDocumentInProgressError,
        match="already being processed",
    ):
        service.ingest(
            file_path=file_path,
            filename="duplicate.txt",
        )

    assert processor.process_count == 0
    assert document_store.claim_count == 1
    assert document_store.mark_completed_count == 0
    assert document_store.release_claim_count == 0


def test_ingest_marks_document_completed_after_success(
    tmp_path,
):
    file_path = tmp_path / "test.txt"
    file_path.write_text("hello")

    service, _, _, document_store = create_service()

    result = service.ingest(file_path)

    stored = document_store.get_by_content_hash(
        next(
            iter(document_store.documents_by_hash)
        )
    )

    assert stored is not None
    assert stored.doc_id == result.doc_id
    assert stored.status == "completed"


def test_ingest_releases_claim_when_embedding_fails(
    tmp_path,
):
    file_path = tmp_path / "test.txt"
    file_path.write_text("hello")

    service, _, embedding_service, document_store = (
        create_service()
    )

    embedding_service.should_fail = True

    with pytest.raises(
        RuntimeError,
        match="embedding failed",
    ):
        service.ingest(file_path)

    assert document_store.claim_count == 1
    assert document_store.mark_completed_count == 0
    assert document_store.release_claim_count == 1
    assert len(document_store.documents_by_hash) == 0

def test_ingest_cleans_vector_store_when_lexical_build_fails(
    tmp_path,
):
    file_path = tmp_path / "test.txt"

    file_path.write_text(
        "hello world"
    )

    vector_store = FakeVectorStore()

    retrieval_manager = (
        FakeRetrievalIndexManager()
    )

    retrieval_manager.should_fail = True

    (
        service,
        _,
        _,
        document_store,
    ) = create_service(
        vector_store=vector_store,
        retrieval_index_manager=retrieval_manager,
    )

    with pytest.raises(
        RuntimeError,
        match="lexical build failed",
    ):
        service.ingest(file_path)

    assert vector_store.delete_count == 1

    assert (
        len(
            vector_store.deleted_doc_ids
        )
        == 1
    )

    assert (
        document_store.release_claim_count
        == 1
    )    

def test_cleanup_failure_does_not_hide_original_error(
    tmp_path,
):
    file_path = tmp_path / "test.txt"

    file_path.write_text(
        "hello world"
    )

    class BrokenVectorStore(
        FakeVectorStore,
    ):
        def delete_document(
            self,
            doc_id,
        ):
            raise RuntimeError(
                "cleanup failed"
            )

    vector_store = BrokenVectorStore()

    retrieval_manager = (
        FakeRetrievalIndexManager()
    )

    retrieval_manager.should_fail = True

    (
        service,
        _,
        _,
        document_store,
    ) = create_service(
        vector_store=vector_store,
        retrieval_index_manager=retrieval_manager,
    )

    with pytest.raises(
        RuntimeError,
        match="lexical build failed",
    ):
        service.ingest(file_path)

    assert (
        document_store.release_claim_count
        == 1
    )

def test_success_does_not_trigger_cleanup(
    tmp_path,
):
    file_path = tmp_path / "test.txt"

    file_path.write_text(
        "hello world"
    )

    vector_store = FakeVectorStore()

    retrieval_manager = (
        FakeRetrievalIndexManager()
    )

    (
        service,
        _,
        _,
        document_store,
    ) = create_service(
        vector_store=vector_store,
        retrieval_index_manager=retrieval_manager,
    )

    service.ingest(file_path)

    assert vector_store.delete_count == 0

    assert (
        retrieval_manager.remove_count
        == 0
    )

    assert (
        document_store.release_claim_count
        == 0
    )

    assert (
        document_store.mark_completed_count
        == 1
    )        