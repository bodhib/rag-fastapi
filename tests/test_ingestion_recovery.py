from pathlib import Path

import pytest

from app.ingestion_service import DocumentIngestionService
from app.document_store import DuplicateDocumentInProgressError


class FailingRetrievalIndexManager:
    def __init__(self) -> None:
        self.removed_doc_ids: list[str] = []

    def rebuild(self, chunks) -> int:
        raise RuntimeError("Simulated lexical index failure")

    def remove_document(self, doc_id: str) -> None:
        self.removed_doc_ids.append(doc_id)

class PartiallyFailingRetrievalIndexManager:
    def __init__(self) -> None:
        self.rebuild_calls = 0
        self.chunks = []

    def rebuild(self, chunks) -> int:
        self.rebuild_calls += 1

        if self.rebuild_calls == 1:
            # Simulate a lexical rebuild that partially
            # modifies the index and then fails.
            self.chunks = list(chunks[:1])

            raise RuntimeError(
                "Simulated partial lexical index failure"
            )

        # Recovery rebuild.
        self.chunks = list(chunks)

        return len(chunks)

    def remove_document(self, doc_id: str) -> None:
        self.chunks = [
            chunk
            for chunk in self.chunks
            if chunk.doc_id != doc_id
        ]

class RecordingVectorStore:
    def __init__(self, chunks=None) -> None:
        self.chunks = list(chunks or [])
        self.added_doc_ids: list[str] = []
        self.deleted_doc_ids: list[str] = []

    def add_chunks(self, embedded_chunks) -> None:
        for item in embedded_chunks:
            self.added_doc_ids.append(item.chunk.doc_id)
            self.chunks.append(item.chunk)

    def delete_document(self, doc_id: str) -> None:
        self.deleted_doc_ids.append(doc_id)

        self.chunks = [
            chunk
            for chunk in self.chunks
            if chunk.doc_id != doc_id
        ]

    def get_all_chunks(self):
        return list(self.chunks)
    

class FakeProcessor:
    def process(self, file_path: Path):
        return [
            type(
                "Page",
                (),
                {
                    "page_number": 1,
                    "text": "Failure recovery test document.",
                },
            )()
        ]


class FakeChunker:
    def chunk(self, doc_id, pages):
        return [
            type(
                "Chunk",
                (),
                {
                    "doc_id": doc_id,
                    "text": "Failure recovery test document.",
                    "source": "failure-recovery.txt",
                    "page_number": 1,
                    "chunk_index": 0,
                },
            )()
        ]


class FakeEmbeddingService:
    def embed_chunks(self, chunks):
        return [
            type(
                "EmbeddedChunk",
                (),
                {
                    "chunk": chunk,
                    "embedding": [0.1, 0.2, 0.3],
                },
            )()
            for chunk in chunks
        ]


class FakeDocumentStore:
    def __init__(self) -> None:
        self.documents = {}
        self.released_doc_ids: list[str] = []

    def create_timestamp(self):
        from datetime import datetime
        return datetime.now()

    def claim_document(self, document):
        self.documents[document.doc_id] = document
        return document

    def mark_completed(self, doc_id):
        raise AssertionError(
            "mark_completed should not be reached"
        )

    def release_claim(self, doc_id):
        self.released_doc_ids.append(doc_id)
        self.documents.pop(doc_id, None)


def test_ingestion_failure_compensates_vector_and_lexical_state(
    tmp_path,
):
    source_file = tmp_path / "failure-recovery.txt"
    source_file.write_text(
        "Failure recovery test document.",
        encoding="utf-8",
    )

    vector_store = RecordingVectorStore()
    retrieval_manager = FailingRetrievalIndexManager()
    document_store = FakeDocumentStore()

    service = DocumentIngestionService(
        document_processor=FakeProcessor(),
        chunker=FakeChunker(),
        embedding_service=FakeEmbeddingService(),
        vector_store=vector_store,
        retrieval_index_manager=retrieval_manager,
        document_store=document_store,
    )

    with pytest.raises(
        RuntimeError,
        match="Simulated lexical index failure",
    ):
        service.ingest(source_file)

    assert len(vector_store.added_doc_ids) == 1
    assert vector_store.deleted_doc_ids == vector_store.added_doc_ids

    assert len(retrieval_manager.removed_doc_ids) == 0

    assert len(document_store.released_doc_ids) == 1
    assert document_store.documents == {}

def test_ingestion_recovers_from_partial_lexical_mutation(
    tmp_path,
):
    source_file = tmp_path / "partial-recovery.txt"
    source_file.write_text(
        "Partial lexical recovery test document.",
        encoding="utf-8",
    )

    existing_chunk = type(
        "Chunk",
        (),
        {
            "doc_id": "existing-doc",
            "text": "Existing document content.",
            "source": "existing.txt",
            "page_number": 1,
            "chunk_index": 0,
        },
    )()

    vector_store = RecordingVectorStore(
        chunks=[existing_chunk]
    )

    retrieval_manager = (
        PartiallyFailingRetrievalIndexManager()
    )

    document_store = FakeDocumentStore()

    service = DocumentIngestionService(
        document_processor=FakeProcessor(),
        chunker=FakeChunker(),
        embedding_service=FakeEmbeddingService(),
        vector_store=vector_store,
        retrieval_index_manager=retrieval_manager,
        document_store=document_store,
    )

    with pytest.raises(
        RuntimeError,
        match="Simulated partial lexical index failure",
    ):
        service.ingest(source_file)

    new_doc_id = vector_store.added_doc_ids[0]

    # The newly ingested document must be removed
    # from the vector store during compensation.
    assert vector_store.deleted_doc_ids == [new_doc_id]

    # The lexical rebuild should have happened twice:
    #
    # 1. Initial rebuild -> partially mutates and fails
    # 2. Recovery rebuild -> succeeds
    assert retrieval_manager.rebuild_calls == 2

    # After recovery, the lexical index must contain
    # exactly the chunks that remain in the authoritative
    # vector store.
    assert retrieval_manager.chunks == [
        existing_chunk
    ]

    # The failed document claim must be released.
    assert document_store.released_doc_ids == [new_doc_id]

    assert document_store.documents == {}    