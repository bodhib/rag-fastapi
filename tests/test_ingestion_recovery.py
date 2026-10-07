from datetime import datetime
from pathlib import Path

import pytest

from app.chunker import DocumentChunk
from app.document_store import StoredDocument
from app.ingestion_service import DocumentIngestionService


class FakeDocumentProcessor:
    def process(self, file_path):
        return [
            {
                "page_number": 1,
                "text": "test document content",
            }
        ]


class FakeChunker:
    def chunk(
        self,
        doc_id,
        pages,
        knowledge_base_id="kb_default",
    ):
        return [
            DocumentChunk(
                doc_id=doc_id,
                text="chunk one",
                source="test.txt",
                page_number=1,
                chunk_index=0,
                knowledge_base_id=knowledge_base_id,
            ),
            DocumentChunk(
                doc_id=doc_id,
                text="chunk two",
                source="test.txt",
                page_number=1,
                chunk_index=1,
                knowledge_base_id=knowledge_base_id,
            ),
        ]


class FakeEmbeddingService:
    def embed_chunks(self, chunks):
        # For this recovery test we do not need real embeddings.
        # Returning the chunks themselves is sufficient because the
        # fake vector store only needs to track which document was added.
        return chunks


class FakeVectorStore:
    def __init__(self):
        self.added_chunks = []
        self.deleted_doc_ids = []

    def add_chunks(self, embedded_chunks):
        self.added_chunks.extend(embedded_chunks)

    def delete_document(self, doc_id):
        self.deleted_doc_ids.append(doc_id)

        self.added_chunks = [
            chunk
            for chunk in self.added_chunks
            if chunk.doc_id != doc_id
        ]

    def get_all_chunks(self):
        return list(self.added_chunks)


class FakeRetrievalIndexManager:
    def __init__(self):
        self.rebuild_calls = []
        self.remove_calls = []
        self.fail_first_rebuild = False
        self.partial_chunks = []

    def rebuild(self, chunks):
        chunks = list(chunks)

        self.rebuild_calls.append(chunks)

        if self.fail_first_rebuild and len(self.rebuild_calls) == 1:
            # Simulate a lexical-index mutation occurring before
            # the rebuild operation fails.
            self.partial_chunks.extend(chunks[:1])

            raise RuntimeError(
                "lexical index failure"
            )

        return len(chunks)

    def remove_document(self, doc_id):
        self.remove_calls.append(doc_id)


class FakeDocumentStore:
    def __init__(self):
        self.documents = {}
        self.claimed_documents = {}
        self.release_claim_calls = []

    @staticmethod
    def create_timestamp():
        return datetime.now()

    def claim_document(self, document):
        self.documents[document.doc_id] = document
        self.claimed_documents[document.doc_id] = document

        return document

    def mark_completed(self, doc_id):
        self.documents[doc_id].status = "completed"

    def release_claim(self, doc_id):
        self.release_claim_calls.append(doc_id)

        document = self.claimed_documents.pop(
            doc_id,
            None,
        )

        if document is not None:
            self.documents.pop(
                doc_id,
                None,
            )


def _build_service(
    *,
    document_store,
    vector_store,
    retrieval_index_manager,
):
    return DocumentIngestionService(
        document_processor=FakeDocumentProcessor(),
        chunker=FakeChunker(),
        embedding_service=FakeEmbeddingService(),
        vector_store=vector_store,
        retrieval_index_manager=retrieval_index_manager,
        document_store=document_store,
    )


def test_ingestion_failure_compensates_vector_and_lexical_state(
    tmp_path: Path,
):
    source_file = tmp_path / "test.txt"

    source_file.write_text(
        "test document content",
        encoding="utf-8",
    )

    document_store = FakeDocumentStore()
    vector_store = FakeVectorStore()
    retrieval_index_manager = FakeRetrievalIndexManager()

    retrieval_index_manager.fail_first_rebuild = True

    service = _build_service(
        document_store=document_store,
        vector_store=vector_store,
        retrieval_index_manager=retrieval_index_manager,
    )

    with pytest.raises(
        RuntimeError,
        match="lexical index failure",
    ):
        service.ingest(
            file_path=source_file,
            filename="test.txt",
            knowledge_base_id="kb_default",
        )

    # The vector store initially received both chunks.
    assert len(retrieval_index_manager.rebuild_calls[0]) == 2

    # The failed ingestion must remove the document's vectors.
    assert len(vector_store.deleted_doc_ids) == 1

    doc_id = vector_store.deleted_doc_ids[0]

    assert doc_id.startswith("doc_")

    # After vector cleanup, there should be no remaining chunks.
    assert vector_store.get_all_chunks() == []

    # The recovery path rebuilds the lexical index from the
    # remaining vector-store contents.
    assert len(retrieval_index_manager.rebuild_calls) == 2

    assert retrieval_index_manager.rebuild_calls[1] == []

    # The ingestion claim must always be released.
    assert document_store.release_claim_calls == [doc_id]


def test_ingestion_recovers_from_partial_lexical_mutation(
    tmp_path: Path,
):
    source_file = tmp_path / "test.txt"

    source_file.write_text(
        "test document content",
        encoding="utf-8",
    )

    document_store = FakeDocumentStore()
    vector_store = FakeVectorStore()
    retrieval_index_manager = FakeRetrievalIndexManager()

    retrieval_index_manager.fail_first_rebuild = True

    service = _build_service(
        document_store=document_store,
        vector_store=vector_store,
        retrieval_index_manager=retrieval_index_manager,
    )

    with pytest.raises(
        RuntimeError,
        match="lexical index failure",
    ):
        service.ingest(
            file_path=source_file,
            filename="test.txt",
            knowledge_base_id="kb_default",
        )

    # The first rebuild partially mutated the lexical index
    # before raising the error.
    assert len(
        retrieval_index_manager.partial_chunks
    ) == 1

    # The recovery path performs a second rebuild using the
    # current contents of the vector store.
    assert len(
        retrieval_index_manager.rebuild_calls
    ) == 2

    assert retrieval_index_manager.rebuild_calls[0]

    assert retrieval_index_manager.rebuild_calls[1] == []

    # Vector-store compensation happened.
    assert len(vector_store.deleted_doc_ids) == 1

    doc_id = vector_store.deleted_doc_ids[0]

    # The ingestion claim was released.
    assert document_store.release_claim_calls == [doc_id]

