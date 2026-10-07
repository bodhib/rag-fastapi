"""
Document ingestion orchestration.

This module coordinates document processing, chunking, embedding,
vector storage, lexical indexing, and document metadata persistence.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from uuid import uuid4

from app.chunker import RecursiveChunker
from app.document_processor import DocumentProcessor
from app.document_store import (
    DocumentStore,
    DuplicateDocumentInProgressError,
    StoredDocument,
)
from app.retrieval_index import RetrievalIndexManager

from app.candidate_store import CandidateStore

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.embeddings import EmbeddingService
    from app.vector_store import VectorStore


@dataclass
class IngestionResult:
    doc_id: str
    filename: str
    page_count: int
    chunk_count: int


class DocumentIngestionService:
    def __init__(
        self,
        document_processor: DocumentProcessor,
        chunker: RecursiveChunker,
        embedding_service: EmbeddingService,
        vector_store: VectorStore,
        retrieval_index_manager: RetrievalIndexManager,
        document_store: DocumentStore,
        candidate_store: CandidateStore,
    ) -> None:
        self.document_processor = document_processor
        self.chunker = chunker
        self.embedding_service = embedding_service
        self.vector_store = vector_store
        self.retrieval_index_manager = retrieval_index_manager
        self.document_store = document_store
        self.candidate_store = candidate_store

    def ingest(
        self,
        file_path: Path,
        filename: str | None = None,
        knowledge_base_id: str = "kb_default",
    ) -> IngestionResult:
        if not file_path.exists():
            raise FileNotFoundError(
                f"Document not found: {file_path}"
            )

        if not file_path.is_file():
            raise ValueError(
                f"Path is not a file: {file_path}"
            )

        original_filename = filename or file_path.name

        if not original_filename.strip():
            raise ValueError("filename cannot be empty")

        content_hash = self._calculate_content_hash(file_path)

        candidate_document = StoredDocument(
            doc_id=self._generate_doc_id(),
            filename=original_filename,
            stored_path=str(file_path),
            created_at=self.document_store.create_timestamp(),
            knowledge_base_id=knowledge_base_id,
            content_hash=content_hash,
            status="processing",
        )

        claimed_document = self.document_store.claim_document(
            candidate_document
        )

        # A completed document with the same content already exists.
        # No processing is necessary.
        if claimed_document.status == "completed":
            return self._build_existing_result(
                claimed_document
            )

        vector_store_written = False
        lexical_operation_started = False
        lexical_index_written = False

        try:
            pages = self.document_processor.process(file_path)

            chunks = self.chunker.chunk(
                doc_id=claimed_document.doc_id,
                pages=pages,
                knowledge_base_id=knowledge_base_id,
            )

            embedded_chunks = self.embedding_service.embed_chunks(chunks)

            self.vector_store.add_chunks(embedded_chunks)
            vector_store_written = True

            lexical_operation_started = True

            self.retrieval_index_manager.rebuild(chunks)

            lexical_index_written = True

            self.document_store.mark_completed(
                claimed_document.doc_id
            )

            self.candidate_store.attach_resume_document(
                knowledge_base_id=knowledge_base_id,
                resume_filename=claimed_document.filename,
                resume_document_id=claimed_document.doc_id,
            )

        except Exception:
            try:
                if vector_store_written:
                    try:
                        self.vector_store.delete_document(
                            claimed_document.doc_id
                        )
                    except Exception:
                        pass

                if lexical_index_written:
                    try:
                        self.retrieval_index_manager.remove_document(
                            claimed_document.doc_id
                        )
                    except Exception:
                        pass

                elif lexical_operation_started:
                    try:
                        remaining_chunks = (
                            self.vector_store.get_all_chunks()
                        )

                        self.retrieval_index_manager.rebuild(
                            remaining_chunks
                        )
                    except Exception:
                        pass

            finally:
                self.document_store.release_claim(
                    claimed_document.doc_id
                )

            raise

        return IngestionResult(
            doc_id=claimed_document.doc_id,
            filename=claimed_document.filename,
            page_count=len(pages),
            chunk_count=len(chunks),
        )

    @staticmethod
    def _calculate_content_hash(
        file_path: Path,
    ) -> str:
        sha256 = hashlib.sha256()

        with file_path.open("rb") as file:
            while chunk := file.read(1024 * 1024):
                sha256.update(chunk)

        return sha256.hexdigest()

    @staticmethod
    def _generate_doc_id() -> str:
        return f"doc_{uuid4().hex}"

    @staticmethod
    def _build_existing_result(
        document: StoredDocument,
    ) -> IngestionResult:
        return IngestionResult(
            doc_id=document.doc_id,
            filename=document.filename,
            page_count=0,
            chunk_count=0,
        )