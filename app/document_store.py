"""
Persistence layer for document metadata.

DocumentStore is responsible for reading and writing document metadata
to the SQL database. It also provides the database-backed claim mechanism
used to prevent concurrent ingestion of identical document content.
"""

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime

from sqlalchemy import delete, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.models.document import DocumentModel


@dataclass
class StoredDocument:
    doc_id: str
    filename: str
    stored_path: str
    created_at: datetime
    content_hash: str | None = None
    status: str = "completed"


class DuplicateDocumentInProgressError(Exception):
    """Raised when identical document content is already being ingested."""


class DocumentStore:
    def __init__(
        self,
        session_factory: Callable[[], Session],
    ) -> None:
        self.session_factory = session_factory

    def save(self, document: StoredDocument) -> None:
        """
        Persist a document record.

        Normally new documents should enter the database through
        claim_document(). This method remains available for ordinary
        document persistence and existing callers/tests.
        """
        with self.session_factory() as session:
            session.add(
                DocumentModel(
                    doc_id=document.doc_id,
                    filename=document.filename,
                    stored_path=document.stored_path,
                    created_at=document.created_at,
                    content_hash=document.content_hash,
                    status=document.status,
                )
            )
            session.commit()

    def claim_document(
        self,
        document: StoredDocument,
    ) -> StoredDocument:
        """
        Atomically claim a document for ingestion.

        Possible outcomes:

        1. No document with the same content hash exists:
           create a processing document and return it.

        2. A completed document already exists:
           return the existing document.

        3. A processing document already exists:
           raise DuplicateDocumentInProgressError.

        The database UNIQUE constraint on content_hash protects
        against race conditions between concurrent requests.
        """
        if not document.content_hash:
            raise ValueError(
                "content_hash is required to claim a document"
            )

        with self.session_factory() as session:
            existing = self._get_model_by_content_hash(
                session=session,
                content_hash=document.content_hash,
            )

            if existing is not None:
                return self._handle_existing_document(existing)

            try:
                with session.begin_nested():
                    session.add(
                        DocumentModel(
                            doc_id=document.doc_id,
                            filename=document.filename,
                            stored_path=document.stored_path,
                            created_at=document.created_at,
                            content_hash=document.content_hash,
                            status="processing",
                        )
                    )

                    # Force the INSERT now so a possible UNIQUE
                    # constraint violation occurs inside the savepoint.
                    session.flush()

            except IntegrityError:
                # Another concurrent request may have inserted the same
                # content hash after our initial SELECT.
                existing = self._get_model_by_content_hash(
                    session=session,
                    content_hash=document.content_hash,
                )

                if existing is None:
                    raise

                return self._handle_existing_document(existing)

            session.commit()

            return StoredDocument(
                doc_id=document.doc_id,
                filename=document.filename,
                stored_path=document.stored_path,
                created_at=document.created_at,
                content_hash=document.content_hash,
                status="processing",
            )

    def mark_completed(self, doc_id: str) -> None:
        """
        Mark a previously claimed document as successfully ingested.
        """
        with self.session_factory() as session:
            document = session.get(DocumentModel, doc_id)

            if document is None:
                raise ValueError(
                    f"Document not found: {doc_id}"
                )

            if document.status != "processing":
                raise ValueError(
                    f"Document is not processing: {doc_id}"
                )

            document.status = "completed"
            session.commit()

    def release_claim(self, doc_id: str) -> None:
        """
        Release a processing claim after ingestion failure.

        Only a document currently in processing state is deleted.
        """
        with self.session_factory() as session:
            document = session.get(DocumentModel, doc_id)

            if document is None:
                return

            if document.status != "processing":
                raise ValueError(
                    f"Document is not processing: {doc_id}"
                )

            session.delete(document)
            session.commit()

    def get(self, doc_id: str) -> StoredDocument | None:
        with self.session_factory() as session:
            document = session.get(DocumentModel, doc_id)

            if document is None:
                return None

            return self._to_stored_document(document)

    def get_by_content_hash(
        self,
        content_hash: str,
    ) -> StoredDocument | None:
        if not content_hash:
            raise ValueError("content_hash cannot be empty")

        with self.session_factory() as session:
            document = self._get_model_by_content_hash(
                session=session,
                content_hash=content_hash,
            )

            if document is None:
                return None

            return self._to_stored_document(document)

    def get_all(self) -> list[StoredDocument]:
        with self.session_factory() as session:
            documents = session.execute(
                select(DocumentModel).order_by(
                    DocumentModel.created_at
                )
            ).scalars().all()

            return [
                self._to_stored_document(document)
                for document in documents
            ]

    def update(self, document: StoredDocument) -> None:
        with self.session_factory() as session:
            existing = session.get(
                DocumentModel,
                document.doc_id,
            )

            if existing is None:
                raise ValueError(
                    f"Document not found: {document.doc_id}"
                )

            existing.filename = document.filename
            existing.stored_path = document.stored_path
            existing.created_at = document.created_at
            existing.content_hash = document.content_hash
            existing.status = document.status

            session.commit()

    def delete(self, doc_id: str) -> None:
        with self.session_factory() as session:
            document = session.get(DocumentModel, doc_id)

            if document is None:
                return

            session.execute(
                delete(DocumentModel).where(
                    DocumentModel.doc_id == doc_id
                )
            )
            session.commit()

    @staticmethod
    def create_timestamp() -> datetime:
        return datetime.now()

    @staticmethod
    def _get_model_by_content_hash(
        session: Session,
        content_hash: str,
    ) -> DocumentModel | None:
        return session.execute(
            select(DocumentModel).where(
                DocumentModel.content_hash == content_hash
            )
        ).scalar_one_or_none()

    @staticmethod
    def _handle_existing_document(
        document: DocumentModel,
    ) -> StoredDocument:
        if document.status == "completed":
            return DocumentStore._to_stored_document(document)

        if document.status == "processing":
            raise DuplicateDocumentInProgressError(
                "Document with the same content is already "
                "being processed."
            )

        raise ValueError(
            f"Unsupported document status: {document.status}"
        )

    @staticmethod
    def _to_stored_document(
        document: DocumentModel,
    ) -> StoredDocument:
        return StoredDocument(
            doc_id=document.doc_id,
            filename=document.filename,
            stored_path=document.stored_path,
            created_at=document.created_at,
            content_hash=document.content_hash,
            status=document.status,
        )