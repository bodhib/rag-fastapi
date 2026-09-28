from collections.abc import Callable
from datetime import datetime
from pathlib import Path

from app.ingestion_job_store import IngestionJobStore
from app.ingestion_service import DocumentIngestionService


class BackgroundIngestionService:
    """
    Execute a document ingestion job in the background.

    Both the ingestion service and job store are created lazily,
    only when the background job actually executes.
    """

    def __init__(
        self,
        ingestion_service_factory: Callable[
            [], DocumentIngestionService
        ],
        job_store_factory: Callable[
            [], IngestionJobStore
        ],
    ) -> None:
        self.ingestion_service_factory = ingestion_service_factory
        self.job_store_factory = job_store_factory

    def execute(self, job_id: str) -> None:
        job_store = self.job_store_factory()

        job = job_store.get(job_id)

        if job is None:
            return

        if job.status != "pending":
            return

        job_store.mark_processing(
            job_id=job_id,
            started_at=datetime.now(),
        )

        try:
            ingestion_service = self.ingestion_service_factory()

            result = ingestion_service.ingest(
                file_path=Path(job.file_path),
                filename=job.filename,
            )

        except Exception as exc:
            job_store.mark_failed(
                job_id=job_id,
                error_message=str(exc),
                completed_at=datetime.now(),
            )
            return

        job_store.mark_completed(
            job_id=job_id,
            doc_id=result.doc_id,
            page_count=result.page_count,
            chunk_count=result.chunk_count,
            completed_at=datetime.now(),
        )