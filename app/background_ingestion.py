"""
Background document ingestion orchestration.

This module connects the ingestion job lifecycle with the existing
DocumentIngestionService.

The execution mechanism is intentionally kept independent from the
document ingestion pipeline so that a real worker system can replace
FastAPI BackgroundTasks later.
"""

from datetime import datetime
from pathlib import Path

from app.ingestion_job_store import IngestionJobStore
from app.ingestion_service import DocumentIngestionService


class BackgroundIngestionService:
    """
    Execute a document ingestion job in the background.
    """

    def __init__(
        self,
        ingestion_service: DocumentIngestionService,
        job_store: IngestionJobStore,
    ) -> None:
        self.ingestion_service = ingestion_service
        self.job_store = job_store

    def execute(
        self,
        job_id: str,
    ) -> None:
        """
        Execute one pending ingestion job.

        All execution failures are converted into a failed job state
        rather than being allowed to escape from the background task.
        """

        job = self.job_store.get(job_id)

        if job is None:
            return

        if job.status != "pending":
            return

        self.job_store.mark_processing(
            job_id=job_id,
            started_at=datetime.now(),
        )

        try:
            result = self.ingestion_service.ingest(
                file_path=Path(job.file_path),
                filename=job.filename,
            )

        except Exception as exc:
            self.job_store.mark_failed(
                job_id=job_id,
                error_message=str(exc),
                completed_at=datetime.now(),
            )

            return

        self.job_store.mark_completed(
            job_id=job_id,
            doc_id=result.doc_id,
            page_count=result.page_count,
            chunk_count=result.chunk_count,
            completed_at=datetime.now(),
        )
