"""
Background document ingestion orchestration.

This module connects the ingestion job lifecycle with the existing
DocumentIngestionService.

The execution mechanism is intentionally kept independent from the
document ingestion pipeline so that a real worker system can replace
FastAPI BackgroundTasks later.
"""

from collections.abc import Callable
from datetime import datetime
from pathlib import Path

from app.ingestion_job_store import IngestionJobStore
from app.ingestion_service import DocumentIngestionService


class BackgroundIngestionService:
    """
    Execute a document ingestion job in the background.

    The ingestion service is created lazily so that heavy RAG
    components are not initialized while handling the upload request.
    """

    def __init__(
        self,
        ingestion_service_factory: Callable[
            [], DocumentIngestionService
        ],
        job_store: IngestionJobStore,
    ) -> None:
        self.ingestion_service_factory = ingestion_service_factory
        self.job_store = job_store

    def execute(
        self,
        job_id: str,
    ) -> None:
        """
        Execute one pending ingestion job.

        Heavy ingestion dependencies are initialized only when the
        background task actually starts.

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
            print(f"[INGESTION] Starting job {job_id}", flush=True)

            ingestion_service = self.ingestion_service_factory()

            print(
                f"[INGESTION] Ingestion service created for job {job_id}",
                flush=True,
            )


            result = ingestion_service.ingest(
                file_path=Path(job.file_path),
                filename=job.filename,
            )

            print(
                f"[INGESTION] Ingestion completed for job {job_id}",
                flush=True,
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