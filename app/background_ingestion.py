from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.ingestion_job_store import IngestionJobStore
    from app.ingestion_service import DocumentIngestionService


class BackgroundIngestionService:
    """
    Execute an ingestion job in the background.

    Heavy ingestion components are not imported or initialized
    while FastAPI resolves this dependency. They are created only
   
when execute() actually runs.
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
        print(f"[INGESTION] Starting job {job_id}", flush=True)

        job_store = self.job_store_factory()

        print(f"[INGESTION] Job store ready: {job_id}", flush=True)

        job = job_store.get(job_id)

        if job is None:
            print(f"[INGESTION] Job not found: {job_id}", flush=True)
            return

        if job.status != "pending":
            print(
                f"[INGESTION] Job {job_id} has status {job.status}; skipping",
                flush=True,
            )
            return

        job_store.mark_processing(
            job_id=job_id,
            started_at=datetime.now(),
        )

        print(f"[INGESTION] Job marked processing: {job_id}", flush=True)

        try:
            print(
                f"[INGESTION] About to call ingestion_service_factory: {job_id}",
                flush=True,
            )

            print(
                f"[INGESTION] Factory object: {self.ingestion_service_factory!r}",
                flush=True,
            )

            print(
                f"[INGESTION] Factory module: "
                f"{getattr(self.ingestion_service_factory, '__module__', None)}",
                flush=True,
            )

            print(
                f"[INGESTION] Factory name: "
                f"{getattr(self.ingestion_service_factory, '__name__', None)}",
                flush=True,
            )

            print(
                f"[INGESTION] Calling ingestion_service_factory NOW: {job_id}",
                flush=True,
            )

            ingestion_service = self.ingestion_service_factory()

            print(
                f"[INGESTION] ingestion_service_factory RETURNED: {job_id}",
                flush=True,
            )

            print(
                f"[INGESTION] Ingestion service type: "
                f"{type(ingestion_service)!r}",
                flush=True,
            )

            print(
                f"[INGESTION] Starting ingest: {job_id}",
                flush=True,
            )

            result = ingestion_service.ingest(
                file_path=Path(job.file_path),
                filename=job.filename,
            )

            print(
                f"[INGESTION] Ingest finished: {job_id}",
                flush=True,
            )

        except Exception as exc:
            print(
                f"[INGESTION] Ingestion failed: {job_id}: {exc!r}",
                flush=True,
            )

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

        print(
            f"[INGESTION] Job completed: {job_id}",
            flush=True,
        )