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

    Production code can provide factories so heavy ingestion components
    are created only when execute() runs.

    Tests may provide already-created ingestion_service and job_store
    instances for lightweight dependency injection.
    """

    def __init__(
        self,
        ingestion_service: DocumentIngestionService | None = None,
        job_store: IngestionJobStore | None = None,
        ingestion_service_factory: Callable[
            [], DocumentIngestionService
        ] | None = None,
        job_store_factory: Callable[
            [], IngestionJobStore
        ] | None = None,
    ) -> None:
        self.ingestion_service = ingestion_service
        self.job_store = job_store

        self.ingestion_service_factory = ingestion_service_factory
        self.job_store_factory = job_store_factory

        if self.job_store is None and self.job_store_factory is None:
            raise ValueError(
                "Either job_store or job_store_factory must be provided"
            )

        if (
            self.ingestion_service is None
            and self.ingestion_service_factory is None
        ):
            raise ValueError(
                "Either ingestion_service or "
                "ingestion_service_factory must be provided"
            )

    def _get_job_store(self) -> IngestionJobStore:
        if self.job_store is not None:
            return self.job_store

        assert self.job_store_factory is not None
        return self.job_store_factory()

    def _get_ingestion_service(self) -> DocumentIngestionService:
        if self.ingestion_service is not None:
            return self.ingestion_service

        assert self.ingestion_service_factory is not None
        return self.ingestion_service_factory()

    def execute(self, job_id: str) -> None:
        print(f"[INGESTION] Starting job {job_id}", flush=True)

        job_store = self._get_job_store()

        print(
            f"[INGESTION] Job store ready: {job_id}",
            flush=True,
        )

        job = job_store.get(job_id)

        if job is None:
            print(
                f"[INGESTION] Job not found: {job_id}",
                flush=True,
            )
            return

        if job.status != "pending":
            print(
                f"[INGESTION] Job {job_id} has status "
                f"{job.status}; skipping",
                flush=True,
            )
            return

        job_store.mark_processing(
            job_id=job_id,
            started_at=datetime.now(),
        )

        print(
            f"[INGESTION] Job marked processing: {job_id}",
            flush=True,
        )

        try:
            print(
                f"[INGESTION] Starting ingestion: {job_id}",
                flush=True,
            )

            ingestion_service = self._get_ingestion_service()

            result = ingestion_service.ingest(
                file_path=Path(job.file_path),
                filename=job.filename,
                knowledge_base_id=job.knowledge_base_id,
            )

            print(
                f"[INGESTION] Ingest finished: {job_id}",
                flush=True,
            )

        except Exception as exc:
            print(
                f"[INGESTION] Ingestion failed: "
                f"{job_id}: {exc!r}",
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