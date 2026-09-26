from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime

from sqlalchemy.orm import Session

from app.models.ingestion_job import IngestionJobModel


@dataclass
class IngestionJob:
    """
    Application-level representation of an ingestion job.
    """

    job_id: str
    filename: str
    file_path: str
    status: str
    doc_id: str | None = None
    page_count: int | None = None
    chunk_count: int | None = None
    error_message: str | None = None
    created_at: datetime | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None


class IngestionJobStore:
    """
    SQL-backed store for ingestion job state.
    """

    def __init__(
        self,
        session_factory: Callable[[], Session],
    ) -> None:
        self.session_factory = session_factory

    def create(
        self,
        job: IngestionJob,
    ) -> None:
        with self.session_factory() as session:
            session.add(
                IngestionJobModel(
                    job_id=job.job_id,
                    filename=job.filename,
                    file_path=job.file_path,
                    status=job.status,
                    doc_id=job.doc_id,
                    page_count=job.page_count,
                    chunk_count=job.chunk_count,
                    error_message=job.error_message,
                    created_at=job.created_at or datetime.now(),
                    started_at=job.started_at,
                    completed_at=job.completed_at,
                )
            )

            session.commit()

    def get(
        self,
        job_id: str,
    ) -> IngestionJob | None:
        with self.session_factory() as session:
            job = session.get(
                IngestionJobModel,
                job_id,
            )

            if job is None:
                return None

            return self._to_job(job)

    def mark_processing(
        self,
        job_id: str,
        started_at: datetime,
    ) -> None:
        with self.session_factory() as session:
            job = session.get(
                IngestionJobModel,
                job_id,
            )

            if job is None:
                raise ValueError(
                    f"Ingestion job not found: {job_id}"
                )

            job.status = "processing"
            job.started_at = started_at

            session.commit()

    def mark_completed(
        self,
        job_id: str,
        doc_id: str,
        page_count: int,
        chunk_count: int,
        completed_at: datetime,
    ) -> None:
        with self.session_factory() as session:
            job = session.get(
                IngestionJobModel,
                job_id,
            )

            if job is None:
                raise ValueError(
                    f"Ingestion job not found: {job_id}"
                )

            job.status = "completed"
            job.doc_id = doc_id
            job.page_count = page_count
            job.chunk_count = chunk_count
            job.error_message = None
            job.completed_at = completed_at

            session.commit()

    def mark_failed(
        self,
        job_id: str,
        error_message: str,
        completed_at: datetime,
    ) -> None:
        with self.session_factory() as session:
            job = session.get(
                IngestionJobModel,
                job_id,
            )

            if job is None:
                raise ValueError(
                    f"Ingestion job not found: {job_id}"
                )

            job.status = "failed"
            job.error_message = error_message
            job.completed_at = completed_at

            session.commit()

    @staticmethod
    def _to_job(
        job: IngestionJobModel,
    ) -> IngestionJob:
        return IngestionJob(
            job_id=job.job_id,
            filename=job.filename,
            file_path=job.file_path,
            status=job.status,
            doc_id=job.doc_id,
            page_count=job.page_count,
            chunk_count=job.chunk_count,
            error_message=job.error_message,
            created_at=job.created_at,
            started_at=job.started_at,
            completed_at=job.completed_at,
        )
