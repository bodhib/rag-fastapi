from pathlib import Path

from app.background_ingestion import (
    BackgroundIngestionService,
)
from app.ingestion_service import IngestionResult
from app.ingestion_job_store import IngestionJob


class FakeIngestionService:
    def __init__(
        self,
        result: IngestionResult | None = None,
        error: Exception | None = None,
    ) -> None:
        self.result = result
        self.error = error
        self.received_file_path: Path | None = None
        self.received_filename: str | None = None

    def ingest(
        self,
        file_path: Path,
        filename: str | None = None,
    ) -> IngestionResult:
        self.received_file_path = file_path
        self.received_filename = filename

        if self.error is not None:
            raise self.error

        assert self.result is not None

        return self.result


class FakeJobStore:
    def __init__(
        self,
        job: IngestionJob | None,
    ) -> None:
        self.job = job
        self.processing = False
        self.completed = None
        self.failed = None

    def get(
        self,
        job_id: str,
    ) -> IngestionJob | None:
        if self.job is None:
            return None

        if self.job.job_id != job_id:
            return None

        return self.job

    def mark_processing(
        self,
        job_id: str,
        started_at,
    ) -> None:
        self.processing = True
        self.job.status = "processing"
        self.job.started_at = started_at

    def mark_completed(
        self,
        job_id: str,
        doc_id: str,
        page_count: int,
        chunk_count: int,
        completed_at,
    ) -> None:
        self.completed = {
            "job_id": job_id,
            "doc_id": doc_id,
            "page_count": page_count,
            "chunk_count": chunk_count,
        }

        self.job.status = "completed"
        self.job.doc_id = doc_id
        self.job.page_count = page_count
        self.job.chunk_count = chunk_count
        self.job.completed_at = completed_at

    def mark_failed(
        self,
        job_id: str,
        error_message: str,
        completed_at,
    ) -> None:
        self.failed = {
            "job_id": job_id,
            "error_message": error_message,
        }

        self.job.status = "failed"
        self.job.error_message = error_message
        self.job.completed_at = completed_at


def test_background_ingestion_completes_job() -> None:
    job = IngestionJob(
        job_id="job_test_001",
        filename="policy.txt",
        file_path="tests/policy.txt",
        status="pending",
    )

    ingestion_service = FakeIngestionService(
        result=IngestionResult(
            doc_id="doc_test_001",
            filename="policy.txt",
            page_count=2,
            chunk_count=5,
        )
    )

    job_store = FakeJobStore(job)

    service = BackgroundIngestionService(
        ingestion_service=ingestion_service,
        job_store=job_store,
    )

    service.execute("job_test_001")

    assert job_store.processing is True
    assert job_store.completed == {
        "job_id": "job_test_001",
        "doc_id": "doc_test_001",
        "page_count": 2,
        "chunk_count": 5,
    }

    assert job.status == "completed"
    assert job.doc_id == "doc_test_001"


def test_background_ingestion_marks_failed_job() -> None:
    job = IngestionJob(
        job_id="job_test_002",
        filename="broken.txt",
        file_path="tests/broken.txt",
        status="pending",
    )

    ingestion_service = FakeIngestionService(
        error=ValueError("Document contains no usable content")
    )

    job_store = FakeJobStore(job)

    service = BackgroundIngestionService(
        ingestion_service=ingestion_service,
        job_store=job_store,
    )

    service.execute("job_test_002")

    assert job_store.processing is True

    assert job_store.failed == {
        "job_id": "job_test_002",
        "error_message": (
            "Document contains no usable content"
        ),
    }

    assert job.status == "failed"


def test_background_ingestion_ignores_missing_job() -> None:
    ingestion_service = FakeIngestionService()

    job_store = FakeJobStore(None)

    service = BackgroundIngestionService(
        ingestion_service=ingestion_service,
        job_store=job_store,
    )

    service.execute("job_missing")

    assert job_store.processing is False


def test_background_ingestion_ignores_non_pending_job() -> None:
    job = IngestionJob(
        job_id="job_test_003",
        filename="policy.txt",
        file_path="policy.txt",
        status="completed",
    )

    ingestion_service = FakeIngestionService()

    job_store = FakeJobStore(job)

    service = BackgroundIngestionService(
        ingestion_service=ingestion_service,
        job_store=job_store,
    )

    service.execute("job_test_003")

    assert job_store.processing is False
