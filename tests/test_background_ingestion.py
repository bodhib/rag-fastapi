from pathlib import Path

import pytest

from app.background_ingestion import BackgroundIngestionService
from app.ingestion_job_store import IngestionJob
from app.ingestion_service import IngestionResult


class FakeIngestionService:
    def __init__(self):
        self.calls = []
        self.should_fail = False

    def ingest(
        self,
        file_path,
        filename,
        knowledge_base_id,
    ):
        self.calls.append(
            {
                "file_path": file_path,
                "filename": filename,
                "knowledge_base_id": knowledge_base_id,
            }
        )

        if self.should_fail:
            raise RuntimeError("ingestion failed")

        return IngestionResult(
            doc_id="doc_test",
            filename=filename,
            page_count=2,
            chunk_count=4,
        )


class FakeJobStore:
    def __init__(self, job):
        self.job = job
        self.processing_calls = []
        self.completed_calls = []
        self.failed_calls = []

    def get(self, job_id):
        if self.job.job_id == job_id:
            return self.job
        return None

    def mark_processing(self, job_id, started_at):
        self.processing_calls.append(
            {
                "job_id": job_id,
                "started_at": started_at,
            }
        )

        self.job.status = "processing"

    def mark_completed(
        self,
        job_id,
        doc_id,
        page_count,
        chunk_count,
        completed_at,
    ):
        self.completed_calls.append(
            {
                "job_id": job_id,
                "doc_id": doc_id,
                "page_count": page_count,
                "chunk_count": chunk_count,
                "completed_at": completed_at,
            }
        )

        self.job.status = "completed"
        self.job.doc_id = doc_id

    def mark_failed(
        self,
        job_id,
        error_message,
        completed_at,
    ):
        self.failed_calls.append(
            {
                "job_id": job_id,
                "error_message": error_message,
                "completed_at": completed_at,
            }
        )

        self.job.status = "failed"

def create_job(
    *,
    knowledge_base_id="kb_default",
):
    return IngestionJob(
        job_id="job_test",
        filename="policy.txt",
        file_path="/tmp/policy.txt",
        knowledge_base_id=knowledge_base_id,
        status="pending",
        doc_id=None,
    )

def test_background_ingestion_completes_job():
    job = create_job(
        knowledge_base_id="kb_company",
    )

    ingestion_service = FakeIngestionService()
    job_store = FakeJobStore(job)

    service = BackgroundIngestionService(
        ingestion_service=ingestion_service,
        job_store=job_store,
    )

    service.execute("job_test")

    assert len(job_store.processing_calls) == 1

    processing = job_store.processing_calls[0]

    assert processing["job_id"] == "job_test"
    assert processing["started_at"] is not None

    assert len(job_store.completed_calls) == 1

    completed = job_store.completed_calls[0]

    assert completed["job_id"] == "job_test"
    assert completed["doc_id"] == "doc_test"
    assert completed["page_count"] == 2
    assert completed["chunk_count"] == 4
    assert completed["completed_at"] is not None

    assert job.status == "completed"
    assert job.doc_id == "doc_test"

    assert job_store.failed_calls == []


def test_background_ingestion_marks_failed_job():
    job = create_job(
        knowledge_base_id="kb_company",
    )

    ingestion_service = FakeIngestionService()
    ingestion_service.should_fail = True

    job_store = FakeJobStore(job)

    service = BackgroundIngestionService(
        ingestion_service=ingestion_service,
        job_store=job_store,
    )

    service.execute("job_test")

    assert len(job_store.processing_calls) == 1

    processing = job_store.processing_calls[0]

    assert processing["job_id"] == "job_test"
    assert processing["started_at"] is not None

    assert job_store.completed_calls == []

    assert len(job_store.failed_calls) == 1

    failed = job_store.failed_calls[0]

    assert failed["job_id"] == "job_test"
    assert failed["error_message"] == "ingestion failed"
    assert failed["completed_at"] is not None

    assert job.status == "failed"