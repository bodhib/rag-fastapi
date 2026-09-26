from datetime import datetime

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database import Base
from app.ingestion_job_store import (
    IngestionJob,
    IngestionJobStore,
)


def create_test_store(tmp_path):
    database_url = (
        f"sqlite:///{tmp_path / 'ingestion_jobs.db'}"
    )

    engine = create_engine(database_url)

    Base.metadata.create_all(bind=engine)

    session_factory = sessionmaker(
        bind=engine,
        autocommit=False,
        autoflush=False,
    )

    return IngestionJobStore(
        session_factory=session_factory
    )


def test_create_and_get_job(tmp_path) -> None:
    store = create_test_store(tmp_path)

    created_at = datetime.now()

    store.create(
        IngestionJob(
            job_id="job_test_001",
            filename="policy.txt",
            file_path="tests/uploads/policy.txt",
            status="pending",
            created_at=created_at,
        )
    )

    job = store.get("job_test_001")

    assert job is not None
    assert job.job_id == "job_test_001"
    assert job.filename == "policy.txt"
    assert job.status == "pending"


def test_mark_processing(tmp_path) -> None:
    store = create_test_store(tmp_path)

    store.create(
        IngestionJob(
            job_id="job_test_002",
            filename="policy.txt",
            file_path="policy.txt",
            status="pending",
        )
    )

    started_at = datetime.now()

    store.mark_processing(
        job_id="job_test_002",
        started_at=started_at,
    )

    job = store.get("job_test_002")

    assert job is not None
    assert job.status == "processing"
    assert job.started_at == started_at


def test_mark_completed(tmp_path) -> None:
    store = create_test_store(tmp_path)

    store.create(
        IngestionJob(
            job_id="job_test_003",
            filename="policy.txt",
            file_path="policy.txt",
            status="pending",
        )
    )

    completed_at = datetime.now()

    store.mark_completed(
        job_id="job_test_003",
        doc_id="doc_test_003",
        page_count=3,
        chunk_count=8,
        completed_at=completed_at,
    )

    job = store.get("job_test_003")

    assert job is not None
    assert job.status == "completed"
    assert job.doc_id == "doc_test_003"
    assert job.page_count == 3
    assert job.chunk_count == 8
    assert job.completed_at == completed_at
    assert job.error_message is None


def test_mark_failed(tmp_path) -> None:
    store = create_test_store(tmp_path)

    store.create(
        IngestionJob(
            job_id="job_test_004",
            filename="broken.txt",
            file_path="broken.txt",
            status="pending",
        )
    )

    completed_at = datetime.now()

    store.mark_failed(
        job_id="job_test_004",
        error_message="Document processing failed",
        completed_at=completed_at,
    )

    job = store.get("job_test_004")

    assert job is not None
    assert job.status == "failed"
    assert job.error_message == (
        "Document processing failed"
    )
    assert job.completed_at == completed_at
