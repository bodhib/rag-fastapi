import io

import pytest

from fastapi.testclient import TestClient

from app.main import app
from app.dependencies import get_ingestion_service
from app.ingestion_service import IngestionResult
from app.security import (
Principal,
UserRole,
get_current_principal,
)

class FakeIngestionService:
    def ingest(
    self,
    source_file,
    filename: str,
    ) -> IngestionResult:


        assert filename == "policy.txt"

        return IngestionResult(
            doc_id="doc_123",
            filename="policy.txt",
            chunk_count=3,
        )


@pytest.fixture
def client() -> TestClient:
    app.dependency_overrides[
    get_ingestion_service
    ] = lambda: FakeIngestionService()


    app.dependency_overrides[
        get_current_principal
    ] = lambda: Principal(
        name="test-admin",
        role=UserRole.ADMIN,
    )

    with TestClient(app) as test_client:
        yield test_client

    app.dependency_overrides.clear()


def test_upload_document(
client: TestClient,
) -> None:


    response = client.post(
        "/documents/upload",
        files={
            "file": (
                "policy.txt",
                io.BytesIO(
                    b"Refunds are available within 30 days."
                ),
                "text/plain",
            )
        },
    )

    assert response.status_code == 202

    body = response.json()

    assert "job_id" in body
    assert body["job_id"]
    assert body["status"] == "pending"


def test_upload_requires_file(
client: TestClient,
) -> None:


    response = client.post(
        "/documents/upload",
    )

    assert response.status_code == 422

