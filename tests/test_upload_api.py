import io

import pytest

from fastapi.testclient import TestClient

from app.main import app
from app.dependencies import (
    get_ingestion_service,
    get_knowledge_base_store,
)
from app.knowledge_base_store import KnowledgeBase
from app.ingestion_service import IngestionResult
from app.security import (
Principal,
UserRole,
get_current_principal,
)

from datetime import datetime

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
        data={
            "knowledge_base_id": "kb_default",
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


def test_upload_rejects_unknown_knowledge_base(
    client: TestClient,
) -> None:
    class FakeKnowledgeBaseStore:
        def get(self, knowledge_base_id: str):
            return None

    app.dependency_overrides[
        get_knowledge_base_store
    ] = lambda: FakeKnowledgeBaseStore()

    try:
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
            data={
                "knowledge_base_id": "does-not-exist",
            },
        )

        assert response.status_code == 404
        assert response.json()["detail"] == "Knowledge base not found"

    finally:
        app.dependency_overrides.pop(
            get_knowledge_base_store,
            None,
        )

def test_upload_rejects_inactive_knowledge_base(
    client: TestClient,
) -> None:
    inactive_knowledge_base = KnowledgeBase(
        knowledge_base_id="kb_inactive",
        name="Inactive Knowledge Base",
        slug="inactive-knowledge-base",
        description="Test inactive knowledge base",
        active=False,
        created_at=datetime.now(),
    )

    class FakeKnowledgeBaseStore:
        def get(self, knowledge_base_id: str):
            return inactive_knowledge_base

    app.dependency_overrides[
        get_knowledge_base_store
    ] = lambda: FakeKnowledgeBaseStore()

    try:
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
            data={
                "knowledge_base_id": "kb_inactive",
            },
        )

        assert response.status_code == 400
        assert response.json()["detail"] == "Knowledge base is inactive"

    finally:
        app.dependency_overrides.pop(
            get_knowledge_base_store,
            None,
        )            

