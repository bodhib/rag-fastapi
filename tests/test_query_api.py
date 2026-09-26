import pytest

from fastapi.testclient import TestClient

from app.main import app, get_query_service
from app.query_service import QueryResult
from app.reranker import RerankedResult

from app.security import (
    Principal,
    UserRole,
    get_current_principal,
)

class FakeQueryService:
    def query(
    self,
    question: str,
    retrieval_top_k: int = 10,
    rerank_top_k: int = 5,
    context_top_k: int = 3,
    ) -> QueryResult:


        assert question == (
            "How long do I have to get a refund?"
        )

        context = RerankedResult(
            text="Refunds are available within 30 days.",
            source="policy.pdf",
            page_number=4,
            chunk_index=12,
            dense_score=0.9,
            bm25_score=0.8,
            hybrid_score=0.85,
            rerank_score=9.2,
        )

        return QueryResult(
            answer=(
                "You have 30 days to request a refund. [1]"
            ),
            contexts=[context],
        )


TEST_USER_API_KEY = (
"test-user-key-123456789012345678901234567890"
)

@pytest.fixture
def client() -> TestClient:
    app.dependency_overrides[
        get_query_service
    ] = lambda: FakeQueryService()

    app.dependency_overrides[
        get_current_principal
    ] = lambda: Principal(
        name="test-user",
        role=UserRole.USER,
    )

    with TestClient(app) as test_client:
        yield test_client

    app.dependency_overrides.clear()


def test_query_endpoint(
client: TestClient,
) -> None:


    response = client.post(
        "/query",
        json={
            "query": (
                "How long do I have to get a refund?"
            )
        },
        headers={
            "X-API-Key": TEST_USER_API_KEY,
        },
    )

    assert response.status_code == 200

    assert response.json() == {
    "answer": (
        "You have 30 days to request a refund. [1]"
    ),
    "citations": [
        {
            "source": "policy.pdf",
            "page_number": 4,
            "chunk_index": 12,
        }
    ],
}


def test_query_endpoint_rejects_empty_query(
client: TestClient,
) -> None:


    response = client.post(
        "/query",
        json={
            "query": ""
        },
        headers={
            "X-API-Key": TEST_USER_API_KEY,
        },
    )

    assert response.status_code == 422


def test_query_endpoint_rejects_whitespace_query(
client: TestClient,
) -> None:


    response = client.post(
        "/query",
        json={
            "query": "   "
        },
        headers={
            "X-API-Key": TEST_USER_API_KEY,
        },
    )

    assert response.status_code == 422


def test_query_endpoint_rejects_missing_query(
client: TestClient,
) -> None:


    response = client.post(
        "/query",
        json={},
        headers={
            "X-API-Key": TEST_USER_API_KEY,
        },
    )

    assert response.status_code == 422


def test_query_endpoint_rejects_oversized_query(
client: TestClient,
) -> None:


    response = client.post(
        "/query",
        json={
            "query": "a" * 2001
        },
        headers={
            "X-API-Key": TEST_USER_API_KEY,
        },
    )

    assert response.status_code == 422

