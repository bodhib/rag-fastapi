from datetime import datetime

import pytest
from fastapi.testclient import TestClient

from app.dependencies import get_knowledge_base_store
from app.knowledge_base_store import KnowledgeBase
from app.main import app, get_query_service
from app.query_service import QueryResult
from app.reranker import RerankedResult
from app.security import (
    Principal,
    UserRole,
    get_current_principal,
)


TEST_USER_API_KEY = (
    "test-user-key-123456789012345678901234567890"
)


class FakeQueryService:
    def __init__(self) -> None:
        self.last_question: str | None = None
        self.last_knowledge_base_id: str | None = None

    def query(
        self,
        question: str,
        knowledge_base_id: str = "kb_default",
        retrieval_top_k: int = 10,
        rerank_top_k: int = 5,
        context_top_k: int = 3,
    ) -> QueryResult:
        self.last_question = question
        self.last_knowledge_base_id = knowledge_base_id

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


class FakeKnowledgeBaseStore:
    def __init__(
        self,
        knowledge_bases: dict[str, KnowledgeBase],
    ) -> None:
        self.knowledge_bases = knowledge_bases

    def get(
        self,
        knowledge_base_id: str,
    ) -> KnowledgeBase | None:
        return self.knowledge_bases.get(knowledge_base_id)


@pytest.fixture
def default_knowledge_base() -> KnowledgeBase:
    return KnowledgeBase(
        knowledge_base_id="kb_default",
        name="Default Knowledge Base",
        slug="default",
        description="Test default knowledge base",
        active=True,
        created_at=datetime.now(),
    )


@pytest.fixture
def query_service() -> FakeQueryService:
    return FakeQueryService()


@pytest.fixture
def client(
    query_service: FakeQueryService,
    default_knowledge_base: KnowledgeBase,
) -> TestClient:
    app.dependency_overrides[
        get_query_service
    ] = lambda: query_service

    app.dependency_overrides[
        get_knowledge_base_store
    ] = lambda: FakeKnowledgeBaseStore(
        {
            "kb_default": default_knowledge_base,
        }
    )

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
    query_service: FakeQueryService,
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

    assert query_service.last_question == (
        "How long do I have to get a refund?"
    )

    assert query_service.last_knowledge_base_id == (
        "kb_default"
    )

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


def test_query_endpoint_passes_explicit_knowledge_base(
    client: TestClient,
    query_service: FakeQueryService,
) -> None:
    company_knowledge_base = KnowledgeBase(
        knowledge_base_id="kb_company",
        name="Company Knowledge Base",
        slug="company",
        description="Company policies",
        active=True,
        created_at=datetime.now(),
    )

    app.dependency_overrides[
        get_knowledge_base_store
    ] = lambda: FakeKnowledgeBaseStore(
        {
            "kb_default": KnowledgeBase(
                knowledge_base_id="kb_default",
                name="Default Knowledge Base",
                slug="default",
                description="Default KB",
                active=True,
                created_at=datetime.now(),
            ),
            "kb_company": company_knowledge_base,
        }
    )

    try:
        response = client.post(
            "/query",
            json={
                "query": (
                    "How long do I have to get a refund?"
                ),
                "knowledge_base_id": "kb_company",
            },
            headers={
                "X-API-Key": TEST_USER_API_KEY,
            },
        )

        assert response.status_code == 200

        assert query_service.last_knowledge_base_id == (
            "kb_company"
        )

    finally:
        # Restore the fixture-level dependency.
        app.dependency_overrides[
            get_knowledge_base_store
        ] = lambda: FakeKnowledgeBaseStore(
            {
                "kb_default": KnowledgeBase(
                    knowledge_base_id="kb_default",
                    name="Default Knowledge Base",
                    slug="default",
                    description="Default KB",
                    active=True,
                    created_at=datetime.now(),
                ),
            }
        )


def test_query_endpoint_rejects_unknown_knowledge_base(
    client: TestClient,
) -> None:
    class UnknownKnowledgeBaseStore:
        def get(
            self,
            knowledge_base_id: str,
        ) -> KnowledgeBase | None:
            return None

    app.dependency_overrides[
        get_knowledge_base_store
    ] = lambda: UnknownKnowledgeBaseStore()

    try:
        response = client.post(
            "/query",
            json={
                "query": (
                    "How long do I have to get a refund?"
                ),
                "knowledge_base_id": "does-not-exist",
            },
            headers={
                "X-API-Key": TEST_USER_API_KEY,
            },
        )

        assert response.status_code == 404
        assert response.json()["detail"] == (
            "Knowledge base not found"
        )

    finally:
        app.dependency_overrides.pop(
            get_knowledge_base_store,
            None,
        )


def test_query_endpoint_rejects_inactive_knowledge_base(
    client: TestClient,
) -> None:
    inactive_knowledge_base = KnowledgeBase(
        knowledge_base_id="kb_inactive",
        name="Inactive Knowledge Base",
        slug="inactive",
        description="Inactive test KB",
        active=False,
        created_at=datetime.now(),
    )

    class InactiveKnowledgeBaseStore:
        def get(
            self,
            knowledge_base_id: str,
        ) -> KnowledgeBase | None:
            if knowledge_base_id == "kb_inactive":
                return inactive_knowledge_base

            return None

    app.dependency_overrides[
        get_knowledge_base_store
    ] = lambda: InactiveKnowledgeBaseStore()

    try:
        response = client.post(
            "/query",
            json={
                "query": (
                    "How long do I have to get a refund?"
                ),
                "knowledge_base_id": "kb_inactive",
            },
            headers={
                "X-API-Key": TEST_USER_API_KEY,
            },
        )

        assert response.status_code == 400
        assert response.json()["detail"] == (
            "Knowledge base is inactive"
        )

    finally:
        app.dependency_overrides.pop(
            get_knowledge_base_store,
            None,
        )


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


def test_query_endpoint_rejects_invalid_knowledge_base_id(
    client: TestClient,
) -> None:
    response = client.post(
        "/query",
        json={
            "query": (
                "How long do I have to get a refund?"
            ),
            "knowledge_base_id": "",
        },
        headers={
            "X-API-Key": TEST_USER_API_KEY,
        },
    )

    assert response.status_code == 422


def test_query_endpoint_rejects_whitespace_knowledge_base_id(
    client: TestClient,
) -> None:
    response = client.post(
        "/query",
        json={
            "query": (
                "How long do I have to get a refund?"
            ),
            "knowledge_base_id": "   ",
        },
        headers={
            "X-API-Key": TEST_USER_API_KEY,
        },
    )

    assert response.status_code == 422