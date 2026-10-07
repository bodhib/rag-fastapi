from fastapi.testclient import TestClient

from app.main import app
from app.dependencies import (
    get_knowledge_base_store,
    get_search_service,
)
from app.search_service import SearchResult


class FakeKnowledgeBase:
    def __init__(
        self,
        knowledge_base_id: str,
        active: bool = True,
    ):
        self.knowledge_base_id = knowledge_base_id
        self.active = active


class FakeKnowledgeBaseStore:
    def __init__(self):
        self.knowledge_bases = {
            "kb_default": FakeKnowledgeBase(
                "kb_default",
                active=True,
            ),
            "kb_company": FakeKnowledgeBase(
                "kb_company",
                active=True,
            ),
        }

    def get(self, knowledge_base_id: str):
        return self.knowledge_bases.get(
            knowledge_base_id
        )


class FakeSearchService:
    def __init__(self):
        self.last_query = None
        self.last_knowledge_base_id = None
        self.last_top_k = None

    def search(
        self,
        query: str,
        knowledge_base_id: str,
        top_k: int,
    ):
        self.last_query = query
        self.last_knowledge_base_id = knowledge_base_id
        self.last_top_k = top_k

        return [
            SearchResult(
                text=(
                    "Employees must submit remote-work "
                    "requests at least 14 calendar days "
                    "before the requested period."
                ),
                source="remote_work.txt",
                page_number=1,
                chunk_index=0,
                dense_score=0.91,
                bm25_score=0.84,
                hybrid_score=0.875,
                rerank_score=9.7,
            ),
            SearchResult(
                text=(
                    "Remote-work requests are subject "
                    "to manager approval."
                ),
                source="remote_work.txt",
                page_number=2,
                chunk_index=1,
                dense_score=0.82,
                bm25_score=0.76,
                hybrid_score=0.79,
                rerank_score=8.4,
            ),
        ][:top_k]


def setup_overrides(
    search_service,
    knowledge_base_store,
):
    app.dependency_overrides[
        get_search_service
    ] = lambda: search_service

    app.dependency_overrides[
        get_knowledge_base_store
    ] = lambda: knowledge_base_store


def teardown_overrides():
    app.dependency_overrides.pop(
        get_search_service,
        None,
    )

    app.dependency_overrides.pop(
        get_knowledge_base_store,
        None,
    )


def test_search_endpoint_returns_ranked_results():
    search_service = FakeSearchService()
    knowledge_base_store = FakeKnowledgeBaseStore()

    setup_overrides(
        search_service,
        knowledge_base_store,
    )

    try:
        with TestClient(app) as client:
            response = client.post(
                "/search",
                json={
                    "query": (
                        "What are the remote work requirements?"
                    ),
                    "knowledge_base_id": "kb_company",
                    "top_k": 2,
                },
            )

        assert response.status_code == 200

        body = response.json()

        assert body["query"] == (
            "What are the remote work requirements?"
        )

        assert body["knowledge_base_id"] == "kb_company"

        assert len(body["results"]) == 2

        first = body["results"][0]

        assert first["source"] == "remote_work.txt"
        assert first["page_number"] == 1
        assert first["chunk_index"] == 0

        assert first["snippet"] == first["text"]

        assert first["rerank_score"] == 9.7

    finally:
        teardown_overrides()


def test_search_endpoint_forwards_request_parameters():
    search_service = FakeSearchService()
    knowledge_base_store = FakeKnowledgeBaseStore()

    setup_overrides(
        search_service,
        knowledge_base_store,
    )

    try:
        with TestClient(app) as client:
            response = client.post(
                "/search",
                json={
                    "query": "remote work",
                    "knowledge_base_id": "kb_company",
                    "top_k": 1,
                },
            )

        assert response.status_code == 200

        assert search_service.last_query == (
            "remote work"
        )

        assert (
            search_service.last_knowledge_base_id
            == "kb_company"
        )

        assert search_service.last_top_k == 1

    finally:
        teardown_overrides()


def test_search_endpoint_rejects_unknown_knowledge_base():
    search_service = FakeSearchService()
    knowledge_base_store = FakeKnowledgeBaseStore()

    setup_overrides(
        search_service,
        knowledge_base_store,
    )

    try:
        with TestClient(app) as client:
            response = client.post(
                "/search",
                json={
                    "query": "remote work",
                    "knowledge_base_id": (
                        "kb_does_not_exist"
                    ),
                },
            )

        assert response.status_code == 404

        assert response.json() == {
            "detail": "Knowledge base not found"
        }

    finally:
        teardown_overrides()


def test_search_endpoint_rejects_inactive_knowledge_base():
    search_service = FakeSearchService()
    knowledge_base_store = FakeKnowledgeBaseStore()

    knowledge_base_store.knowledge_bases[
        "kb_company"
    ].active = False

    setup_overrides(
        search_service,
        knowledge_base_store,
    )

    try:
        with TestClient(app) as client:
            response = client.post(
                "/search",
                json={
                    "query": "remote work",
                    "knowledge_base_id": "kb_company",
                },
            )

        assert response.status_code == 400

        assert response.json() == {
            "detail": "Knowledge base is inactive"
        }

    finally:
        teardown_overrides()


def test_search_endpoint_uses_default_knowledge_base():
    search_service = FakeSearchService()
    knowledge_base_store = FakeKnowledgeBaseStore()

    setup_overrides(
        search_service,
        knowledge_base_store,
    )

    try:
        with TestClient(app) as client:
            response = client.post(
                "/search",
                json={
                    "query": "remote work",
                },
            )

        assert response.status_code == 200

        assert (
            search_service.last_knowledge_base_id
            == "kb_default"
        )

    finally:
        teardown_overrides()


def test_search_endpoint_rejects_blank_query():
    search_service = FakeSearchService()
    knowledge_base_store = FakeKnowledgeBaseStore()

    setup_overrides(
        search_service,
        knowledge_base_store,
    )

    try:
        with TestClient(app) as client:
            response = client.post(
                "/search",
                json={
                    "query": "   ",
                    "knowledge_base_id": "kb_company",
                },
            )

        assert response.status_code == 422

    finally:
        teardown_overrides()


def test_search_endpoint_rejects_top_k_above_50():
    search_service = FakeSearchService()
    knowledge_base_store = FakeKnowledgeBaseStore()

    setup_overrides(
        search_service,
        knowledge_base_store,
    )

    try:
        with TestClient(app) as client:
            response = client.post(
                "/search",
                json={
                    "query": "remote work",
                    "knowledge_base_id": "kb_company",
                    "top_k": 51,
                },
            )

        assert response.status_code == 422

    finally:
        teardown_overrides()