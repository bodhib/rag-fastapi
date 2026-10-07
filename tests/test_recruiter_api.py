from fastapi.testclient import TestClient

from app.dependencies import (
    get_candidate_store,
    get_knowledge_base_store,
    get_recruiter_search_service,
)
from app.main import app


class FakeKnowledgeBase:

    def __init__(self, active=True):
        self.active = active


class FakeKnowledgeBaseStore:

    def get(self, knowledge_base_id):
        if knowledge_base_id == "kb_recruitment":
            return FakeKnowledgeBase(active=True)

        return None


class FakeCandidate:

    candidate_id = "cand_alice"
    knowledge_base_id = "kb_recruitment"
    full_name = "Alice Sen"
    email = "alice@example.com"
    phone = None
    location = "Kolkata"
    current_title = "Senior Python Backend Engineer"
    current_company = "Example Technologies"
    years_experience = 6
    skills = ["fastapi", "python"]
    education = "B.Tech Computer Science"
    availability = "30 days"
    resume_filename = "alice_resume.txt"
    resume_document_id = None
    active = True


class FakeCandidateStore:

    def get(self, candidate_id):
        if candidate_id == "cand_alice":
            return FakeCandidate()

        return None

    def list(
        self,
        knowledge_base_id,
        active_only=True,
    ):
        return [FakeCandidate()]


class FakeRecruiterSearchResult:

    def __init__(self):
        self.candidate_id = "cand_alice"
        self.full_name = "Alice Sen"
        self.email = "alice@example.com"
        self.location = "Kolkata"
        self.current_title = (
            "Senior Python Backend Engineer"
        )
        self.current_company = (
            "Example Technologies"
        )
        self.years_experience = 6
        self.skills = [
            "fastapi",
            "python",
        ]
        self.availability = "30 days"
        self.resume_filename = (
            "alice_resume.txt"
        )
        self.resume_document_id = "doc_alice_123"
        self.best_score = 9.5

        self.evidence = []


class FakeRecruiterSearchResponse:

    query = "Python FastAPI developer"
    knowledge_base_id = "kb_recruitment"

    results = [
        FakeRecruiterSearchResult()
    ]


class FakeRecruiterSearchService:

    def search(self, **kwargs):
        assert (
            kwargs["knowledge_base_id"]
            == "kb_recruitment"
        )

        assert kwargs["min_years_experience"] == 5

        return FakeRecruiterSearchResponse()


def test_get_candidate():

    app.dependency_overrides[
        get_candidate_store
    ] = lambda: FakeCandidateStore()

    try:
        with TestClient(app) as client:

            response = client.get(
                "/recruiter/candidates/cand_alice"
            )

        assert response.status_code == 200

        body = response.json()

        assert body["candidate_id"] == "cand_alice"
        assert body["full_name"] == "Alice Sen"
        assert body["years_experience"] == 6

    finally:
        app.dependency_overrides.pop(
            get_candidate_store,
            None,
        )


def test_recruiter_search():

    app.dependency_overrides[
        get_knowledge_base_store
    ] = lambda: FakeKnowledgeBaseStore()

    app.dependency_overrides[
        get_recruiter_search_service
    ] = lambda: FakeRecruiterSearchService()

    try:
        with TestClient(app) as client:

            response = client.post(
                "/recruiter/search",
                json={
                    "query": (
                        "Python FastAPI developer"
                    ),
                    "knowledge_base_id": (
                        "kb_recruitment"
                    ),
                    "skills": [
                        "python",
                        "fastapi",
                    ],
                    "min_years_experience": 5,
                    "top_k": 10,
                },
            )

        assert response.status_code == 200

        body = response.json()

        assert body["knowledge_base_id"] == (
            "kb_recruitment"
        )

        assert len(body["results"]) == 1

        assert (
            body["results"][0]["candidate_id"]
            == "cand_alice"
        )

    finally:
        app.dependency_overrides.pop(
            get_knowledge_base_store,
            None,
        )

        app.dependency_overrides.pop(
            get_recruiter_search_service,
            None,
        )


def test_recruiter_search_rejects_invalid_experience_range():

    with TestClient(app) as client:

        response = client.post(
            "/recruiter/search",
            json={
                "query": "Python developer",
                "knowledge_base_id": (
                    "kb_recruitment"
                ),
                "min_years_experience": 8,
                "max_years_experience": 3,
            },
        )

    assert response.status_code == 422