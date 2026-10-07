# tests/test_recruiter_search_service.py

from dataclasses import dataclass

import app.recruiter_search_service as recruiter_search_service
from app.recruiter_search_service import RecruiterSearchService


# The production service references MAX_TOP_K.
# Importing/setting it here keeps the test focused on the service contract
# while the production module is being completed.
if not hasattr(recruiter_search_service, "MAX_TOP_K"):
    recruiter_search_service.MAX_TOP_K = 100


@dataclass
class FakeCandidate:
    candidate_id: str
    full_name: str
    resume_filename: str
    resume_document_id: str | None
    email: str | None = None
    phone: str | None = None
    location: str | None = "Kolkata"
    current_title: str | None = "Senior Python Backend Engineer"
    current_company: str | None = "Example Technologies"
    years_experience: float | None = 6
    skills: list[str] | None = None
    education: str | None = None
    availability: str | None = "Immediate"
    active: bool = True

    def __post_init__(self):
        if self.skills is None:
            self.skills = ["python", "fastapi"]


class FakeCandidateStore:
    def __init__(self, candidates):
        self.candidates = candidates

    def find_matching(
        self,
        knowledge_base_id,
        skills=None,
        min_years_experience=None,
        max_years_experience=None,
        location=None,
        title=None,
    ):
        return self.candidates


class FakeSearchResult:
    def __init__(
        self,
        doc_id,
        source,
        rerank_score,
        text="Resume evidence",
    ):
        self.doc_id = doc_id
        self.text = text
        self.snippet = text
        self.source = source
        self.page_number = 1
        self.chunk_index = 0
        self.dense_score = 0.64
        self.bm25_score = 0.0
        self.hybrid_score = 0.64
        self.rerank_score = rerank_score


class FakeSearchService:
    def __init__(self, results):
        self.results = results

    def search(
        self,
        query,
        knowledge_base_id,
        top_k,
    ):
        return self.results


def create_service(candidates, results):
    return RecruiterSearchService(
        candidate_store=FakeCandidateStore(candidates),
        search_service=FakeSearchService(results),
    )


def test_recruiter_search_joins_by_document_id_not_filename():
    candidate = FakeCandidate(
        candidate_id="cand_alice",
        full_name="Alice Sen",
        resume_filename="alice_resume.txt",
        resume_document_id="doc_alice_123",
    )

    search_result = FakeSearchResult(
        doc_id="doc_alice_123",
        source="alice_resume.txt.txt",
        rerank_score=7.51,
        text="Alice has 6 years of Python and FastAPI backend experience.",
    )

    service = create_service(
        [candidate],
        [search_result],
    )

    result = service.search(
        query="Python developers with FastAPI backend experience",
        knowledge_base_id="kb_recruitment",
        top_k=10,
    )

    assert len(result.results) == 1

    recruiter_result = result.results[0]

    assert recruiter_result.candidate_id == "cand_alice"
    assert recruiter_result.full_name == "Alice Sen"
    assert recruiter_result.resume_document_id == "doc_alice_123"
    assert recruiter_result.resume_filename == "alice_resume.txt"

    assert len(recruiter_result.evidence) == 1

    evidence = recruiter_result.evidence[0]

    assert evidence.doc_id == "doc_alice_123"
    assert evidence.source == "alice_resume.txt.txt"
    assert evidence.rerank_score == 7.51
    assert recruiter_result.best_score == 7.51


def test_recruiter_search_does_not_depend_on_filename():
    candidate = FakeCandidate(
        candidate_id="cand_alice",
        full_name="Alice Sen",
        resume_filename="alice_resume.txt",
        resume_document_id="doc_alice",
    )

    search_result = FakeSearchResult(
        doc_id="doc_alice",
        source="completely_different_filename.pdf",
        rerank_score=8.0,
    )

    service = create_service(
        [candidate],
        [search_result],
    )

    result = service.search(
        query="FastAPI",
        knowledge_base_id="kb_recruitment",
        top_k=10,
    )

    assert len(result.results) == 1
    assert result.results[0].candidate_id == "cand_alice"
    assert result.results[0].resume_document_id == "doc_alice"


def test_recruiter_search_ignores_filename_mismatch():
    candidate = FakeCandidate(
        candidate_id="cand_alice",
        full_name="Alice Sen",
        resume_filename="alice_resume.txt",
        resume_document_id="doc_alice",
    )

    search_result = FakeSearchResult(
        doc_id="doc_alice",
        source="another_name.txt",
        rerank_score=8.0,
    )

    service = create_service(
        [candidate],
        [search_result],
    )

    result = service.search(
        query="Python",
        knowledge_base_id="kb_recruitment",
        top_k=10,
    )

    assert len(result.results) == 1
    assert result.results[0].candidate_id == "cand_alice"


def test_recruiter_search_excludes_candidate_without_document_id():
    candidate = FakeCandidate(
        candidate_id="cand_unlinked",
        full_name="Unlinked Candidate",
        resume_filename="candidate.txt",
        resume_document_id=None,
    )

    search_result = FakeSearchResult(
        doc_id="doc_unrelated",
        source="candidate.txt.txt",
        rerank_score=9.0,
    )

    service = create_service(
        [candidate],
        [search_result],
    )

    result = service.search(
        query="Python",
        knowledge_base_id="kb_recruitment",
        top_k=10,
    )

    assert result.results == []


def test_recruiter_search_uses_rerank_score_after_document_join():
    alice = FakeCandidate(
        candidate_id="cand_alice",
        full_name="Alice Sen",
        resume_filename="alice_resume.txt",
        resume_document_id="doc_alice",
    )

    rahul = FakeCandidate(
        candidate_id="cand_rahul",
        full_name="Rahul Das",
        resume_filename="rahul_resume.txt",
        resume_document_id="doc_rahul",
        years_experience=3,
        skills=["python", "django"],
    )

    search_results = [
        FakeSearchResult(
            doc_id="doc_rahul",
            source="rahul_resume.txt.txt",
            rerank_score=5.7,
            text="Rahul has Python backend experience.",
        ),
        FakeSearchResult(
            doc_id="doc_alice",
            source="alice_resume.txt.txt",
            rerank_score=7.5,
            text="Alice has FastAPI backend experience.",
        ),
    ]

    service = create_service(
        [alice, rahul],
        search_results,
    )

    result = service.search(
        query="Python backend",
        knowledge_base_id="kb_recruitment",
        top_k=10,
    )

    assert len(result.results) == 2
    assert result.results[0].candidate_id == "cand_alice"
    assert result.results[0].best_score == 7.5
    assert result.results[1].candidate_id == "cand_rahul"
    assert result.results[1].best_score == 5.7


def test_recruiter_search_does_not_use_filename_as_fallback():
    candidate = FakeCandidate(
        candidate_id="cand_alice",
        full_name="Alice Sen",
        resume_filename="alice_resume.txt",
        resume_document_id="doc_alice",
    )

    search_result = FakeSearchResult(
        doc_id="different_document",
        source="alice_resume.txt",
        rerank_score=9.0,
    )

    service = create_service(
        [candidate],
        [search_result],
    )

    result = service.search(
        query="Python",
        knowledge_base_id="kb_recruitment",
        top_k=10,
    )

    assert result.results == []


def test_recruiter_search_supports_same_filename_for_multiple_candidates():
    alice = FakeCandidate(
        candidate_id="cand_alice",
        full_name="Alice Sen",
        resume_filename="resume.txt",
        resume_document_id="doc_alice",
    )

    rahul = FakeCandidate(
        candidate_id="cand_rahul",
        full_name="Rahul Das",
        resume_filename="resume.txt",
        resume_document_id="doc_rahul",
    )

    search_results = [
        FakeSearchResult(
            doc_id="doc_alice",
            source="resume.txt",
            rerank_score=8.0,
        ),
        FakeSearchResult(
            doc_id="doc_rahul",
            source="resume.txt",
            rerank_score=6.0,
        ),
    ]

    service = create_service(
        [alice, rahul],
        search_results,
    )

    result = service.search(
        query="Python",
        knowledge_base_id="kb_recruitment",
        top_k=10,
    )

    assert len(result.results) == 2
    assert result.results[0].candidate_id == "cand_alice"
    assert result.results[1].candidate_id == "cand_rahul"