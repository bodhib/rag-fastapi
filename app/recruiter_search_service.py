from __future__ import annotations

from dataclasses import dataclass

from app.candidate_store import (
    Candidate,
    CandidateStore,
)
from app.search_service import SearchService


@dataclass(frozen=True)
class RecruiterEvidence:
    """
    One piece of resume evidence supporting a candidate match.

    Scores are chunk-level retrieval diagnostics.
    """

    text: str
    snippet: str
    source: str
    page_number: int
    chunk_index: int
    dense_score: float
    bm25_score: float
    hybrid_score: float
    rerank_score: float
    doc_id: str = ""


@dataclass(frozen=True)
class RecruiterCandidateResult:
    """
    Candidate-level recruiter result.

    A candidate can have multiple pieces of evidence,
    but the candidate itself receives one final ranking.
    """

    candidate_id: str
    full_name: str
    email: str | None
    location: str | None
    current_title: str | None
    current_company: str | None
    years_experience: float | None
    skills: list[str]
    availability: str | None
    resume_filename: str
    best_score: float
    evidence: list[RecruiterEvidence]
    resume_document_id: str | None = None


@dataclass(frozen=True)
class RecruiterSearchResponse:
    query: str
    knowledge_base_id: str
    results: list[RecruiterCandidateResult]


class RecruiterSearchService:
    """
    Recruiter search orchestration.

    Pipeline:

        recruiter query
              |
              v
        structured candidate filtering
              |
              v
        eligible candidates
              |
              v
        candidate resume_document_ids
              |
              v
        restricted hybrid RAG retrieval
              |
              v
        CrossEncoder reranking
              |
              v
        evidence grouped by candidate
              |
              v
        candidate-level ranking
              |
              v
        recruiter results
    """

    DEFAULT_TOP_K = 10
    MAX_TOP_K = 50

    # Retrieve more evidence chunks than final candidates.
    # This prevents a single candidate's first chunk from
    # consuming the entire retrieval budget.
    EVIDENCE_MULTIPLIER = 5

    MAX_EVIDENCE_RESULTS = 50

    MAX_EVIDENCE_PER_CANDIDATE = 3

    def __init__(
        self,
        candidate_store: CandidateStore,
        search_service: SearchService,
    ) -> None:
        self.candidate_store = candidate_store
        self.search_service = search_service

    def search(
        self,
        query: str,
        knowledge_base_id: str,
        top_k: int = DEFAULT_TOP_K,
        skills: list[str] | None = None,
        min_years_experience: float | None = None,
        max_years_experience: float | None = None,
        location: str | None = None,
        title: str | None = None,
    ) -> RecruiterSearchResponse:

        # ---------------------------------------------------------
        # 1. Validate request
        # ---------------------------------------------------------

        if not query.strip():
            raise ValueError(
                "query cannot be empty"
            )

        if not knowledge_base_id.strip():
            raise ValueError(
                "knowledge_base_id cannot be empty"
            )

        if top_k <= 0:
            raise ValueError(
                "top_k must be greater than zero"
            )

        if top_k > self.MAX_TOP_K:
            raise ValueError(
                f"top_k cannot be greater than "
                f"{self.MAX_TOP_K}"
            )

        if (
            min_years_experience is not None
            and max_years_experience is not None
            and min_years_experience
            > max_years_experience
        ):
            raise ValueError(
                "min_years_experience cannot be greater "
                "than max_years_experience"
            )

        # ---------------------------------------------------------
        # 2. Structured candidate filtering
        # ---------------------------------------------------------

        candidates = self.candidate_store.find_matching(
            knowledge_base_id=knowledge_base_id,
            skills=skills,
            min_years_experience=min_years_experience,
            max_years_experience=max_years_experience,
            location=location,
            title=title,
        )

        if not candidates:
            return RecruiterSearchResponse(
                query=query,
                knowledge_base_id=knowledge_base_id,
                results=[],
            )

        # ---------------------------------------------------------
        # 3. Collect canonical resume document IDs
        # ---------------------------------------------------------

        candidates_with_resumes = [
            candidate
            for candidate in candidates
            if candidate.resume_document_id
        ]

        if not candidates_with_resumes:
            return RecruiterSearchResponse(
                query=query,
                knowledge_base_id=knowledge_base_id,
                results=[],
            )

        allowed_doc_ids = {
            candidate.resume_document_id
            for candidate in candidates_with_resumes
            if candidate.resume_document_id
        }

        candidate_by_document_id = {
            candidate.resume_document_id: candidate
            for candidate in candidates_with_resumes
            if candidate.resume_document_id
        }

        # ---------------------------------------------------------
        # 4. Restricted RAG retrieval
        # ---------------------------------------------------------
        #
        # IMPORTANT:
        #
        # RAG is no longer searching the entire knowledge base.
        #
        # It can only search resume documents belonging to
        # candidates that passed structured filtering.
        # ---------------------------------------------------------

        evidence_top_k = min(
            self.MAX_EVIDENCE_RESULTS,
            max(
                top_k * self.EVIDENCE_MULTIPLIER,
                len(allowed_doc_ids) * 3,
            ),
        )

        search_results = self.search_service.search(
            query=query,
            knowledge_base_id=knowledge_base_id,
            top_k=evidence_top_k,
            allowed_doc_ids=allowed_doc_ids,
        )

        if not search_results:
            return RecruiterSearchResponse(
                query=query,
                knowledge_base_id=knowledge_base_id,
                results=[],
            )

        # ---------------------------------------------------------
        # 5. Aggregate evidence by candidate
        # ---------------------------------------------------------

        grouped: dict[
            str,
            list[RecruiterEvidence],
        ] = {}

        for result in search_results:

            if not result.doc_id:
                continue

            candidate = candidate_by_document_id.get(
                result.doc_id
            )

            if candidate is None:
                # Defensive check.
                #
                # Restricted retrieval should already guarantee
                # this cannot happen, but keeping the check makes
                # the service robust against stale index data.
                continue

            evidence = RecruiterEvidence(
                text=result.text,
                snippet=result.snippet,
                source=result.source,
                page_number=result.page_number,
                chunk_index=result.chunk_index,
                dense_score=result.dense_score,
                bm25_score=result.bm25_score,
                hybrid_score=result.hybrid_score,
                rerank_score=result.rerank_score,
                doc_id=result.doc_id,
            )

            grouped.setdefault(
                candidate.candidate_id,
                [],
            ).append(evidence)

        # ---------------------------------------------------------
        # 6. Build candidate-level results
        # ---------------------------------------------------------

        results: list[RecruiterCandidateResult] = []

        for candidate in candidates_with_resumes:

            evidence = grouped.get(
                candidate.candidate_id,
                [],
            )

            if not evidence:
                continue

            # CrossEncoder scores individual chunks.
            #
            # We keep only the strongest few pieces of evidence
            # for each candidate.
            evidence = sorted(
                evidence,
                key=lambda item: item.rerank_score,
                reverse=True,
            )[:self.MAX_EVIDENCE_PER_CANDIDATE]

            # Candidate-level evidence score.
            #
            # The strongest relevant resume passage is the primary
            # signal that this candidate has evidence for the query.
            best_score = evidence[0].rerank_score

            results.append(
                RecruiterCandidateResult(
                    candidate_id=candidate.candidate_id,
                    full_name=candidate.full_name,
                    email=candidate.email,
                    location=candidate.location,
                    current_title=candidate.current_title,
                    current_company=candidate.current_company,
                    years_experience=candidate.years_experience,
                    skills=candidate.skills,
                    availability=candidate.availability,
                    resume_filename=candidate.resume_filename,
                    best_score=best_score,
                    evidence=evidence,
                    resume_document_id=(
                        candidate.resume_document_id
                    ),
                )
            )

        # ---------------------------------------------------------
        # 7. Final candidate-level ranking
        # ---------------------------------------------------------
        #
        # CrossEncoder already ranked CHUNKS.
        #
        # This ranking now ranks CANDIDATES.
        #
        # Primary:
        #   strongest resume evidence
        #
        # Tie-breaker 1:
        #   years of experience
        #
        # Tie-breaker 2:
        #   amount of supporting evidence
        # ---------------------------------------------------------

        results.sort(
            key=lambda item: (
                item.best_score,
                (
                    item.years_experience
                    if item.years_experience is not None
                    else -1.0
                ),
                len(item.evidence),
            ),
            reverse=True,
        )

        # ---------------------------------------------------------
        # 8. Final top-K candidates
        # ---------------------------------------------------------

        return RecruiterSearchResponse(
            query=query,
            knowledge_base_id=knowledge_base_id,
            results=results[:top_k],
        )