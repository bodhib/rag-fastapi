"""
API request and response schemas.

This module contains Pydantic models used by the FastAPI endpoints.
"""

from pydantic import (
    BaseModel,
    Field,
    field_validator,
    model_validator,
)


class QueryRequest(BaseModel):
    """
    Represent a user's RAG query request.
    """

    query: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="Natural-language question",
    )

    knowledge_base_id: str = Field(
        default="kb_default",
        min_length=1,
        max_length=64,
        description="Knowledge base used for retrieval",
    )

    @field_validator("query")
    @classmethod
    def validate_query(
        cls,
        value: str,
    ) -> str:
        value = value.strip()

        if not value:
            raise ValueError(
                "query cannot be empty"
            )

        return value

    @field_validator("knowledge_base_id")
    @classmethod
    def validate_knowledge_base_id(
        cls,
        value: str,
    ) -> str:
        value = value.strip()

        if not value:
            raise ValueError(
                "knowledge_base_id cannot be empty"
            )

        return value


class Citation(BaseModel):
    """
    Represent a source used as context for an answer.
    """

    source: str
    page_number: int
    chunk_index: int


class QueryResponse(BaseModel):
    """
    Represent the response returned by the RAG query endpoint.
    """

    answer: str
    citations: list[Citation]


class DocumentUploadResponse(BaseModel):
    """
    Response returned after a document has been accepted
    for background ingestion.
    """

    job_id: str
    filename: str
    status: str


class IngestionJobResponse(BaseModel):
    """
    Current state of a document ingestion job.
    """

    job_id: str
    filename: str
    status: str
    doc_id: str | None = None
    page_count: int | None = None
    chunk_count: int | None = None
    error_message: str | None = None


class HealthResponse(BaseModel):
    status: str
    service: str


class ReadinessCheckResponse(BaseModel):
    ok: bool


class ReadinessResponse(BaseModel):
    status: str
    service: str
    checks: dict[str, ReadinessCheckResponse]

class SearchRequest(BaseModel):
    """
    Request for enterprise knowledge search.
    """

    query: str = Field(
        ...,
        min_length=1,
        max_length=2000,
    )

    knowledge_base_id: str = Field(
        default="kb_default",
        min_length=1,
        max_length=64,
    )

    top_k: int = Field(
        default=10,
        ge=1,
        le=50,
    )

    @field_validator("query", "knowledge_base_id")
    @classmethod
    def reject_whitespace_only(
        cls,
        value: str,
    ) -> str:
        if not value.strip():
            raise ValueError(
                "value cannot be empty"
            )

        return value


class SearchResultResponse(BaseModel):
    """
    One ranked search result.
    """

    doc_id: str
    text: str
    snippet: str
    source: str
    page_number: int
    chunk_index: int
    dense_score: float
    bm25_score: float
    hybrid_score: float
    rerank_score: float


class SearchResponse(BaseModel):
    """
    Enterprise search response.
    """

    query: str
    knowledge_base_id: str
    results: list[SearchResultResponse]    

class CandidateCreateRequest(BaseModel):
    knowledge_base_id: str = Field(
        default="kb_recruitment",
        min_length=1,
        max_length=64,
    )

    full_name: str = Field(
        ...,
        min_length=1,
        max_length=255,
    )

    email: str | None = Field(
        default=None,
        max_length=320,
    )

    phone: str | None = Field(
        default=None,
        max_length=64,
    )

    location: str | None = Field(
        default=None,
        max_length=255,
    )

    current_title: str | None = Field(
        default=None,
        max_length=255,
    )

    current_company: str | None = Field(
        default=None,
        max_length=255,
    )

    years_experience: float | None = Field(
        default=None,
        ge=0,
        le=60,
    )

    skills: list[str] = Field(
        default_factory=list,
        max_length=100,
    )

    education: str | None = Field(
        default=None,
        max_length=2000,
    )

    availability: str | None = Field(
        default=None,
        max_length=128,
    )

    resume_filename: str = Field(
        ...,
        min_length=1,
        max_length=255,
    )

    resume_document_id: str | None = Field(
        default=None,
        max_length=64,
    )

    @field_validator(
        "knowledge_base_id",
        "full_name",
        "resume_filename",
    )
    @classmethod
    def reject_whitespace_only(
        cls,
        value: str,
    ) -> str:

        if not value.strip():
            raise ValueError(
                "value cannot be empty"
            )

        return value


class CandidateUpdateRequest(BaseModel):
    full_name: str | None = Field(
        default=None,
        min_length=1,
        max_length=255,
    )

    email: str | None = Field(
        default=None,
        max_length=320,
    )

    phone: str | None = Field(
        default=None,
        max_length=64,
    )

    location: str | None = Field(
        default=None,
        max_length=255,
    )

    current_title: str | None = Field(
        default=None,
        max_length=255,
    )

    current_company: str | None = Field(
        default=None,
        max_length=255,
    )

    years_experience: float | None = Field(
        default=None,
        ge=0,
        le=60,
    )

    skills: list[str] | None = Field(
        default=None,
        max_length=100,
    )

    education: str | None = Field(
        default=None,
        max_length=2000,
    )

    availability: str | None = Field(
        default=None,
        max_length=128,
    )

    resume_filename: str | None = Field(
        default=None,
        min_length=1,
        max_length=255,
    )

    resume_document_id: str | None = Field(
        default=None,
        max_length=64,
    )

    active: bool | None = None


class CandidateResponse(BaseModel):
    candidate_id: str
    knowledge_base_id: str
    full_name: str
    email: str | None
    phone: str | None
    location: str | None
    current_title: str | None
    current_company: str | None
    years_experience: float | None
    skills: list[str]
    education: str | None
    availability: str | None
    resume_filename: str
    resume_document_id: str | None
    active: bool


class RecruiterSearchRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=1,
        max_length=2000,
    )

    knowledge_base_id: str = Field(
        default="kb_recruitment",
        min_length=1,
        max_length=64,
    )

    top_k: int = Field(
        default=10,
        ge=1,
        le=50,
    )

    skills: list[str] = Field(
        default_factory=list,
        max_length=100,
    )

    min_years_experience: float | None = Field(
        default=None,
        ge=0,
        le=60,
    )

    max_years_experience: float | None = Field(
        default=None,
        ge=0,
        le=60,
    )

    location: str | None = Field(
        default=None,
        max_length=255,
    )

    title: str | None = Field(
        default=None,
        max_length=255,
    )

    @field_validator(
        "query",
        "knowledge_base_id",
    )
    @classmethod
    def reject_whitespace_only(
        cls,
        value: str,
    ) -> str:

        if not value.strip():
            raise ValueError(
                "value cannot be empty"
            )

        return value

    @model_validator(mode="after")
    def validate_experience_range(self):
        if (
            self.min_years_experience is not None
            and self.max_years_experience is not None
            and self.min_years_experience
            > self.max_years_experience
        ):
            raise ValueError(
                "min_years_experience cannot be greater "
                "than max_years_experience"
            )

        return self


class RecruiterEvidenceResponse(BaseModel):
    doc_id: str
    text: str
    snippet: str
    source: str
    page_number: int
    chunk_index: int
    dense_score: float
    bm25_score: float
    hybrid_score: float
    rerank_score: float


class RecruiterCandidateResultResponse(BaseModel):
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
    resume_document_id: str | None
    best_score: float
    evidence: list[RecruiterEvidenceResponse]


class RecruiterSearchResponse(BaseModel):
    query: str
    knowledge_base_id: str
    results: list[RecruiterCandidateResultResponse]    