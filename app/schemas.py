"""
API request and response schemas.

This module contains Pydantic models used by the FastAPI endpoints.
"""

from pydantic import BaseModel, Field, field_validator


class QueryRequest(BaseModel):
    """
    Represent a user's RAG query request.

    Attributes:
        query: Natural-language question submitted by the user.
    """

    query: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="Natural-language question",
    )

    @field_validator("query")
    @classmethod
    def validate_query(cls, value: str) -> str:
        """
        Validate and normalize the query.

        Args:
            value: User-supplied query.

        Returns:
            Stripped query text.

        Raises:
            ValueError: If the query contains only whitespace.
        """

        value = value.strip()

        if not value:
            raise ValueError(
                "query cannot be empty"
            )

        return value


class Citation(BaseModel):
    """
    Represent a source used as context for an answer.

    Attributes:
        source: Original document filename.
        page_number: Original page number.
        chunk_index: Position of the chunk within the document.
    """

    source: str
    page_number: int
    chunk_index: int


class QueryResponse(BaseModel):
    """
    Represent the response returned by the RAG query endpoint.

    Attributes:
        answer: Generated answer.
        citations: Source chunks supplied to the generator.
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
  