from __future__ import annotations

from uuid import uuid4

from fastapi import (
    BackgroundTasks,
    Depends,
    FastAPI,
    File,
    HTTPException,
    UploadFile,
)

from fastapi import Response, status

from app.health import run_readiness_checks
from app.schemas import (
    HealthResponse,
    ReadinessResponse,
)

from app.security import (
    UserRole,
    require_role,
)

from app.dependencies import (
    get_background_ingestion_service,
    get_file_storage,
    get_ingestion_job_store,
    get_query_service,
)
from app.schemas import (
    Citation,
    DocumentUploadResponse,
    IngestionJobResponse,
    QueryRequest,
    QueryResponse,
)
from app.lifecycle import lifespan

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.background_ingestion import BackgroundIngestionService
    from app.file_storage import UploadedFileStorage
    from app.ingestion_job_store import IngestionJob, IngestionJobStore
    from app.query_service import QueryService


app = FastAPI(
    title="Production RAG API",
    description="End-to-end Retrieval Augmented Generation backend",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse,)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        service="rag-api",
    )


@app.get(
    "/health/ready",
    response_model=ReadinessResponse,
)
def readiness(
    response: Response,
) -> ReadinessResponse:
    result = run_readiness_checks()

    readiness_response = ReadinessResponse(
        status=(
            "ready"
            if result.ready
            else "not_ready"
        ),
        service="rag-api",
        checks={
            check.name: {
                "ok": check.ok,
            }
            for check in result.checks
        },
    )

    if not result.ready:
        response.status_code = (
            status.HTTP_503_SERVICE_UNAVAILABLE
        )

    return readiness_response


@app.post(
    "/query",
    response_model=QueryResponse,
    dependencies=[
        Depends(
            require_role(UserRole.USER)
        )
    ],
)
def query(
    request: QueryRequest,
    service: QueryService = Depends(
        get_query_service
    ),
) -> QueryResponse:
    """Execute a RAG query."""

    try:
        result = service.query(
            question=request.query
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=404,
            detail=str(exc),
        ) from exc

    citations = [
        Citation(
            source=context.source,
            page_number=context.page_number,
            chunk_index=context.chunk_index,
        )
        for context in result.contexts
    ]

    return QueryResponse(
        answer=result.answer,
        citations=citations,
    )


@app.post(
    "/documents/upload",
    response_model=DocumentUploadResponse,
    status_code=202,
    dependencies=[
        Depends(
            require_role(UserRole.ADMIN)
        )
    ],
)
def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    file_storage: UploadedFileStorage = Depends(
        get_file_storage
    ),
    job_store: IngestionJobStore = Depends(
        get_ingestion_job_store
    ),
    background_ingestion_service: BackgroundIngestionService = Depends(
        get_background_ingestion_service
    ),
) -> DocumentUploadResponse:
    """
    Upload a document and schedule it for background ingestion.

    Supported formats:
    PDF, TXT, and DOCX.
    """

    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file must have a filename",
        )

    try:
        stored_path = file_storage.save(
            source_file=file.file,
            filename=file.filename,
        )

        job_id = f"job_{uuid4().hex}"

        job_store.create(
            IngestionJob(
                job_id=job_id,
                filename=file.filename,
                file_path=str(stored_path),
                status="pending",
            )
        )

        # background_tasks.add_task(
        #     background_ingestion_service.execute,
        #     job_id,
        # )

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Document upload failed",
        ) from exc

    return DocumentUploadResponse(
        job_id=job_id,
        filename=file.filename,
        status="pending",
    )


@app.get(
    "/documents/jobs/{job_id}",
    response_model=IngestionJobResponse,
)
def get_ingestion_job(
    job_id: str,
    job_store: IngestionJobStore = Depends(
        get_ingestion_job_store
    ),
) -> IngestionJobResponse:
    """
    Return the current state of an ingestion job.
    """

    job = job_store.get(job_id)

    if job is None:
        raise HTTPException(
            status_code=404,
            detail=f"Ingestion job not found: {job_id}",
        )

    return IngestionJobResponse(
        job_id=job.job_id,
        filename=job.filename,
        status=job.status,
        doc_id=job.doc_id,
        page_count=job.page_count,
        chunk_count=job.chunk_count,
        error_message=job.error_message,
    )
