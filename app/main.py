from __future__ import annotations

from typing import TYPE_CHECKING
from uuid import uuid4

from fastapi import (
    BackgroundTasks,
    Depends,
    FastAPI,
    File,
    Form,
    HTTPException,
    UploadFile,
)
from fastapi import Response, status

from app.dependencies import (
    get_background_ingestion_service,
    get_file_storage,
    get_ingestion_job_store,
    get_knowledge_base_store,
    get_query_service,
)

from app.dependencies import (
    get_search_service,
)

from app.dependencies import (
    get_candidate_store,
    get_recruiter_search_service,
    get_recruiter_query_understanding_service
)

from app.candidate_store import (
    CandidateNotFoundError,
    DuplicateCandidateError,
)

from app.recruiter_search_service import (
    RecruiterSearchService,
)

from app.query_understanding import (
    RecruiterQueryUnderstandingService,
)

from app.schemas import (
    CandidateCreateRequest,
    CandidateResponse,
    CandidateUpdateRequest,
    RecruiterSearchRequest,
    RecruiterSearchResponse,
)

from app.health import run_readiness_checks
from app.ingestion_job_store import IngestionJob
from app.lifecycle import lifespan
from app.schemas import (
    Citation,
    DocumentUploadResponse,
    HealthResponse,
    IngestionJobResponse,
    QueryRequest,
    QueryResponse,
    ReadinessResponse,
)

from app.schemas import (
    SearchRequest,
    SearchResponse,
    SearchResultResponse,
    RecruiterCandidateResultResponse,
    RecruiterEvidenceResponse
)

from app.security import (
    UserRole,
    require_role,
)


if TYPE_CHECKING:
    from app.background_ingestion import BackgroundIngestionService
    from app.file_storage import UploadedFileStorage
    from app.ingestion_job_store import IngestionJobStore
    from app.knowledge_base_store import KnowledgeBaseStore
    from app.query_service import QueryService


app = FastAPI(
    title="Production RAG API",
    description="End-to-end Retrieval Augmented Generation backend",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get(
    "/health",
    response_model=HealthResponse,
)
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
    knowledge_base_store: KnowledgeBaseStore = Depends(
        get_knowledge_base_store
    ),
) -> QueryResponse:
    """
    Execute a RAG query against one knowledge base.
    """

    knowledge_base = knowledge_base_store.get(
        request.knowledge_base_id
    )

    if knowledge_base is None:
        raise HTTPException(
            status_code=404,
            detail="Knowledge base not found",
        )

    if not knowledge_base.active:
        raise HTTPException(
            status_code=400,
            detail="Knowledge base is inactive",
        )

    try:
        result = service.query(
            question=request.query,
            knowledge_base_id=request.knowledge_base_id,
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
    "/search",
    response_model=SearchResponse,
)
def search_documents(
    request: SearchRequest,
    search_service=Depends(get_search_service),
    knowledge_base_store=Depends(get_knowledge_base_store),
):
    """
    Search a knowledge base without LLM generation.

    This endpoint performs:

        hybrid retrieval
        +
        CrossEncoder reranking

    and returns ranked evidence for the caller.
    """

    knowledge_base = knowledge_base_store.get(
        request.knowledge_base_id
    )

    if knowledge_base is None:
        raise HTTPException(
            status_code=404,
            detail="Knowledge base not found",
        )

    if not knowledge_base.active:
        raise HTTPException(
            status_code=400,
            detail="Knowledge base is inactive",
        )

    results = search_service.search(
        query=request.query,
        knowledge_base_id=request.knowledge_base_id,
        top_k=request.top_k,
    )

    return SearchResponse(
        query=request.query,
        knowledge_base_id=request.knowledge_base_id,
        results=[
            SearchResultResponse(
                doc_id=result.doc_id,
                text=result.text,
                snippet=result.snippet,
                source=result.source,
                page_number=result.page_number,
                chunk_index=result.chunk_index,
                dense_score=result.dense_score,
                bm25_score=result.bm25_score,
                hybrid_score=result.hybrid_score,
                rerank_score=result.rerank_score,
            )
            for result in results
        ],
    )

@app.post(
    "/recruiter/candidates",
    response_model=CandidateResponse,
)
def create_candidate(
    request: CandidateCreateRequest,
    candidate_store=Depends(get_candidate_store),
    knowledge_base_store=Depends(
        get_knowledge_base_store
    ),
):
    knowledge_base = knowledge_base_store.get(
        request.knowledge_base_id
    )

    if knowledge_base is None:
        raise HTTPException(
            status_code=404,
            detail="Knowledge base not found",
        )

    if not knowledge_base.active:
        raise HTTPException(
            status_code=400,
            detail="Knowledge base is inactive",
        )

    try:
        candidate = candidate_store.create(
            knowledge_base_id=request.knowledge_base_id,
            full_name=request.full_name,
            email=request.email,
            phone=request.phone,
            location=request.location,
            current_title=request.current_title,
            current_company=request.current_company,
            years_experience=request.years_experience,
            skills=request.skills,
            education=request.education,
            availability=request.availability,
            resume_filename=request.resume_filename,
            resume_document_id=request.resume_document_id,
        )
    except DuplicateCandidateError as exc:
        raise HTTPException(
            status_code=409,
            detail=str(exc),
        ) from exc

    return CandidateResponse(
        candidate_id=candidate.candidate_id,
        knowledge_base_id=candidate.knowledge_base_id,
        full_name=candidate.full_name,
        email=candidate.email,
        phone=candidate.phone,
        location=candidate.location,
        current_title=candidate.current_title,
        current_company=candidate.current_company,
        years_experience=candidate.years_experience,
        skills=candidate.skills,
        education=candidate.education,
        availability=candidate.availability,
        resume_filename=candidate.resume_filename,
        resume_document_id=candidate.resume_document_id,
        active=candidate.active,
    )

@app.get(
    "/recruiter/candidates",
    response_model=list[CandidateResponse],
)
def list_candidates(
    knowledge_base_id: str = "kb_recruitment",
    candidate_store=Depends(get_candidate_store),
    knowledge_base_store=Depends(
        get_knowledge_base_store
    ),
):
    knowledge_base = knowledge_base_store.get(
        knowledge_base_id
    )

    if knowledge_base is None:
        raise HTTPException(
            status_code=404,
            detail="Knowledge base not found",
        )

    candidates = candidate_store.list(
        knowledge_base_id=knowledge_base_id,
        active_only=True,
    )

    return [
        CandidateResponse(
            candidate_id=candidate.candidate_id,
            knowledge_base_id=candidate.knowledge_base_id,
            full_name=candidate.full_name,
            email=candidate.email,
            phone=candidate.phone,
            location=candidate.location,
            current_title=candidate.current_title,
            current_company=candidate.current_company,
            years_experience=candidate.years_experience,
            skills=candidate.skills,
            education=candidate.education,
            availability=candidate.availability,
            resume_filename=candidate.resume_filename,
            resume_document_id=candidate.resume_document_id,
            active=candidate.active,
        )
        for candidate in candidates
    ]

@app.get(
    "/recruiter/candidates/{candidate_id}",
    response_model=CandidateResponse,
)
def get_candidate(
    candidate_id: str,
    candidate_store=Depends(get_candidate_store),
):
    candidate = candidate_store.get(candidate_id)

    if candidate is None:
        raise HTTPException(
            status_code=404,
            detail="Candidate not found",
        )

    return CandidateResponse(
        candidate_id=candidate.candidate_id,
        knowledge_base_id=candidate.knowledge_base_id,
        full_name=candidate.full_name,
        email=candidate.email,
        phone=candidate.phone,
        location=candidate.location,
        current_title=candidate.current_title,
        current_company=candidate.current_company,
        years_experience=candidate.years_experience,
        skills=candidate.skills,
        education=candidate.education,
        availability=candidate.availability,
        resume_filename=candidate.resume_filename,
        resume_document_id=candidate.resume_document_id,
        active=candidate.active,
    )

@app.patch(
    "/recruiter/candidates/{candidate_id}",
    response_model=CandidateResponse,
)
def update_candidate(
    candidate_id: str,
    request: CandidateUpdateRequest,
    candidate_store=Depends(get_candidate_store),
):
    changes = request.model_dump(
        exclude_unset=True
    )

    try:
        candidate = candidate_store.update(
            candidate_id,
            **changes,
        )
    except CandidateNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail=str(exc),
        ) from exc

    return CandidateResponse(
        candidate_id=candidate.candidate_id,
        knowledge_base_id=candidate.knowledge_base_id,
        full_name=candidate.full_name,
        email=candidate.email,
        phone=candidate.phone,
        location=candidate.location,
        current_title=candidate.current_title,
        current_company=candidate.current_company,
        years_experience=candidate.years_experience,
        skills=candidate.skills,
        education=candidate.education,
        availability=candidate.availability,
        resume_filename=candidate.resume_filename,
        resume_document_id=candidate.resume_document_id,
        active=candidate.active,
    )

@app.post(
    "/recruiter/search",
    response_model=RecruiterSearchResponse,
)
def recruiter_search(
    request: RecruiterSearchRequest,
    recruiter_search_service=Depends(
        get_recruiter_search_service
    ),
    knowledge_base_store=Depends(
        get_knowledge_base_store
    ),
):
    knowledge_base = knowledge_base_store.get(
        request.knowledge_base_id
    )

    if knowledge_base is None:
        raise HTTPException(
            status_code=404,
            detail="Knowledge base not found",
        )

    if not knowledge_base.active:
        raise HTTPException(
            status_code=400,
            detail="Knowledge base is inactive",
        )

    result = recruiter_search_service.search(
        query=request.query,
        knowledge_base_id=request.knowledge_base_id,
        top_k=request.top_k,
        skills=request.skills,
        min_years_experience=(
            request.min_years_experience
        ),
        max_years_experience=(
            request.max_years_experience
        ),
        location=request.location,
        title=request.title,
    )

    return RecruiterSearchResponse(
        query=result.query,
        knowledge_base_id=result.knowledge_base_id,
        results=[
            RecruiterCandidateResultResponse(
                candidate_id=item.candidate_id,
                full_name=item.full_name,
                email=item.email,
                location=item.location,
                current_title=item.current_title,
                current_company=item.current_company,
                years_experience=item.years_experience,
                skills=item.skills,
                availability=item.availability,
                resume_filename=item.resume_filename,
                resume_document_id=item.resume_document_id,
                best_score=item.best_score,
                evidence=[
                    RecruiterEvidenceResponse(
                        doc_id=evidence.doc_id,
                        text=evidence.text,
                        snippet=evidence.snippet,
                        source=evidence.source,
                        page_number=evidence.page_number,
                        chunk_index=evidence.chunk_index,
                        dense_score=evidence.dense_score,
                        bm25_score=evidence.bm25_score,
                        hybrid_score=evidence.hybrid_score,
                        rerank_score=evidence.rerank_score,
                    )
                    for evidence in item.evidence
                ],
            )
            for item in result.results
        ],
    )

@app.post(
    "/debug/job-test",
    dependencies=[
        Depends(
            require_role(UserRole.ADMIN)
        )
    ],
)
def debug_job_test(
    job_store: IngestionJobStore = Depends(
        get_ingestion_job_store
    ),
) -> dict:
    """
    Temporary production diagnostic endpoint.

    Creates a minimal ingestion job using the default
    knowledge base.
    """

    job_id = f"debug_{uuid4().hex}"

    job_store.create(
        IngestionJob(
            job_id=job_id,
            filename="debug.txt",
            file_path="/tmp/debug.txt",
            status="pending",
            knowledge_base_id="kb_default",
        )
    )

    return {
        "job_id": job_id,
        "status": "pending",
    }


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
    knowledge_base_id: str = Form(...),
    file_storage: UploadedFileStorage = Depends(
        get_file_storage
    ),
    job_store: IngestionJobStore = Depends(
        get_ingestion_job_store
    ),
    knowledge_base_store: KnowledgeBaseStore = Depends(
        get_knowledge_base_store
    ),
    background_ingestion_service: BackgroundIngestionService = Depends(
        get_background_ingestion_service
    ),
) -> DocumentUploadResponse:

    knowledge_base_id = knowledge_base_id.strip()

    if not knowledge_base_id:
        raise HTTPException(
            status_code=400,
            detail="knowledge_base_id cannot be empty",
        )

    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file must have a filename",
        )

    knowledge_base = knowledge_base_store.get(
        knowledge_base_id
    )

    if knowledge_base is None:
        raise HTTPException(
            status_code=404,
            detail="Knowledge base not found",
        )

    if not knowledge_base.active:
        raise HTTPException(
            status_code=400,
            detail="Knowledge base is inactive",
        )

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
            knowledge_base_id=knowledge_base_id,
        )
    )

    background_tasks.add_task(
        background_ingestion_service.execute,
        job_id,
    )

    return DocumentUploadResponse(
        job_id=job_id,
        filename=file.filename,
        status="pending",
    )


@app.post("/debug/upload-test")
def debug_upload_test(
    file: UploadFile = File(...),
) -> dict:
    return {
        "filename": file.filename,
        "content_type": file.content_type,
    }


@app.post("/debug/storage-test")
def debug_storage_test(
    file: UploadFile = File(...),
    file_storage: UploadedFileStorage = Depends(
        get_file_storage
    ),
) -> dict:

    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file must have a filename",
        )

    stored_path = file_storage.save(
        source_file=file.file,
        filename=file.filename,
    )

    return {
        "filename": file.filename,
        "stored_path": str(stored_path),
    }


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

    job = job_store.get(
        job_id
    )

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

@app.post("/recruiter/search")
def recruiter_search(
    request: RecruiterSearchRequest,
    recruiter_search_service: RecruiterSearchService = Depends(
        get_recruiter_search_service
    ),
    query_understanding_service: RecruiterQueryUnderstandingService = Depends(
        get_recruiter_query_understanding_service
    ),
    ):
        intent = query_understanding_service.understand(request.query)

        return recruiter_search_service.search(
            query=intent.semantic_query,
            knowledge_base_id=request.knowledge_base_id,
            top_k=request.top_k,
            skills = (
                request.skills
                if request.skills
                else intent.skills
            ),

            min_years_experience = (
                request.min_years_experience
                if request.min_years_experience is not None
                else intent.min_years_experience
            ),

            max_years_experience = (
                request.max_years_experience
                if request.max_years_experience is not None
                else intent.max_years_experience
            ),

            location = (
                request.location
                if request.location is not None
                else intent.location
            ),

            title = (
                request.title
                if request.title is not None
                else intent.title
            ),
        )