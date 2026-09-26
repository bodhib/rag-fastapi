from io import BytesIO

from fastapi.testclient import TestClient

import app.lifecycle as lifecycle
from app.main import app

from app.background_ingestion import BackgroundIngestionService
from app.bm25_retriever import BM25Retriever
from app.compressor import ContextCompressor
from app.document_processor import DocumentProcessor
from app.document_store import DocumentStore
from app.embeddings import EmbeddedChunk
from app.file_storage import UploadedFileStorage
from app.hybrid_retriever import HybridRetriever
from app.ingestion_job_store import IngestionJobStore
from app.ingestion_service import DocumentIngestionService
from app.query_service import QueryService
from app.retrieval_index import RetrievalIndexManager
from app.reranker import RerankedResult
from app.schemas import QueryResponse
from app.security import Principal, UserRole
from app.vector_store import ChromaVectorStore

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database import Base

class FakeEmbeddingService:
    """
    Deterministic embedding service used by integration tests.


    It avoids loading the real SentenceTransformer model while
    preserving the same interface used by the ingestion and
    retrieval pipeline.
    """

    def __init__(self) -> None:
        self._dimension = 4

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed_text(self, text: str) -> list[float]:
        if not text.strip():
            raise ValueError(
                "Cannot generate embedding for empty text"
            )

        return self._create_embedding(text)

    def embed_chunks(
        self,
        chunks,
    ) -> list[EmbeddedChunk]:
        if not chunks:
            raise ValueError(
                "Cannot generate embeddings for empty chunk list"
            )

        return [
            EmbeddedChunk(
                chunk=chunk,
                embedding=self._create_embedding(chunk.text),
            )
            for chunk in chunks
        ]

    @staticmethod
    def _create_embedding(text: str) -> list[float]:
        """
        Create a deterministic four-dimensional vector.

        The exact values are not important for this integration test.
        We only need valid, consistent vectors for Chroma.
        """

        text_lower = text.lower()

        return [
            1.0 if "annual leave" in text_lower else 0.5,
            1.0 if "carry forward" in text_lower else 0.5,
            1.0 if "employees" in text_lower else 0.5,
            1.0,
        ]


class FakeReranker:
    """
    Deterministic reranker used by the integration test.
    """


    def rerank(
        self,
        query: str,
        candidates,
        top_k: int = 5,
    ) -> list[RerankedResult]:

        if not query.strip():
            raise ValueError(
                "query cannot be empty"
            )

        if top_k <= 0:
            raise ValueError(
                "top_k must be greater than zero"
            )

        results = []

        for index, candidate in enumerate(candidates):
            results.append(
                RerankedResult(
                    text=candidate.text,
                    source=candidate.source,
                    page_number=candidate.page_number,
                    chunk_index=candidate.chunk_index,
                    dense_score=candidate.dense_score,
                    bm25_score=candidate.bm25_score,
                    hybrid_score=candidate.score,
                    rerank_score=float(
                        len(candidates) - index
                    ),
                )
            )

        return results[:top_k]


class FakeGenerator:
    """
    Deterministic generator used by the integration test.


    This replaces Gemini so the test does not require an API key
    or external network access.
    """

    def generate(
        self,
        query: str,
        contexts,
    ):
        if not query.strip():
            raise ValueError(
                "query cannot be empty"
            )

        if not contexts:
            raise ValueError(
                "contexts cannot be empty"
            )

        from app.generator import GenerationResult

        return GenerationResult(
            answer=(
                "Employees can carry forward up to "
                "5 unused annual leave days."
            ),
            contexts=contexts,
        )


def test_fastapi_lifespan_starts_successfully(
tmp_path,
monkeypatch,
):
    """
    Verify that FastAPI starts successfully through its
    configured lifespan using isolated test resources.
    """


    vector_store = ChromaVectorStore(
        persist_directory=tmp_path / "chroma",
    )

    bm25_retriever = BM25Retriever()

    retrieval_index_manager = RetrievalIndexManager(
        lexical_index=bm25_retriever,
    )

    monkeypatch.setattr(
        lifecycle,
        "get_retrieval_index_manager",
        lambda: retrieval_index_manager,
    )

    with TestClient(app) as client:
        response = client.get("/health")

        assert response.status_code == 200

        assert response.json() == {
            "status": "ok",
            "service": "rag-api",
        }


def test_document_upload_then_query(
tmp_path,
monkeypatch,
):
    """
    Test the complete FastAPI document ingestion and query flow.


    The test uses:
        - real FastAPI
        - real lifespan
        - real file storage
        - real document processor
        - real chunker
        - real Chroma
        - real BM25
        - real hybrid retrieval
        - real compression
        - real background ingestion orchestration

    Expensive external/model-dependent components are replaced with
    deterministic fakes.
    """

    # ---------------------------------------------------------
    # 1. Create isolated persistent resources
    # ---------------------------------------------------------

    vector_store = ChromaVectorStore(
        persist_directory=tmp_path / "chroma",
    )

    bm25_retriever = BM25Retriever()

    retrieval_index_manager = RetrievalIndexManager(
        lexical_index=bm25_retriever,
    )

    file_storage = UploadedFileStorage(
        storage_directory=tmp_path / "uploads",
    )

    database_url = f"sqlite:///{tmp_path / 'documents.db'}"

    engine = create_engine(database_url)

    Base.metadata.create_all(bind=engine)

    session_factory = sessionmaker(
        bind=engine,
        autocommit=False,
        autoflush=False,
    )

    document_store = DocumentStore(
        session_factory=session_factory,
    )

    ingestion_job_store = IngestionJobStore(
        session_factory=session_factory,
    )

    # ---------------------------------------------------------
    # 2. Create real ingestion pipeline components
    # ---------------------------------------------------------

    document_processor = DocumentProcessor()

    from app.chunker import RecursiveChunker

    chunker = RecursiveChunker(
        chunk_size=500,
        chunk_overlap=50,
    )

    embedding_service = FakeEmbeddingService()

    ingestion_service = DocumentIngestionService(
        document_processor=document_processor,
        chunker=chunker,
        embedding_service=embedding_service,
        vector_store=vector_store,
        retrieval_index_manager=retrieval_index_manager,
        document_store=document_store,
    )

    # ---------------------------------------------------------
    # 3. Create background ingestion service
    # ---------------------------------------------------------

    background_ingestion_service = BackgroundIngestionService(
        ingestion_service=ingestion_service,
        job_store=ingestion_job_store,
    )

    # ---------------------------------------------------------
    # 4. Create real retrieval pipeline
    # ---------------------------------------------------------

    from app.retrieval import DenseRetriever

    dense_retriever = DenseRetriever(
        embedding_service=embedding_service,
        vector_store=vector_store,
    )

    hybrid_retriever = HybridRetriever(
        dense_retriever=dense_retriever,
        lexical_retriever=bm25_retriever,
    )

    reranker = FakeReranker()

    compressor = ContextCompressor()

    generator = FakeGenerator()

    query_service = QueryService(
        hybrid_retriever=hybrid_retriever,
        reranker=reranker,
        compressor=compressor,
        generator=generator,
    )

    # ---------------------------------------------------------
    # 5. Override FastAPI dependencies
    # ---------------------------------------------------------

    from app.dependencies import (
        get_background_ingestion_service,
        get_file_storage,
        get_ingestion_job_store,
        get_query_service,
    )

    app.dependency_overrides[
        get_file_storage
    ] = lambda: file_storage

    app.dependency_overrides[
        get_ingestion_job_store
    ] = lambda: ingestion_job_store

    app.dependency_overrides[
        get_background_ingestion_service
    ] = lambda: background_ingestion_service

    app.dependency_overrides[
        get_query_service
    ] = lambda: query_service

    # ---------------------------------------------------------
    # 6. Override authentication
    # ---------------------------------------------------------

    from app.security import get_current_principal

    app.dependency_overrides[
        get_current_principal
    ] = lambda: Principal(
        name="test-admin",
        role=UserRole.ADMIN,
    )

    # ---------------------------------------------------------
    # 7. Override lifecycle retrieval manager
    # ---------------------------------------------------------

    monkeypatch.setattr(
        lifecycle,
        "get_retrieval_index_manager",
        lambda: retrieval_index_manager,
    )

    try:
        # -----------------------------------------------------
        # 8. Start the actual FastAPI application
        # -----------------------------------------------------

        with TestClient(app) as client:
            # -------------------------------------------------
            # 9. Upload a real TXT document
            # -------------------------------------------------

            document_content = """
            Employee Annual Leave Policy

            Employees receive annual leave according to
            company policy.

            Employees can carry forward up to 5 unused
            annual leave days into the next calendar year.

            Any unused leave beyond this limit will expire
            at the end of the calendar year.
            """

            response = client.post(
                "/documents/upload",
                files={
                    "file": (
                        "annual_leave.txt",
                        BytesIO(
                            document_content.encode(
                                "utf-8"
                            )
                        ),
                        "text/plain",
                    )
                },
            )

            # -------------------------------------------------
            # 10. Upload should return immediately
            # -------------------------------------------------

            assert response.status_code == 202

            upload_result = response.json()

            assert upload_result["job_id"].startswith(
                "job_"
            )

            assert upload_result["filename"] == (
                "annual_leave.txt"
            )

            assert upload_result["status"] == "pending"

            job_id = upload_result["job_id"]

            # -------------------------------------------------
            # 11. Verify the background job completed
            # -------------------------------------------------

            job_response = client.get(
                f"/documents/jobs/{job_id}"
            )

            assert job_response.status_code == 200

            job_result = job_response.json()

            assert job_result["job_id"] == job_id

            assert job_result["filename"] == (
                "annual_leave.txt"
            )

            assert job_result["status"] == "completed"

            assert job_result["doc_id"].startswith(
                "doc_"
            )

            assert job_result["page_count"] == 1

            assert job_result["chunk_count"] > 0

            doc_id = job_result["doc_id"]

            # -------------------------------------------------
            # 12. Verify document metadata
            # -------------------------------------------------

            stored_document = document_store.get(
                doc_id
            )

            assert stored_document is not None

            assert stored_document.doc_id == doc_id

            assert stored_document.filename == (
                "annual_leave.txt"
            )

            # -------------------------------------------------
            # 13. Verify Chroma contains the chunks
            # -------------------------------------------------

            assert vector_store.count == (
                job_result["chunk_count"]
            )

            stored_chunks = (
                vector_store.get_all_chunks()
            )

            assert len(stored_chunks) == (
                job_result["chunk_count"]
            )

            assert all(
                chunk.doc_id == doc_id
                for chunk in stored_chunks
            )

            # -------------------------------------------------
            # 14. Query the uploaded document
            # -------------------------------------------------

            response = client.post(
                "/query",
                json={
                    "query": (
                        "How many unused annual leave "
                        "days can employees carry forward?"
                    )
                },
            )

            assert response.status_code == 200

            query_result = response.json()

            # -------------------------------------------------
            # 15. Validate API response
            # -------------------------------------------------

            assert query_result["answer"] == (
                "Employees can carry forward up to "
                "5 unused annual leave days."
            )

            assert len(
                query_result["citations"]
            ) > 0

            citation = query_result["citations"][0]

            assert citation["source"] == (
                "annual_leave.txt"
            )

            assert citation["page_number"] == 1

            assert citation["chunk_index"] >= 0

            # -------------------------------------------------
            # 16. Validate response schema
            # -------------------------------------------------

            QueryResponse.model_validate(
                query_result
            )

    finally:
        # -----------------------------------------------------
        # 17. Always clean up dependency overrides
        # -----------------------------------------------------

        app.dependency_overrides.clear()


def test_fastapi_lifespan_clears_dependency_caches(
monkeypatch,
):
    cleanup_called = []


    def fake_get_retrieval_index_manager():
        return object()

    def fake_get_generator():
        return object()

    def fake_clear_dependency_caches():
        cleanup_called.append(True)

    monkeypatch.setattr(
        lifecycle,
        "get_retrieval_index_manager",
        fake_get_retrieval_index_manager,
    )

    monkeypatch.setattr(
        lifecycle,
        "get_generator",
        fake_get_generator,
    )

    monkeypatch.setattr(
        lifecycle,
        "clear_dependency_caches",
        fake_clear_dependency_caches,
    )

    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert cleanup_called == [True]


def test_lifespan_fails_when_generator_initialization_fails(
monkeypatch,
) -> None:
    import pytest


    def fake_get_retrieval_index_manager():
        return object()

    def fake_get_generator():
        raise ValueError(
            "Gemini API key is not configured"
        )

    monkeypatch.setattr(
        lifecycle,
        "get_retrieval_index_manager",
        fake_get_retrieval_index_manager,
    )

    monkeypatch.setattr(
        lifecycle,
        "get_generator",
        fake_get_generator,
    )

    with pytest.raises(
        ValueError,
        match="Gemini API key",
    ):
        with TestClient(app):
            pass


def test_lifespan_fails_when_retrieval_initialization_fails(
monkeypatch,
) -> None:
    import pytest


    def fake_get_retrieval_index_manager():
        raise RuntimeError(
            "Retrieval index initialization failed"
        )

    def fake_get_generator():
        return object()

    monkeypatch.setattr(
        lifecycle,
        "get_retrieval_index_manager",
        fake_get_retrieval_index_manager,
    )

    monkeypatch.setattr(
        lifecycle,
        "get_generator",
        fake_get_generator,
    )

    with pytest.raises(
        RuntimeError,
        match="Retrieval index initialization failed",
    ):
        with TestClient(app):
            pass

