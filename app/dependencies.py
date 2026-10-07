from __future__ import annotations

from functools import lru_cache

from app.config import settings
from app.database import SessionLocal

from app.query_understanding import RecruiterQueryUnderstandingService

from app.config import get_settings


@lru_cache
def get_document_processor():
    from app.document_processor import DocumentProcessor

    return DocumentProcessor()


@lru_cache
def get_chunker():
    from app.chunker import RecursiveChunker

    return RecursiveChunker(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )


@lru_cache
def get_embedding_service():
    from app.embeddings import EmbeddingService

    return EmbeddingService(
        config=settings,
    )


@lru_cache
def get_vector_store():
    from app.vector_store import ChromaVectorStore

    return ChromaVectorStore(
        persist_directory=settings.chroma_directory,
    )


@lru_cache
def get_bm25_retriever():
    from app.bm25_retriever import BM25Retriever

    return BM25Retriever()


@lru_cache
def get_postgres_fts_retriever():
    from app.postgres_fts_retriever import PostgresFTSRetriever

    return PostgresFTSRetriever(
        session_factory=SessionLocal,
    )


@lru_cache
def get_lexical_retriever():
    if settings.lexical_backend == "bm25":
        return get_bm25_retriever()

    if settings.lexical_backend == "postgres_fts":
        return get_postgres_fts_retriever()

    raise ValueError(
        f"Unsupported lexical backend: "
        f"{settings.lexical_backend}"
    )


@lru_cache
def get_retrieval_index_manager():
    from app.retrieval_index import RetrievalIndexManager

    manager = RetrievalIndexManager(
        lexical_index=get_lexical_retriever(),
    )

    chunks = get_vector_store().get_all_chunks()

    if chunks:
        manager.rebuild(
            chunks
        )

    return manager


@lru_cache
def get_document_store():
    from app.document_store import DocumentStore

    return DocumentStore(
        session_factory=SessionLocal,
    )


@lru_cache
def get_knowledge_base_store():
    from app.knowledge_base_store import KnowledgeBaseStore

    return KnowledgeBaseStore(
        session_factory=SessionLocal,
    )


@lru_cache
def get_file_storage():
    from app.file_storage import UploadedFileStorage

    return UploadedFileStorage(
        storage_directory=settings.upload_directory,
    )


@lru_cache
def get_ingestion_service():
    from app.ingestion_service import DocumentIngestionService

    return DocumentIngestionService(
        document_processor=get_document_processor(),
        chunker=get_chunker(),
        embedding_service=get_embedding_service(),
        vector_store=get_vector_store(),
        retrieval_index_manager=get_retrieval_index_manager(),
        document_store=get_document_store(),
        candidate_store=get_candidate_store(),
    )


def create_ingestion_service():
    """
    Create an ingestion service lazily.

    This factory intentionally avoids creating heavy ML components
    until background ingestion actually starts.
    """

    from app.ingestion_service import DocumentIngestionService

    return DocumentIngestionService(
        document_processor=get_document_processor(),
        chunker=get_chunker(),
        embedding_service=get_embedding_service(),
        vector_store=get_vector_store(),
        retrieval_index_manager=get_retrieval_index_manager(),
        document_store=get_document_store(),
        candidate_store=get_candidate_store(),
    )


@lru_cache
def get_dense_retriever():
    from app.retrieval import DenseRetriever

    return DenseRetriever(
        embedding_service=get_embedding_service(),
        vector_store=get_vector_store(),
    )


@lru_cache
def get_hybrid_retriever():
    from app.hybrid_retriever import HybridRetriever

    return HybridRetriever(
        dense_retriever=get_dense_retriever(),
        lexical_retriever=get_lexical_retriever(),
    )


@lru_cache
def get_reranker():
    from app.reranker import CrossEncoderReranker

    return CrossEncoderReranker(
        config=settings,
    )


@lru_cache
def get_compressor():
    from app.compressor import ContextCompressor

    return ContextCompressor()


@lru_cache
def get_generator():
    from app.generator import GeminiGenerator

    return GeminiGenerator(
        config=settings,
    )


@lru_cache
def get_query_service():
    from app.query_service import QueryService

    return QueryService(
        hybrid_retriever=get_hybrid_retriever(),
        reranker=get_reranker(),
        compressor=get_compressor(),
        generator=get_generator(),
        config=settings,
    )


@lru_cache
def get_search_service():
    from app.search_service import SearchService

    return SearchService(
        hybrid_retriever=get_hybrid_retriever(),
        reranker=get_reranker(),
        config=settings,
    )

@lru_cache
def get_ingestion_job_store():
    from app.ingestion_job_store import IngestionJobStore

    return IngestionJobStore(
        session_factory=SessionLocal,
    )


@lru_cache
def get_background_ingestion_service():
    from app.background_ingestion import (
        BackgroundIngestionService,
    )

    return BackgroundIngestionService(
        ingestion_service_factory=create_ingestion_service,
        job_store_factory=get_ingestion_job_store,
    )

@lru_cache
def get_candidate_store():
    from app.candidate_store import CandidateStore

    return CandidateStore(
        session_factory=SessionLocal,
        document_store=get_document_store(),
    )


@lru_cache
def get_recruiter_search_service():
    from app.recruiter_search_service import (
        RecruiterSearchService,
    )

    return RecruiterSearchService(
        candidate_store=get_candidate_store(),
        search_service=get_search_service(),
    )

@lru_cache
def get_recruiter_query_understanding_service() -> RecruiterQueryUnderstandingService:
    return RecruiterQueryUnderstandingService(
        settings=get_settings(),
    )


def clear_dependency_caches() -> None:
    """
    Clear all cached application dependency instances.

    This is primarily useful during application shutdown and tests.
    """

    get_query_service.cache_clear()
    get_search_service.cache_clear()
    get_generator.cache_clear()
    get_compressor.cache_clear()
    get_reranker.cache_clear()

    get_hybrid_retriever.cache_clear()
    get_dense_retriever.cache_clear()

    get_lexical_retriever.cache_clear()
    get_postgres_fts_retriever.cache_clear()
    get_bm25_retriever.cache_clear()

    get_retrieval_index_manager.cache_clear()

    get_embedding_service.cache_clear()
    get_vector_store.cache_clear()

    get_background_ingestion_service.cache_clear()
    get_ingestion_job_store.cache_clear()

    get_ingestion_service.cache_clear()
    get_file_storage.cache_clear()
    get_document_store.cache_clear()
    get_knowledge_base_store.cache_clear()

    get_recruiter_search_service.cache_clear()
    get_candidate_store.cache_clear()

    get_chunker.cache_clear()
    get_document_processor.cache_clear()