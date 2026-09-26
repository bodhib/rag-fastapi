from functools import lru_cache

from app.bm25_retriever import BM25Retriever, LexicalIndex
from app.postgres_fts_retriever import PostgresFTSRetriever
from app.chunker import RecursiveChunker
from app.compressor import ContextCompressor
from app.config import Settings, settings
from app.document_processor import DocumentProcessor
from app.document_store import DocumentStore
from app.embeddings import EmbeddingService
from app.file_storage import UploadedFileStorage
from app.generator import GeminiGenerator
from app.hybrid_retriever import HybridRetriever
from app.ingestion_service import DocumentIngestionService
from app.query_service import QueryService
from app.retrieval_index import RetrievalIndexManager
from app.reranker import CrossEncoderReranker
from app.vector_store import ChromaVectorStore

from app.database import SessionLocal

from app.background_ingestion import BackgroundIngestionService
from app.ingestion_job_store import IngestionJobStore


@lru_cache
def get_document_processor() -> DocumentProcessor:
    return DocumentProcessor()


@lru_cache
def get_chunker() -> RecursiveChunker:
    return RecursiveChunker(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )


@lru_cache
def get_embedding_service() -> EmbeddingService:
    return EmbeddingService(
        config=settings,
    )


@lru_cache
def get_vector_store() -> ChromaVectorStore:
    return ChromaVectorStore(
        persist_directory=settings.chroma_directory,
    )


@lru_cache
def get_bm25_retriever() -> BM25Retriever:
    return BM25Retriever()

@lru_cache
def get_postgres_fts_retriever() -> PostgresFTSRetriever:
    return PostgresFTSRetriever(
        session_factory=SessionLocal,
    )

@lru_cache
def get_lexical_retriever() -> LexicalIndex:
    if settings.lexical_backend == "bm25":
        return get_bm25_retriever()

    if settings.lexical_backend == "postgres_fts":
        return get_postgres_fts_retriever()

    raise ValueError(
        f"Unsupported lexical backend: "
        f"{settings.lexical_backend}"
    )

@lru_cache
def get_retrieval_index_manager() -> RetrievalIndexManager:
    manager = RetrievalIndexManager(
        lexical_index=get_lexical_retriever(),
    )

    chunks = get_vector_store().get_all_chunks()

    if chunks:
        manager.rebuild(chunks)

    return manager


@lru_cache
def get_document_store() -> DocumentStore:
    return DocumentStore(
        session_factory=SessionLocal,
    )


@lru_cache
def get_file_storage() -> UploadedFileStorage:
    return UploadedFileStorage(
        storage_directory=settings.upload_directory,
    )


@lru_cache
def get_ingestion_service() -> DocumentIngestionService:
    return DocumentIngestionService(
        document_processor=get_document_processor(),
        chunker=get_chunker(),
        embedding_service=get_embedding_service(),
        vector_store=get_vector_store(),
        retrieval_index_manager=get_retrieval_index_manager(),
        document_store=get_document_store(),
    )


@lru_cache
def get_dense_retriever():
    from app.retrieval import DenseRetriever

    return DenseRetriever(
        embedding_service=get_embedding_service(),
        vector_store=get_vector_store(),
    )


@lru_cache
def get_hybrid_retriever() -> HybridRetriever:
    return HybridRetriever(
        dense_retriever=get_dense_retriever(),
        lexical_retriever=get_lexical_retriever(),

    )


@lru_cache
def get_reranker() -> CrossEncoderReranker:
    return CrossEncoderReranker(
        config=settings,
    )


@lru_cache
def get_compressor() -> ContextCompressor:
    return ContextCompressor()


@lru_cache
def get_generator() -> GeminiGenerator:
    return GeminiGenerator(
        config=settings,
    )


@lru_cache
def get_query_service() -> QueryService:
    return QueryService(
        hybrid_retriever=get_hybrid_retriever(),
        reranker=get_reranker(),
        compressor=get_compressor(),
        generator=get_generator(),
        config=settings,
    )

@lru_cache
def get_ingestion_job_store() -> IngestionJobStore:
    return IngestionJobStore(
        session_factory=SessionLocal,
    )


@lru_cache
def get_background_ingestion_service() -> BackgroundIngestionService:
    return BackgroundIngestionService(
        ingestion_service=get_ingestion_service(),
        job_store=get_ingestion_job_store(),
    )

def clear_dependency_caches() -> None:
    """
    Clear all cached application dependency instances.

    This is primarily useful during application shutdown and tests.
    """

    get_query_service.cache_clear()
    get_generator.cache_clear()
    get_compressor.cache_clear()
    get_reranker.cache_clear()
    get_hybrid_retriever.cache_clear()
    get_dense_retriever.cache_clear()

    get_lexical_retriever.cache_clear()
    get_postgres_fts_retriever.cache_clear()

    get_retrieval_index_manager.cache_clear()
    get_bm25_retriever.cache_clear()
    get_embedding_service.cache_clear()
    get_vector_store.cache_clear()

    get_background_ingestion_service.cache_clear()
    get_ingestion_job_store.cache_clear()

    get_ingestion_service.cache_clear()
    get_file_storage.cache_clear()
    get_document_store.cache_clear()
    get_chunker.cache_clear()
    get_document_processor.cache_clear()
