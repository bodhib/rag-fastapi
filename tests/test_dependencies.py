from pathlib import Path

from app.config import Settings
from app.dependencies import (
    get_chunker,
    get_document_store,
    get_file_storage,
    get_vector_store,
)

from app.database import SessionLocal

from app.bm25_retriever import (
    BM25Retriever,
    LexicalIndex,
)
from app.dependencies import (
    clear_dependency_caches,
    get_bm25_retriever,
    get_lexical_retriever,
    get_postgres_fts_retriever,
    get_retrieval_index_manager,
)
from app.vector_store import VectorStore

from app.postgres_fts_retriever import PostgresFTSRetriever


def test_get_chunker_uses_settings(monkeypatch):
    import app.dependencies as dependencies

    test_settings = Settings(
        chunk_size=750,
        chunk_overlap=100,
    )

    monkeypatch.setattr(
        dependencies,
        "settings",
        test_settings,
    )

    get_chunker.cache_clear()

    chunker = get_chunker()

    assert chunker.chunk_size == 750
    assert chunker.chunk_overlap == 100

    get_chunker.cache_clear()


def test_get_vector_store_uses_settings(
    tmp_path,
    monkeypatch,
):
    import app.dependencies as dependencies

    chroma_directory = tmp_path / "custom-chroma"

    test_settings = Settings(
        chroma_directory=chroma_directory,
    )

    monkeypatch.setattr(
        dependencies,
        "settings",
        test_settings,
    )

    get_vector_store.cache_clear()

    vector_store = get_vector_store()

    assert vector_store.persist_directory == chroma_directory

    get_vector_store.cache_clear()


def test_get_document_store_uses_settings(
    tmp_path,
    monkeypatch,
):
    import app.dependencies as dependencies

    document_store_path = tmp_path / "custom-documents.json"

    test_settings = Settings(
        document_store_path=document_store_path,
    )

    monkeypatch.setattr(
        dependencies,
        "settings",
        test_settings,
    )

    get_document_store.cache_clear()

    document_store = get_document_store()

    assert document_store.session_factory is SessionLocal

    get_document_store.cache_clear()


def test_get_file_storage_uses_settings(
    tmp_path,
    monkeypatch,
):
    import app.dependencies as dependencies

    upload_directory = tmp_path / "custom-uploads"

    test_settings = Settings(
        upload_directory=upload_directory,
    )

    monkeypatch.setattr(
        dependencies,
        "settings",
        test_settings,
    )

    get_file_storage.cache_clear()

    file_storage = get_file_storage()

    assert file_storage.storage_directory == upload_directory

    get_file_storage.cache_clear()


def test_get_query_service_is_cached():
    import app.dependencies as dependencies

    get_query_service = dependencies.get_query_service

    get_query_service.cache_clear()

    first = get_query_service()
    second = get_query_service()

    assert first is second

    get_query_service.cache_clear()


def test_clear_dependency_caches():
    import app.dependencies as dependencies

    dependencies.get_chunker.cache_clear()

    first = dependencies.get_chunker()

    dependencies.clear_dependency_caches()

    second = dependencies.get_chunker()

    assert first is not second

    dependencies.clear_dependency_caches()


def test_get_generator_uses_configured_settings(
    monkeypatch,
):
    import app.dependencies as dependencies

    from app.config import Settings

    test_settings = Settings(
        gemini_api_key="test-key",
        gemini_model="test-model",
    )

    dependencies.get_generator.cache_clear()

    monkeypatch.setattr(
        dependencies,
        "settings",
        test_settings,
    )

    generator = dependencies.get_generator()

    assert generator.config is test_settings
    assert generator.model_name == "test-model"

    dependencies.get_generator.cache_clear()


def test_get_embedding_service_uses_configured_model(
    monkeypatch,
):
    import app.dependencies as dependencies

    from app.config import Settings

    test_settings = Settings(
        embedding_model="test-embedding-model",
    )

    dependencies.get_embedding_service.cache_clear()

    monkeypatch.setattr(
        dependencies,
        "settings",
        test_settings,
    )

    class FakeEmbeddingService:
        def __init__(self, config=None):
            self.config = config

    monkeypatch.setattr(
        dependencies,
        "EmbeddingService",
        FakeEmbeddingService,
    )

    service = dependencies.get_embedding_service()

    assert service.config is test_settings

    dependencies.get_embedding_service.cache_clear()


def test_get_reranker_uses_configured_settings(
    monkeypatch,
):
    import app.dependencies as dependencies

    from app.config import Settings

    test_settings = Settings(
        reranker_model="test-reranker-model",
    )

    dependencies.get_reranker.cache_clear()

    monkeypatch.setattr(
        dependencies,
        "settings",
        test_settings,
    )

    class FakeReranker:
        def __init__(self, config=None):
            self.config = config

    monkeypatch.setattr(
        dependencies,
        "CrossEncoderReranker",
        FakeReranker,
    )

    reranker = dependencies.get_reranker()

    assert reranker.config is test_settings

    dependencies.get_reranker.cache_clear()


def test_retrieval_index_manager_uses_abstractions(
    monkeypatch,
) -> None:
    import app.dependencies as dependencies

    test_settings = Settings(
        lexical_backend="bm25",
    )

    monkeypatch.setattr(
        dependencies,
        "settings",
        test_settings,
    )

    clear_dependency_caches()

    manager = get_retrieval_index_manager()

    assert isinstance(
        manager.lexical_index,
        LexicalIndex,
    )

    assert isinstance(
        manager.lexical_index,
        BM25Retriever,
    )

    clear_dependency_caches()
    
def test_get_lexical_retriever_selects_bm25(
    monkeypatch,
) -> None:
    import app.dependencies as dependencies

    test_settings = Settings(
        lexical_backend="bm25",
    )

    monkeypatch.setattr(
        dependencies,
        "settings",
        test_settings,
    )

    dependencies.clear_dependency_caches()

    retriever = dependencies.get_lexical_retriever()

    assert isinstance(
        retriever,
        BM25Retriever,
    )

    dependencies.clear_dependency_caches()


def test_get_lexical_retriever_selects_postgres_fts(
    monkeypatch,
) -> None:
    import app.dependencies as dependencies

    test_settings = Settings(
        lexical_backend="postgres_fts",
    )

    monkeypatch.setattr(
        dependencies,
        "settings",
        test_settings,
    )

    dependencies.clear_dependency_caches()

    retriever = dependencies.get_lexical_retriever()

    assert isinstance(
        retriever,
        PostgresFTSRetriever,
    )

    dependencies.clear_dependency_caches()


def test_lexical_backend_defaults_to_bm25(
    monkeypatch,
) -> None:
    monkeypatch.delenv(
        "LEXICAL_BACKEND",
        raising=False,
    )

    config = Settings(_env_file=None)

    assert config.lexical_backend == "bm25"


def test_settings_default_lexical_backend(
    monkeypatch,
) -> None:
    monkeypatch.delenv(
        "LEXICAL_BACKEND",
        raising=False,
    )

    assert Settings(
        _env_file=None
    ).lexical_backend == "bm25"