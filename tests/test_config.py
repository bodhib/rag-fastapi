from pathlib import Path

from app.config import Settings

import pytest


def test_settings_use_default_values():
    settings = Settings()

    assert settings.app_name == "Production RAG API"

    assert settings.app_version == "1.0.0"

    assert (
        settings.gemini_model
        == "gemini-2.5-flash-lite"
    )

    assert (
        settings.embedding_model
        == "sentence-transformers/all-MiniLM-L6-v2"
    )

    assert (
        settings.reranker_model
        == "cross-encoder/ms-marco-MiniLM-L-6-v2"
    )

    assert settings.chunk_size == 1000

    assert settings.chunk_overlap == 200

    assert settings.retrieval_top_k == 10

    assert settings.rerank_top_k == 5

    assert settings.context_top_k == 3

    assert settings.base_data_directory == Path(
        "data"
    )

def test_settings_load_environment_variables(
    monkeypatch,
):
    monkeypatch.setenv(
        "GEMINI_API_KEY",
        "test-key",
    )

    monkeypatch.setenv(
        "GEMINI_MODEL",
        "test-model",
    )

    monkeypatch.setenv(
        "CHUNK_SIZE",
        "800",
    )

    settings = Settings()

    assert settings.gemini_api_key == "test-key"

    assert settings.gemini_model == "test-model"

    assert settings.chunk_size == 800

def test_settings_load_dotenv_file(
    tmp_path,
):
    env_file = tmp_path / ".env"

    env_file.write_text(
        "GEMINI_API_KEY=test-dotenv-key\n"
        "GEMINI_MODEL=test-dotenv-model\n"
        "CHUNK_SIZE=750\n",
        encoding="utf-8",
    )

    settings = Settings(
        _env_file=env_file
    )

    assert settings.gemini_api_key == (
        "test-dotenv-key"
    )

    assert settings.gemini_model == (
        "test-dotenv-model"
    )

    assert settings.chunk_size == 750

def test_settings_rejects_zero_chunk_size():
    with pytest.raises(ValueError):
        Settings(chunk_size=0)


def test_settings_rejects_negative_chunk_size():
    with pytest.raises(ValueError):
        Settings(chunk_size=-100)


def test_settings_allows_zero_chunk_overlap():
    settings = Settings(chunk_overlap=0)

    assert settings.chunk_overlap == 0


def test_settings_rejects_negative_chunk_overlap():
    with pytest.raises(ValueError):
        Settings(chunk_overlap=-1)


def test_settings_rejects_zero_retrieval_top_k():
    with pytest.raises(ValueError):
        Settings(retrieval_top_k=0)


def test_settings_rejects_negative_rerank_top_k():
    with pytest.raises(ValueError):
        Settings(rerank_top_k=-1)


def test_settings_rejects_zero_context_top_k():
    with pytest.raises(ValueError):
        Settings(context_top_k=0)


def test_settings_rejects_overlap_equal_to_chunk_size():
    with pytest.raises(
        ValueError,
        match="chunk_overlap must be smaller than chunk_size",
    ):
        Settings(
            chunk_size=500,
            chunk_overlap=500,
        )


def test_settings_rejects_overlap_greater_than_chunk_size():
    with pytest.raises(
        ValueError,
        match="chunk_overlap must be smaller than chunk_size",
    ):
        Settings(
            chunk_size=500,
            chunk_overlap=600,
        )


def test_settings_accepts_valid_chunk_configuration():
    settings = Settings(
        chunk_size=500,
        chunk_overlap=100,
    )

    assert settings.chunk_size == 500
    assert settings.chunk_overlap == 100    

def test_settings_is_immutable():
    config = Settings(
        chunk_size=500,
        chunk_overlap=100,
    )

    with pytest.raises((TypeError, ValueError)):
        config.chunk_size = 800

    assert config.chunk_size == 500  

def test_settings_exposes_expected_runtime_configuration():
    from app.config import Settings

    config = Settings(
        gemini_api_key="test-key",
        gemini_model="test-model",
        embedding_model="test-embedding",
        reranker_model="test-reranker",
        chunk_size=800,
        chunk_overlap=100,
        retrieval_top_k=12,
        rerank_top_k=6,
        context_top_k=3,
    )

    assert config.gemini_api_key == "test-key"
    assert config.gemini_model == "test-model"

    assert config.embedding_model == "test-embedding"
    assert config.reranker_model == "test-reranker"

    assert config.chunk_size == 800
    assert config.chunk_overlap == 100

    assert config.retrieval_top_k == 12
    assert config.rerank_top_k == 6
    assert config.context_top_k == 3

def test_settings_rejects_invalid_chunk_relationship():
    import pytest

    from app.config import Settings

    with pytest.raises(
        ValueError,
        match="chunk_overlap must be smaller than chunk_size",
    ):
        Settings(
            chunk_size=100,
            chunk_overlap=100,
        )        