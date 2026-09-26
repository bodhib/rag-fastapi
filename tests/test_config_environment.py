from pathlib import Path

import pytest

from app.config import (
    Settings,
    get_settings,
)


def test_settings_defaults_to_development(
    monkeypatch,
) -> None:
    monkeypatch.delenv(
        "APP_ENV",
        raising=False,
    )

    settings = Settings(
        _env_file=None,
    )

    assert settings.app_env == "development"
    assert settings.is_development is True
    assert settings.is_test is False
    assert settings.is_production is False


def test_environment_specific_file_overrides_base_file(
    tmp_path,
    monkeypatch,
) -> None:
    base_env = tmp_path / ".env"
    test_env = tmp_path / ".env.test"

    base_env.write_text(
        "APP_NAME=Base App\n"
        "RETRIEVAL_TOP_K=10\n",
        encoding="utf-8",
    )

    test_env.write_text(
        "APP_NAME=Test App\n"
        "RETRIEVAL_TOP_K=25\n",
        encoding="utf-8",
    )

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv(
        "APP_ENV",
        "test",
    )

    settings = get_settings()

    assert settings.app_name == "Test App"
    assert settings.retrieval_top_k == 25
    assert settings.app_env == "test"
    assert settings.is_test is True


def test_process_environment_overrides_dotenv_files(
    tmp_path,
    monkeypatch,
) -> None:
    base_env = tmp_path / ".env"
    production_env = tmp_path / ".env.production"

    base_env.write_text(
        "APP_NAME=Base App\n",
        encoding="utf-8",
    )

    production_env.write_text(
        "APP_NAME=Production File App\n",
        encoding="utf-8",
    )

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv(
        "APP_ENV",
        "production",
    )

    monkeypatch.setenv(
    "RAG_USER_API_KEY",
    "test-user-key-123456789012345678901234567890",
    )
    monkeypatch.setenv(
        "RAG_ADMIN_API_KEY",
        "test-admin-key-123456789012345678901234567890",
    )
    monkeypatch.setenv(
        "GEMINI_API_KEY",
        "test-gemini-key-123456789012345678901234567890",
    )

    monkeypatch.setenv("AUTH_ENABLED", "true")
    monkeypatch.setenv(
        "APP_NAME",
        "Process Environment App",
    )

    settings = get_settings()

    assert (
        settings.app_name
        == "Process Environment App"
    )
    assert settings.app_env == "production"


def test_invalid_environment_is_rejected(
    monkeypatch,
) -> None:
    monkeypatch.setenv(
        "APP_ENV",
        "staging",
    )

    with pytest.raises(
        ValueError,
        match=(
            "APP_ENV must be one of: "
            "development, test, production"
        ),
    ):
        get_settings()


def test_environment_properties() -> None:
    development = Settings(
        app_env="development",
        _env_file=None,
    )

    test = Settings(
        app_env="test",
        _env_file=None,
    )

    production = Settings(
        app_env="production",
        auth_enabled=True,
        rag_user_api_key="test-user-key-123456789012345678901234567890",
        rag_admin_api_key="test-admin-key-123456789012345678901234567890",
        gemini_api_key="test-gemini-key-123456789012345678901234567890",
        _env_file=None,
    )

    assert development.is_development is True
    assert development.is_test is False
    assert development.is_production is False

    assert test.is_development is False
    assert test.is_test is True
    assert test.is_production is False

    assert production.is_development is False
    assert production.is_test is False
    assert production.is_production is True


def test_chunk_configuration_remains_validated() -> None:
    with pytest.raises(
        ValueError,
        match=(
            "chunk_overlap must be smaller than "
            "chunk_size"
        ),
    ):
        Settings(
            chunk_size=100,
            chunk_overlap=100,
            _env_file=None,
        )