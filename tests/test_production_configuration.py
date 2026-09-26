import pytest

from app.config import Settings


def create_settings(**overrides) -> Settings:
    values = {
        "app_env": "development",
        "auth_enabled": False,
        "rag_user_api_key": "",
        "rag_admin_api_key": "",
        "gemini_api_key": "",
        "database_url": (
            "sqlite:///./data/rag.db"
        ),
        "chunk_size": 1000,
        "chunk_overlap": 200,
    }

    values.update(overrides)

    return Settings(
        _env_file=None,
        **values,
    )


def test_development_can_have_authentication_disabled() -> None:
    settings = create_settings(
        app_env="development",
        auth_enabled=False,
    )

    assert settings.is_development is True
    assert settings.auth_enabled is False


def test_development_can_enable_authentication() -> None:
    settings = create_settings(
        app_env="development",
        auth_enabled=True,
        rag_user_api_key="user-key",
        rag_admin_api_key="admin-key",
    )

    assert settings.auth_enabled is True


def test_production_requires_authentication() -> None:
    with pytest.raises(
        ValueError,
        match="Authentication must be enabled",
    ):
        create_settings(
            app_env="production",
            auth_enabled=False,
            rag_user_api_key="a" * 32,
            rag_admin_api_key="b" * 32,
            gemini_api_key="gemini-key",
        )


def test_production_requires_user_api_key() -> None:
    with pytest.raises(
        ValueError,
        match="RAG_USER_API_KEY is required",
    ):
        create_settings(
            app_env="production",
            auth_enabled=True,
            rag_user_api_key="",
            rag_admin_api_key="b" * 32,
            gemini_api_key="gemini-key",
        )


def test_production_requires_admin_api_key() -> None:
    with pytest.raises(
        ValueError,
        match="RAG_ADMIN_API_KEY is required",
    ):
        create_settings(
            app_env="production",
            auth_enabled=True,
            rag_user_api_key="a" * 32,
            rag_admin_api_key="",
            gemini_api_key="gemini-key",
        )


def test_production_requires_gemini_api_key() -> None:
    with pytest.raises(
        ValueError,
        match="GEMINI_API_KEY is required",
    ):
        create_settings(
            app_env="production",
            auth_enabled=True,
            rag_user_api_key="a" * 32,
            rag_admin_api_key="b" * 32,
            gemini_api_key="",
        )


def test_production_requires_long_user_api_key() -> None:
    with pytest.raises(
        ValueError,
        match=(
            "RAG_USER_API_KEY must contain "
            "at least 32 characters"
        ),
    ):
        create_settings(
            app_env="production",
            auth_enabled=True,
            rag_user_api_key="short",
            rag_admin_api_key="b" * 32,
            gemini_api_key="gemini-key",
        )


def test_production_requires_long_admin_api_key() -> None:
    with pytest.raises(
        ValueError,
        match=(
            "RAG_ADMIN_API_KEY must contain "
            "at least 32 characters"
        ),
    ):
        create_settings(
            app_env="production",
            auth_enabled=True,
            rag_user_api_key="a" * 32,
            rag_admin_api_key="short",
            gemini_api_key="gemini-key",
        )


def test_production_requires_different_api_keys() -> None:
    same_key = "a" * 32

    with pytest.raises(
        ValueError,
        match=(
            "RAG_USER_API_KEY and "
            "RAG_ADMIN_API_KEY must be "
            "different"
        ),
    ):
        create_settings(
            app_env="production",
            auth_enabled=True,
            rag_user_api_key=same_key,
            rag_admin_api_key=same_key,
            gemini_api_key="gemini-key",
        )


def test_production_accepts_valid_configuration() -> None:
    settings = create_settings(
        app_env="production",
        auth_enabled=True,
        rag_user_api_key="a" * 32,
        rag_admin_api_key="b" * 32,
        gemini_api_key="gemini-key",
        database_url=(
            "postgresql+psycopg://"
            "rag_user:password@localhost:5432/rag_db"
        ),
    )

    assert settings.is_production is True
    assert settings.auth_enabled is True


def test_empty_api_key_header_name_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="api_key_header_name cannot be empty",
    ):
        create_settings(
            api_key_header_name=" ",
        )