"""
Application configuration.

Configuration precedence for application startup:

    1. built-in defaults
    2. .env
    3. .env.<APP_ENV>
    4. real operating-system environment variables

Environment variables have the highest precedence.

Important:

    Direct Settings() construction does not use APP_ENV from
    the process environment to determine app_env.

    get_settings() explicitly determines the application
    environment and passes it to Settings.

This keeps unit tests isolated from the environment in which
pytest happens to be running, while preserving strict
production validation for the real application configuration.
"""

import os
from pathlib import Path
from typing import Literal

from pydantic import (
    Field,
    model_validator,
)
from pydantic_settings import (
    BaseSettings,
    SettingsConfigDict,
)


AppEnvironment = Literal[
    "development",
    "test",
    "production",
]

SUPPORTED_ENVIRONMENTS = {
    "development",
    "test",
    "production",
}


def _get_app_environment() -> AppEnvironment:
    raw_environment = os.getenv(
        "APP_ENV",
        "development",
    ).strip().lower()

    if raw_environment not in SUPPORTED_ENVIRONMENTS:
        raise ValueError(
            "APP_ENV must be one of: "
            "development, test, production"
        )

    return raw_environment  # type: ignore[return-value]


def _get_environment_files(
    environment: AppEnvironment,
) -> tuple[str, ...]:
    return (
        ".env",
        f".env.{environment}",
    )


class Settings(BaseSettings):
    app_name: str = "Production RAG API"
    app_version: str = "1.0.0"

    # Important:
    #
    # app_env intentionally has a normal Python default here.
    # Direct Settings() construction should not inherit APP_ENV
    # from the process environment.
    #
    # get_settings() explicitly passes the environment when
    # creating the application's real configuration.
    app_env: AppEnvironment = "development"

    auth_enabled: bool = False
    api_key_header_name: str = "X-API-Key"
    rag_user_api_key: str = ""
    rag_admin_api_key: str = ""

    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.5-flash-lite"
    embedding_model: str = (
        "sentence-transformers/all-MiniLM-L6-v2"
    )
    reranker_model: str = (
        "cross-encoder/ms-marco-MiniLM-L-6-v2"
    )

    base_data_directory: Path = Path("data")
    upload_directory: Path = Path("data/uploads")
    chroma_directory: Path = Path("data/chroma")
    document_store_path: Path = Path("data/documents.json")

    lexical_backend: Literal[
        "bm25",
        "postgres_fts",
    ] = "bm25"

    database_url: str = "sqlite:///./data/rag.db"

    chunk_size: int = Field(
        default=1000,
        gt=0,
    )

    chunk_overlap: int = Field(
        default=200,
        ge=0,
    )

    retrieval_top_k: int = Field(
        default=10,
        gt=0,
    )

    rerank_top_k: int = Field(
        default=5,
        gt=0,
    )

    context_top_k: int = Field(
        default=3,
        gt=0,
    )

    model_config = SettingsConfigDict(
        env_file=None,
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        frozen=True,
    )

    @model_validator(mode="after")
    def validate_configuration(self) -> "Settings":
        """
        Validate configuration rules that are applicable
        regardless of deployment environment.
        """

        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                "chunk_overlap must be smaller "
                "than chunk_size"
            )

        if not self.api_key_header_name.strip():
            raise ValueError(
                "api_key_header_name cannot be empty"
            )

        if self.app_env == "production":
            _validate_production_settings(self)

        return self

    @property
    def is_development(self) -> bool:
        return self.app_env == "development"

    @property
    def is_test(self) -> bool:
        return self.app_env == "test"

    @property
    def is_production(self) -> bool:
        return self.app_env == "production"


def _validate_production_settings(
    configuration: Settings,
) -> None:
    """
    Validate requirements that must be satisfied before the
    application is allowed to run in production.
    """

    if not configuration.auth_enabled:
        raise ValueError(
            "Authentication must be enabled "
            "in production"
        )

    if not configuration.rag_user_api_key.strip():
        raise ValueError(
            "RAG_USER_API_KEY is required "
            "in production"
        )

    if not configuration.rag_admin_api_key.strip():
        raise ValueError(
            "RAG_ADMIN_API_KEY is required "
            "in production"
        )

    if len(configuration.rag_user_api_key.strip()) < 32:
        raise ValueError(
            "RAG_USER_API_KEY must contain "
            "at least 32 characters in "
            "production"
        )

    if len(configuration.rag_admin_api_key.strip()) < 32:
        raise ValueError(
            "RAG_ADMIN_API_KEY must contain "
            "at least 32 characters in "
            "production"
        )

    if (
        configuration.rag_user_api_key.strip()
        == configuration.rag_admin_api_key.strip()
    ):
        raise ValueError(
            "RAG_USER_API_KEY and "
            "RAG_ADMIN_API_KEY must be "
            "different in production"
        )

    if not configuration.gemini_api_key.strip():
        raise ValueError(
            "GEMINI_API_KEY is required "
            "in production"
        )

    if not configuration.database_url.strip():
        raise ValueError(
            "DATABASE_URL is required "
            "in production"
        )


def get_settings() -> Settings:
    """
    Build the application's real environment-aware settings.

    APP_ENV is read explicitly from the process environment
    here rather than allowing every direct Settings() instance
    to inherit it.
    """

    environment = _get_app_environment()
    environment_files = _get_environment_files(
        environment
    )

    return Settings(
        app_env=environment,
        _env_file=environment_files,
        _env_file_encoding="utf-8",
    )


settings = get_settings()

