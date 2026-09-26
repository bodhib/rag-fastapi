"""
Application configuration.

Configuration precedence:

    1. built-in defaults
    2. .env
    3. .env.<APP_ENV>
    4. real operating-system environment variables

Environment variables have the highest precedence.
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
    """
    Determine the active application environment.

    APP_ENV is intentionally read directly from the process
    environment because we need it before selecting the
    environment-specific dotenv file.
    """

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
    """
    Return dotenv files in increasing precedence order.

    The environment-specific file is loaded after .env so it
    can override base configuration.
    """

    return (
        ".env",
        f".env.{environment}",
    )


class Settings(BaseSettings):
    """
    Application configuration.

    Secrets should normally come from process environment
    variables rather than committed files.
    """

    # ---------------------------------------------------------
    # Application
    # ---------------------------------------------------------

    app_name: str = "Production RAG API"

    app_version: str = "1.0.0"

    app_env: AppEnvironment = "development"

    # ---------------------------------------------------------
    # Authentication
    # ---------------------------------------------------------

    auth_enabled: bool = False

    api_key_header_name: str = "X-API-Key"

    rag_user_api_key: str = ""

    rag_admin_api_key: str = ""

    # ---------------------------------------------------------
    # LLM / Models
    # ---------------------------------------------------------

    gemini_api_key: str = ""

    gemini_model: str = (
        "gemini-2.5-flash-lite"
    )

    embedding_model: str = (
        "sentence-transformers/"
        "all-MiniLM-L6-v2"
    )

    reranker_model: str = (
        "cross-encoder/"
        "ms-marco-MiniLM-L-6-v2"
    )

    # ---------------------------------------------------------
    # Storage
    # ---------------------------------------------------------

    base_data_directory: Path = Path("data")

    upload_directory: Path = Path(
        "data/uploads"
    )

    chroma_directory: Path = Path(
        "data/chroma"
    )

    document_store_path: Path = Path(
        "data/documents.json"
    )

    # ---------------------------------------------------------
    # Database / lexical retrieval
    # ---------------------------------------------------------

    lexical_backend: Literal[
        "bm25",
        "postgres_fts",
    ] = "bm25"

    database_url: str = (
        "sqlite:///./data/rag.db"
    )

    # ---------------------------------------------------------
    # Chunking
    # ---------------------------------------------------------

    chunk_size: int = Field(
        default=1000,
        gt=0,
    )

    chunk_overlap: int = Field(
        default=200,
        ge=0,
    )

    # ---------------------------------------------------------
    # Retrieval
    # ---------------------------------------------------------

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

    # ---------------------------------------------------------
    # Pydantic settings
    # ---------------------------------------------------------

    model_config = SettingsConfigDict(
        env_file=None,
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        frozen=True,
    )

    # ---------------------------------------------------------
    # Cross-field validation
    # ---------------------------------------------------------

    @model_validator(mode="after")
    def validate_configuration(
        self,
    ) -> "Settings":
        """
        Validate relationships between settings.

        Production deliberately fails closed.
        """

        # -----------------------------------------------------
        # Chunk configuration
        # -----------------------------------------------------

        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                "chunk_overlap must be smaller "
                "than chunk_size"
            )

        # -----------------------------------------------------
        # Authentication header
        # -----------------------------------------------------

        if not self.api_key_header_name.strip():
            raise ValueError(
                "api_key_header_name cannot be empty"
            )

        # -----------------------------------------------------
        # Production security requirements
        # -----------------------------------------------------

        if self.app_env == "production":
            if not self.auth_enabled:
                raise ValueError(
                    "Authentication must be enabled "
                    "in production"
                )

            if not self.rag_user_api_key.strip():
                raise ValueError(
                    "RAG_USER_API_KEY is required "
                    "in production"
                )

            if not self.rag_admin_api_key.strip():
                raise ValueError(
                    "RAG_ADMIN_API_KEY is required "
                    "in production"
                )

            if (
                len(self.rag_user_api_key.strip())
                < 32
            ):
                raise ValueError(
                    "RAG_USER_API_KEY must contain "
                    "at least 32 characters in "
                    "production"
                )

            if (
                len(self.rag_admin_api_key.strip())
                < 32
            ):
                raise ValueError(
                    "RAG_ADMIN_API_KEY must contain "
                    "at least 32 characters in "
                    "production"
                )

            if (
                self.rag_user_api_key.strip()
                == self.rag_admin_api_key.strip()
            ):
                raise ValueError(
                    "RAG_USER_API_KEY and "
                    "RAG_ADMIN_API_KEY must be "
                    "different in production"
                )

            if not self.gemini_api_key.strip():
                raise ValueError(
                    "GEMINI_API_KEY is required "
                    "in production"
                )

            if not self.database_url.strip():
                raise ValueError(
                    "DATABASE_URL is required "
                    "in production"
                )

        return self

    # ---------------------------------------------------------
    # Environment helpers
    # ---------------------------------------------------------

    @property
    def is_development(self) -> bool:
        return self.app_env == "development"

    @property
    def is_test(self) -> bool:
        return self.app_env == "test"

    @property
    def is_production(self) -> bool:
        return self.app_env == "production"


def get_settings() -> Settings:
    """
    Construct settings using the active environment.
    """

    environment = _get_app_environment()

    environment_files = _get_environment_files(
        environment
    )

    return Settings(
        _env_file=environment_files,
        _env_file_encoding="utf-8",
    )


settings = get_settings()