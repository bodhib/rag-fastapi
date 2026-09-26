"""
Authentication and authorization for the RAG API.

Authentication:
    API key supplied through a configurable HTTP header.

Authorization:
    Role-based access control using USER and ADMIN roles.
"""

from dataclasses import dataclass
from enum import StrEnum
import secrets
from typing import Annotated, Callable

from fastapi import Depends, HTTPException, Security, status
from fastapi.security import APIKeyHeader

from app.config import Settings, settings


class UserRole(StrEnum):
    """Roles supported by the application."""

    USER = "user"
    ADMIN = "admin"


@dataclass(frozen=True)
class Principal:
    """
    Represents the authenticated caller.

    This abstraction deliberately hides the underlying
    authentication mechanism from the rest of the application.

    The API currently uses API keys, but the application can
    later move to JWT/OAuth without changing service-layer code.
    """

    name: str
    role: UserRole


class APIKeyAuthenticator:
    """
    Authenticate callers using configured API keys.

    Two keys are supported:

        RAG_USER_API_KEY
        RAG_ADMIN_API_KEY

    The comparison uses secrets.compare_digest() so that
    credential comparison is performed in constant time.
    """

    def __init__(
        self,
        config: Settings,
    ) -> None:
        self.config = config

    @property
    def enabled(self) -> bool:
        """Return whether authentication is enabled."""
        return self.config.auth_enabled

    @property
    def configured(self) -> bool:
        """
        Return whether at least one authentication credential
        has been configured.
        """

        return bool(
            self.config.rag_user_api_key.strip()
            or self.config.rag_admin_api_key.strip()
        )

    def authenticate(
        self,
        api_key: str | None,
    ) -> Principal:
        """
        Authenticate an API key and return its principal.

        Authentication disabled:
            Any caller is treated as an anonymous development
            principal.

        Authentication enabled:
            Missing/invalid credentials result in HTTP 401.

        Authentication enabled but no keys configured:
            HTTP 503 because the application security configuration
            is incomplete.
        """

        if not self.enabled:
            return Principal(
                name="anonymous",
                role=UserRole.ADMIN,
            )

        if not self.configured:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Authentication is not configured",
            )

        if api_key is None or not api_key.strip():
            raise self._authentication_error()

        normalized_api_key = api_key.strip()

        if (
            self.config.rag_admin_api_key
            and secrets.compare_digest(
                normalized_api_key,
                self.config.rag_admin_api_key,
            )
        ):
            return Principal(
                name="admin",
                role=UserRole.ADMIN,
            )

        if (
            self.config.rag_user_api_key
            and secrets.compare_digest(
                normalized_api_key,
                self.config.rag_user_api_key,
            )
        ):
            return Principal(
                name="user",
                role=UserRole.USER,
            )

        raise self._authentication_error()

    def authorize(
        self,
        principal: Principal,
        required_role: UserRole,
    ) -> Principal:
        """
        Verify that the authenticated principal has the
        required role.

        ADMIN has all USER permissions.
        """

        if not self.enabled:
            return principal

        if (
            required_role == UserRole.USER
            and principal.role in {
                UserRole.USER,
                UserRole.ADMIN,
            }
        ):
            return principal

        if (
            required_role == UserRole.ADMIN
            and principal.role == UserRole.ADMIN
        ):
            return principal

        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Insufficient permissions",
        )

    @staticmethod
    def _authentication_error() -> HTTPException:
        """
        Build a consistent 401 response.

        API-key authentication is not standardized through
        WWW-Authenticate in the same way as Basic/Bearer schemes,
        so we explicitly identify the challenge as APIKey.
        """

        return HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
            headers={
                "WWW-Authenticate": "APIKey",
            },
        )


api_key_header = APIKeyHeader(
    name=settings.api_key_header_name,
    scheme_name="APIKeyAuth",
    description=(
        "API key used to authenticate requests."
    ),
    auto_error=False,
)


authenticator = APIKeyAuthenticator(
    config=settings,
)


def get_current_principal(
    api_key: Annotated[
        str | None,
        Security(api_key_header),
    ],
) -> Principal:
    """
    FastAPI dependency that authenticates the current request.
    """

    return authenticator.authenticate(api_key)


def require_role(
    required_role: UserRole,
) -> Callable[..., Principal]:
    """
    Create a FastAPI dependency that requires a specific role.

    Example:

        @app.get(
            "/query",
            dependencies=[
                Depends(require_role(UserRole.USER))
            ],
        )
    """

    def dependency(
        principal: Annotated[
            Principal,
            Depends(get_current_principal),
        ],
    ) -> Principal:
        return authenticator.authorize(
            principal=principal,
            required_role=required_role,
        )

    return dependency