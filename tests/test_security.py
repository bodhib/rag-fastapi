import pytest
from fastapi import HTTPException

from app.config import Settings
from app.security import (
    APIKeyAuthenticator,
    Principal,
    UserRole,
)


def create_settings(
    *,
    auth_enabled: bool = True,
    user_api_key: str = "user-secret",
    admin_api_key: str = "admin-secret",
) -> Settings:
    return Settings(
        app_env="test",
        auth_enabled=auth_enabled,
        api_key_header_name="X-API-Key",
        rag_user_api_key=user_api_key,
        rag_admin_api_key=admin_api_key,
        _env_file=None,
    )


def test_authentication_disabled_returns_anonymous_principal() -> None:
    config = create_settings(
        auth_enabled=False,
    )

    authenticator = APIKeyAuthenticator(
        config=config,
    )

    principal = authenticator.authenticate(
        api_key=None,
    )

    assert principal == Principal(
        name="anonymous",
        role=UserRole.ADMIN,
    )


def test_user_api_key_authenticates_as_user() -> None:
    config = create_settings()

    authenticator = APIKeyAuthenticator(
        config=config,
    )

    principal = authenticator.authenticate(
        api_key="user-secret",
    )

    assert principal.name == "user"
    assert principal.role == UserRole.USER


def test_admin_api_key_authenticates_as_admin() -> None:
    config = create_settings()

    authenticator = APIKeyAuthenticator(
        config=config,
    )

    principal = authenticator.authenticate(
        api_key="admin-secret",
    )

    assert principal.name == "admin"
    assert principal.role == UserRole.ADMIN


def test_missing_api_key_returns_401() -> None:
    config = create_settings()

    authenticator = APIKeyAuthenticator(
        config=config,
    )

    with pytest.raises(
        HTTPException,
    ) as error:
        authenticator.authenticate(
            api_key=None,
        )

    assert error.value.status_code == 401
    assert error.value.detail == (
        "Invalid or missing API key"
    )


def test_invalid_api_key_returns_401() -> None:
    config = create_settings()

    authenticator = APIKeyAuthenticator(
        config=config,
    )

    with pytest.raises(
        HTTPException,
    ) as error:
        authenticator.authenticate(
            api_key="wrong-key",
        )

    assert error.value.status_code == 401
    assert error.value.detail == (
        "Invalid or missing API key"
    )


def test_whitespace_only_api_key_returns_401() -> None:
    config = create_settings()

    authenticator = APIKeyAuthenticator(
        config=config,
    )

    with pytest.raises(
        HTTPException,
    ) as error:
        authenticator.authenticate(
            api_key="   ",
        )

    assert error.value.status_code == 401


def test_missing_auth_configuration_returns_503() -> None:
    config = create_settings(
        user_api_key="",
        admin_api_key="",
    )

    authenticator = APIKeyAuthenticator(
        config=config,
    )

    with pytest.raises(
        HTTPException,
    ) as error:
        authenticator.authenticate(
            api_key="anything",
        )

    assert error.value.status_code == 503
    assert error.value.detail == (
        "Authentication is not configured"
    )


def test_user_is_allowed_user_operation() -> None:
    config = create_settings()

    authenticator = APIKeyAuthenticator(
        config=config,
    )

    principal = authenticator.authenticate(
        api_key="user-secret",
    )

    authorized = authenticator.authorize(
        principal=principal,
        required_role=UserRole.USER,
    )

    assert authorized == principal


def test_admin_is_allowed_user_operation() -> None:
    config = create_settings()

    authenticator = APIKeyAuthenticator(
        config=config,
    )

    principal = authenticator.authenticate(
        api_key="admin-secret",
    )

    authorized = authenticator.authorize(
        principal=principal,
        required_role=UserRole.USER,
    )

    assert authorized == principal


def test_user_is_not_allowed_admin_operation() -> None:
    config = create_settings()

    authenticator = APIKeyAuthenticator(
        config=config,
    )

    principal = authenticator.authenticate(
        api_key="user-secret",
    )

    with pytest.raises(
        HTTPException,
    ) as error:
        authenticator.authorize(
            principal=principal,
            required_role=UserRole.ADMIN,
        )

    assert error.value.status_code == 403
    assert error.value.detail == (
        "Insufficient permissions"
    )


def test_admin_is_allowed_admin_operation() -> None:
    config = create_settings()

    authenticator = APIKeyAuthenticator(
        config=config,
    )

    principal = authenticator.authenticate(
        api_key="admin-secret",
    )

    authorized = authenticator.authorize(
        principal=principal,
        required_role=UserRole.ADMIN,
    )

    assert authorized == principal


def test_role_values_are_stable() -> None:
    assert UserRole.USER.value == "user"
    assert UserRole.ADMIN.value == "admin"