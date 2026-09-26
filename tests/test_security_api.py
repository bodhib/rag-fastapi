from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

import app.security as security
from app.security import (
    APIKeyAuthenticator,
    UserRole,
    require_role,
)


def create_test_application() -> FastAPI:
    test_app = FastAPI()

    @test_app.get("/public")
    def public_endpoint():
        return {
            "status": "ok",
        }

    @test_app.get(
        "/user",
        dependencies=[
            Depends(
                require_role(UserRole.USER)
            )
        ],
    )
    def user_endpoint():
        return {
            "status": "user-ok",
        }

    @test_app.get(
        "/admin",
        dependencies=[
            Depends(
                require_role(UserRole.ADMIN)
            )
        ],
    )
    def admin_endpoint():
        return {
            "status": "admin-ok",
        }

    return test_app


def test_public_endpoint_requires_no_auth() -> None:
    client = TestClient(
        create_test_application()
    )

    response = client.get("/public")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
    }


def test_real_api_key_dependency_rejects_missing_key(
    monkeypatch,
) -> None:
    test_config = security.settings.model_copy(
        update={
            "auth_enabled": True,
            "rag_user_api_key": "user-secret",
            "rag_admin_api_key": "admin-secret",
        }
    )

    monkeypatch.setattr(
        security,
        "authenticator",
        APIKeyAuthenticator(
            config=test_config,
        ),
    )

    client = TestClient(
        create_test_application()
    )

    response = client.get("/user")

    assert response.status_code == 401


def test_user_key_can_access_user_endpoint(
    monkeypatch,
) -> None:
    test_config = security.settings.model_copy(
        update={
            "auth_enabled": True,
            "rag_user_api_key": "user-secret",
            "rag_admin_api_key": "admin-secret",
        }
    )

    monkeypatch.setattr(
        security,
        "authenticator",
        APIKeyAuthenticator(
            config=test_config,
        ),
    )

    client = TestClient(
        create_test_application()
    )

    response = client.get(
        "/user",
        headers={
            "X-API-Key": "user-secret",
        },
    )

    assert response.status_code == 200
    assert response.json() == {
        "status": "user-ok",
    }


def test_user_key_cannot_access_admin_endpoint(
    monkeypatch,
) -> None:
    test_config = security.settings.model_copy(
        update={
            "auth_enabled": True,
            "rag_user_api_key": "user-secret",
            "rag_admin_api_key": "admin-secret",
        }
    )

    monkeypatch.setattr(
        security,
        "authenticator",
        APIKeyAuthenticator(
            config=test_config,
        ),
    )

    client = TestClient(
        create_test_application()
    )

    response = client.get(
        "/admin",
        headers={
            "X-API-Key": "user-secret",
        },
    )

    assert response.status_code == 403


def test_admin_key_can_access_admin_endpoint(
    monkeypatch,
) -> None:
    test_config = security.settings.model_copy(
        update={
            "auth_enabled": True,
            "rag_user_api_key": "user-secret",
            "rag_admin_api_key": "admin-secret",
        }
    )

    monkeypatch.setattr(
        security,
        "authenticator",
        APIKeyAuthenticator(
            config=test_config,
        ),
    )

    client = TestClient(
        create_test_application()
    )

    response = client.get(
        "/admin",
        headers={
            "X-API-Key": "admin-secret",
        },
    )

    assert response.status_code == 200
    assert response.json() == {
        "status": "admin-ok",
    }