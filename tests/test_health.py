from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_health_check() -> None:
    """Verify that the health endpoint works."""
    response = client.get("/health")
    print(response.json())

    assert response.status_code == 200

    assert response.json() == {
        "status": "ok",
        "service": "rag-api",
    }
