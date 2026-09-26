from fastapi.testclient import TestClient

from app.health import (
    DependencyCheck,
    ReadinessResult,
    run_readiness_checks,
)
from app.main import app


def test_readiness_is_ready_when_all_checks_pass() -> None:
    checks = (
        lambda: DependencyCheck(
            name="database",
            ok=True,
        ),
        lambda: DependencyCheck(
            name="vector_store",
            ok=True,
        ),
        lambda: DependencyCheck(
            name="retrieval_index",
            ok=True,
        ),
    )

    result = run_readiness_checks(
        checks=checks,
    )

    assert result.ready is True


def test_readiness_is_not_ready_when_one_check_fails() -> None:
    checks = (
        lambda: DependencyCheck(
            name="database",
            ok=False,
        ),
        lambda: DependencyCheck(
            name="vector_store",
            ok=True,
        ),
        lambda: DependencyCheck(
            name="retrieval_index",
            ok=True,
        ),
    )

    result = run_readiness_checks(
        checks=checks,
    )

    assert result.ready is False
    assert result.checks[0].ok is False


def test_readiness_is_not_ready_when_multiple_checks_fail() -> None:
    checks = (
        lambda: DependencyCheck(
            name="database",
            ok=False,
        ),
        lambda: DependencyCheck(
            name="vector_store",
            ok=False,
        ),
        lambda: DependencyCheck(
            name="retrieval_index",
            ok=True,
        ),
    )

    result = run_readiness_checks(
        checks=checks,
    )

    assert result.ready is False

    assert result.checks[0].ok is False
    assert result.checks[1].ok is False
    assert result.checks[2].ok is True


def test_health_endpoint_remains_public() -> None:
    client = TestClient(app)

    response = client.get("/health")

    assert response.status_code == 200

    assert response.json() == {
        "status": "ok",
        "service": "rag-api",
    }


def test_readiness_endpoint_returns_200_when_ready(
    monkeypatch,
) -> None:
    import app.main as main

    monkeypatch.setattr(
        main,
        "run_readiness_checks",
        lambda: ReadinessResult(
            ready=True,
            checks=(
                DependencyCheck(
                    name="database",
                    ok=True,
                ),
                DependencyCheck(
                    name="vector_store",
                    ok=True,
                ),
                DependencyCheck(
                    name="retrieval_index",
                    ok=True,
                ),
            ),
        ),
    )

    client = TestClient(app)

    response = client.get(
        "/health/ready"
    )

    assert response.status_code == 200

    assert response.json() == {
        "status": "ready",
        "service": "rag-api",
        "checks": {
            "database": {
                "ok": True,
            },
            "vector_store": {
                "ok": True,
            },
            "retrieval_index": {
                "ok": True,
            },
        },
    }


def test_readiness_endpoint_returns_503_when_not_ready(
    monkeypatch,
) -> None:
    import app.main as main

    monkeypatch.setattr(
        main,
        "run_readiness_checks",
        lambda: ReadinessResult(
            ready=False,
            checks=(
                DependencyCheck(
                    name="database",
                    ok=False,
                ),
                DependencyCheck(
                    name="vector_store",
                    ok=True,
                ),
                DependencyCheck(
                    name="retrieval_index",
                    ok=True,
                ),
            ),
        ),
    )

    client = TestClient(app)

    response = client.get(
        "/health/ready"
    )

    assert response.status_code == 503

    assert response.json() == {
        "status": "not_ready",
        "service": "rag-api",
        "checks": {
            "database": {
                "ok": False,
            },
            "vector_store": {
                "ok": True,
            },
            "retrieval_index": {
                "ok": True,
            },
        },
    }


def test_database_check_fails_closed(
    monkeypatch,
) -> None:
    import app.health as health

    class FailingSession:
        def __enter__(self):
            raise RuntimeError(
                "database unavailable"
            )

        def __exit__(
            self,
            exc_type,
            exc_value,
            traceback,
        ):
            return False

    monkeypatch.setattr(
        health,
        "SessionLocal",
        lambda: FailingSession(),
    )

    result = health.check_database()

    assert result == DependencyCheck(
        name="database",
        ok=False,
    )


def test_vector_store_check_fails_closed(
    monkeypatch,
) -> None:
    import app.health as health

    class FailingVectorStore:
        @property
        def count(self):
            raise RuntimeError(
                "vector store unavailable"
            )

    monkeypatch.setattr(
        health,
        "get_vector_store",
        lambda: FailingVectorStore(),
    )

    result = health.check_vector_store()

    assert result == DependencyCheck(
        name="vector_store",
        ok=False,
    )


def test_retrieval_index_check_fails_closed(
    monkeypatch,
) -> None:
    import app.health as health

    def failing_manager():
        raise RuntimeError(
            "retrieval index unavailable"
        )

    monkeypatch.setattr(
        health,
        "get_retrieval_index_manager",
        failing_manager,
    )

    result = health.check_retrieval_index()

    assert result == DependencyCheck(
        name="retrieval_index",
        ok=False,
    )