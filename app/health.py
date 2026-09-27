"""
Application health and readiness checks.

Liveness:
    Confirms that the FastAPI process is alive.

Readiness:
    Confirms that the application's critical runtime
    dependencies are available.

Health endpoints deliberately do not expose:
    - API keys
    - database passwords
    - exception stack traces
    - internal filesystem paths
"""

from dataclasses import dataclass
from typing import Callable

from sqlalchemy import text

from app.database import SessionLocal
from app.dependencies import (
    get_retrieval_index_manager,
    get_vector_store,
)


@dataclass(frozen=True)
class DependencyCheck:
    """
    Result of checking one application dependency.
    """

    name: str
    ok: bool


@dataclass(frozen=True)
class ReadinessResult:
    """
    Aggregated readiness result.
    """

    ready: bool
    checks: tuple[DependencyCheck, ...]


def check_database() -> DependencyCheck:
    """
    Verify that the application database can execute a query.
    """

    try:
        with SessionLocal() as session:
            result = session.execute(
                text("SELECT 1")
            )

            if result.scalar_one() != 1:
                return DependencyCheck(
                    name="database",
                    ok=False,
                )

        return DependencyCheck(
            name="database",
            ok=True,
        )

    except Exception:
        return DependencyCheck(
            name="database",
            ok=False,
        )


def check_vector_store() -> DependencyCheck:
    """
    Verify that Chroma is available and can read stored chunks.
    """

    try:
        vector_store = get_vector_store()
        vector_store.get_all_chunks()

        return DependencyCheck(
            name="vector_store",
            ok=True,
        )

    except Exception:
        return DependencyCheck(
            name="vector_store",
            ok=False,
        )
    
def check_retrieval_index() -> DependencyCheck:
    """
    Verify that the lexical retrieval index manager can be
    initialized/accessed.

    The dependency itself is responsible for rebuilding from
    authoritative vector-store data when required.
    """

    try:
        get_retrieval_index_manager()

        return DependencyCheck(
            name="retrieval_index",
            ok=True,
        )

    except Exception:
        return DependencyCheck(
            name="retrieval_index",
            ok=False,
        )


def run_readiness_checks(
    checks: tuple[
        Callable[[], DependencyCheck],
        ...,
    ]
    | None = None,
) -> ReadinessResult:
    """
    Execute readiness checks.

    The optional checks parameter exists primarily to keep this
    function easy to test with deterministic fakes.
    """

    selected_checks = checks or (
        check_database,
        check_vector_store,
        check_retrieval_index,
    )

    results = tuple(
        check()
        for check in selected_checks
    )

    ready = all(
        check.ok
        for check in results
    )

    return ReadinessResult(
        ready=ready,
        checks=results,
    )