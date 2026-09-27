from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.dependencies import clear_dependency_caches


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """
    Manage application startup and shutdown.

    Heavy RAG components are initialized lazily when first needed
    rather than during application startup. This keeps startup memory
    usage low and allows health endpoints to become available quickly.
    """

    try:
        yield
    finally:
        clear_dependency_caches()