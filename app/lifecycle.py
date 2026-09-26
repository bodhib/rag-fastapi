from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.dependencies import (
clear_dependency_caches,
get_generator,
get_retrieval_index_manager,
)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    get_retrieval_index_manager()
    get_generator()

    try:
        yield
    finally:
        clear_dependency_caches()

