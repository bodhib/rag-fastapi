import os

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database import Base
from app.models.chunk import ChunkModel


@pytest.fixture
def postgres_session_factory():
    database_url = os.environ["TEST_DATABASE_URL"]

    engine = create_engine(database_url)

    Base.metadata.create_all(
        bind=engine,
        tables=[ChunkModel.__table__],
    )

    session_factory = sessionmaker(
        bind=engine,
        autocommit=False,
        autoflush=False,
    )

    try:
        yield session_factory
    finally:
        Base.metadata.drop_all(
            bind=engine,
            tables=[ChunkModel.__table__],
        )
        engine.dispose()