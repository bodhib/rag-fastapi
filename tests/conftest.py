import os
from datetime import datetime

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database import Base
from app.models.candidate import CandidateModel
from app.models.chunk import ChunkModel
from app.models.document import DocumentModel
from app.models.knowledge_base import KnowledgeBaseModel


@pytest.fixture
def postgres_session_factory():
    """
    Create an isolated PostgreSQL schema for one test.

    The test database itself is persistent, but the tables used by
    this fixture are created at test setup and dropped at teardown.

    IMPORTANT:
    This fixture intentionally uses SQLAlchemy metadata instead of
    Alembic. Alembic is tested separately through migration checks.
    """

    database_url = os.environ["TEST_DATABASE_URL"]

    engine = create_engine(
        database_url,
    )

    Base.metadata.create_all(
        bind=engine,
        tables=[
            KnowledgeBaseModel.__table__,
            DocumentModel.__table__,
            ChunkModel.__table__,
            CandidateModel.__table__,
        ],
    )

    session_factory = sessionmaker(
        bind=engine,
        autocommit=False,
        autoflush=False,
        expire_on_commit=False,
    )

    with session_factory() as session:

        existing_default = session.get(
            KnowledgeBaseModel,
            "kb_default",
        )

        if existing_default is None:
            session.add(
                KnowledgeBaseModel(
                    knowledge_base_id="kb_default",
                    name="My Documents",
                    slug="my-documents",
                    description=None,
                    active=True,
                    created_at=datetime.now(),
                )
            )

        existing_company = session.get(
            KnowledgeBaseModel,
            "kb_company",
        )

        if existing_company is None:
            session.add(
                KnowledgeBaseModel(
                    knowledge_base_id="kb_company",
                    name="Company Knowledge",
                    slug="company",
                    description=None,
                    active=True,
                    created_at=datetime.now(),
                )
            )

        existing_recruitment = session.get(
            KnowledgeBaseModel,
            "kb_recruitment",
        )

        if existing_recruitment is None:
            session.add(
                KnowledgeBaseModel(
                    knowledge_base_id="kb_recruitment",
                    name="Recruitment",
                    slug="recruitment",
                    description=(
                        "Recruiter resumes and "
                        "candidate knowledge"
                    ),
                    active=True,
                    created_at=datetime.now(),
                )
            )

        session.commit()

    try:
        yield session_factory

    finally:
        Base.metadata.drop_all(
            bind=engine,
            tables=[
                CandidateModel.__table__,
                ChunkModel.__table__,
                DocumentModel.__table__,
                KnowledgeBaseModel.__table__,
            ],
        )

        engine.dispose()