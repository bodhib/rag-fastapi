# tests/test_candidate_store.py

from datetime import datetime

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.candidate_store import CandidateStore
from app.database import Base


@pytest.fixture
def store(tmp_path):
    database_url = f"sqlite:///{tmp_path / 'candidates.db'}"

    engine = create_engine(
        database_url,
        connect_args={"check_same_thread": False},
    )

    Base.metadata.create_all(bind=engine)

    session_factory = sessionmaker(
        bind=engine,
        autocommit=False,
        autoflush=False,
    )

    return CandidateStore(
        session_factory=session_factory,
    )


def test_candidate_create_persists_resume_document_id(
    store: CandidateStore,
):
    candidate = store.create(
        knowledge_base_id="kb_recruitment",
        full_name="Alice Sen",
        email="alice@example.com",
        phone=None,
        location="Kolkata",
        current_title="Senior Python Backend Engineer",
        current_company="Example Technologies",
        years_experience=6,
        skills=["Python", "FastAPI"],
        education="B.Tech",
        availability="Immediate",
        resume_filename="alice_resume.txt",
        resume_document_id="doc_alice_123",
    )

    assert candidate.resume_document_id == "doc_alice_123"

    loaded = store.get(candidate.candidate_id)

    assert loaded is not None
    assert loaded.resume_document_id == "doc_alice_123"


def test_candidate_update_can_change_resume_document_id(
    store: CandidateStore,
):
    candidate = store.create(
        knowledge_base_id="kb_recruitment",
        full_name="Alice Sen",
        email=None,
        phone=None,
        location="Kolkata",
        current_title="Python Engineer",
        current_company="Example Technologies",
        years_experience=6,
        skills=["python"],
        education=None,
        availability=None,
        resume_filename="alice_resume.txt",
        resume_document_id="doc_old",
    )

    updated = store.update(
        candidate.candidate_id,
        resume_document_id="doc_new",
        resume_filename="alice_resume_v2.txt",
    )

    assert updated.resume_document_id == "doc_new"
    assert updated.resume_filename == "alice_resume_v2.txt"

    loaded = store.get(candidate.candidate_id)

    assert loaded is not None
    assert loaded.resume_document_id == "doc_new"


def test_candidate_can_have_resume_filename_different_from_document_filename(
    store: CandidateStore,
):
    candidate = store.create(
        knowledge_base_id="kb_recruitment",
        full_name="Alice Sen",
        email=None,
        phone=None,
        location="Kolkata",
        current_title="Python Engineer",
        current_company="Example Technologies",
        years_experience=6,
        skills=["python", "fastapi"],
        education=None,
        availability=None,
        resume_filename="alice_resume.txt",
        resume_document_id="doc_alice_123",
    )

    assert candidate.resume_filename == "alice_resume.txt"
    assert candidate.resume_document_id == "doc_alice_123"