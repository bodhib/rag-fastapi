"""
SAFE TEST-ENVIRONMENT RESET
===========================

Run this ONLY against a disposable/test RAG database.

It:
1. Refuses APP_ENV=production.
2. Ensures the SQLAlchemy tables exist.
3. Deletes candidates, ingestion jobs, chunks, and documents.
4. Deletes all knowledge bases.
5. Clears the Chroma "documents" collection.
6. Deletes files under the configured upload directory.
7. Recreates exactly three test knowledge bases.

Run from the backend project root:
    python reset_test_environment.py

Stop the FastAPI/Uvicorn process before running this script.
"""

from pathlib import Path
import sys
import shutil

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sqlalchemy import delete

from app.config import settings
from app.database import Base, engine, SessionLocal
from app.models.candidate import CandidateModel
from app.models.chunk import ChunkModel
from app.models.document import DocumentModel
from app.models.ingestion_job import IngestionJobModel
from app.models.knowledge_base import KnowledgeBaseModel
from app.vector_store import ChromaVectorStore


if settings.is_production:
    raise SystemExit(
        "REFUSED: APP_ENV=production. "
        "Run this reset only against a disposable test environment."
    )

print("APP_ENV:", settings.app_env)
print("DATABASE_URL:", settings.database_url)
print("CHROMA_DIRECTORY:", settings.chroma_directory)
print("UPLOAD_DIRECTORY:", settings.upload_directory)

answer = input(
    "\nThis will DELETE all application data in this database "
    "and recreate the three test KBs. Type RESET-TEST to continue: "
).strip()

if answer != "RESET-TEST":
    raise SystemExit("Reset cancelled.")

# Create tables if this is a fresh test database.
Base.metadata.create_all(bind=engine)

# Delete dependent/application data first.
with SessionLocal() as session:
    session.execute(delete(CandidateModel))
    session.execute(delete(IngestionJobModel))
    session.execute(delete(ChunkModel))
    session.execute(delete(DocumentModel))
    session.execute(delete(KnowledgeBaseModel))
    session.commit()

# Clear the application Chroma collection.
vector_store = ChromaVectorStore(
    persist_directory=settings.chroma_directory
)
vector_store.clear()

# Remove uploaded test files.
upload_dir = settings.upload_directory
upload_dir.mkdir(parents=True, exist_ok=True)

for child in upload_dir.iterdir():
    if child.is_dir():
        shutil.rmtree(child)
    else:
        child.unlink()

# Recreate exactly the KBs used by the test plan.
from datetime import datetime

knowledge_bases = [
    {
        "knowledge_base_id": "kb_default",
        "name": "Employee Policies",
        "slug": "employee-policies",
        "description": "HR policies for employee self-service questions.",
    },
    {
        "knowledge_base_id": "kb_company",
        "name": "Acme Corporate Knowledge",
        "slug": "acme-corporate-knowledge",
        "description": "Company policies and operational knowledge.",
    },
    {
        "knowledge_base_id": "kb_recruitment",
        "name": "Recruitment Resumes",
        "slug": "recruitment-resumes",
        "description": "Candidate resumes for recruiter search.",
    },
]

with SessionLocal() as session:
    now = datetime.utcnow()
    for item in knowledge_bases:
        session.add(
            KnowledgeBaseModel(
                **item,
                active=True,
                created_at=now,
            )
        )
    session.commit()

print("\nRESET COMPLETE")
print("PostgreSQL application rows: cleared")
print("Chroma collection 'documents':", vector_store.count, "chunks")
print("Upload directory: cleared")
print("Knowledge bases recreated:")
for item in knowledge_bases:
    print(" -", item["knowledge_base_id"])
