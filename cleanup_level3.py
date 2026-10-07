"""
One-time cleanup for Level 3 recruiter test data.

This script removes only recruiter/candidate test resumes and their
associated data from the configured knowledge base.

It does NOT delete:
- the knowledge base itself
- Level 1 policy documents
- Level 2 company documents that do not use candidate_*.txt
- application source code

Run from the project root:

    python scripts/cleanup_level3.py
"""

from pathlib import Path

from sqlalchemy import delete, select

from app.config import settings
from app.database import SessionLocal
from app.models.candidate import CandidateModel
from app.models.chunk import ChunkModel
from app.models.document import DocumentModel
from app.models.ingestion_job import IngestionJobModel
from app.vector_store import ChromaVectorStore


KNOWLEDGE_BASE_ID = "kb_company"

# We deliberately use the filename pattern rather than deleting every
# document in kb_company.
RESUME_PREFIX = "candidate_"


def main() -> None:
    print()
    print("=" * 70)
    print("LEVEL 3 RECRUITER DATA CLEANUP")
    print("=" * 70)
    print(f"Knowledge base : {KNOWLEDGE_BASE_ID}")
    print(f"Filename rule  : {RESUME_PREFIX}*.txt")
    print()

    # ---------------------------------------------------------------
    # 1. Find candidate resume documents
    # ---------------------------------------------------------------

    with SessionLocal() as session:
        documents = session.scalars(
            select(DocumentModel).where(
                DocumentModel.knowledge_base_id
                == KNOWLEDGE_BASE_ID,
                DocumentModel.filename.startswith(
                    RESUME_PREFIX
                ),
            )
        ).all()

        candidates = session.scalars(
            select(CandidateModel).where(
                CandidateModel.knowledge_base_id
                == KNOWLEDGE_BASE_ID,
                CandidateModel.resume_filename.startswith(
                    RESUME_PREFIX
                ),
            )
        ).all()

        document_ids = {
            document.doc_id
            for document in documents
        }

        candidate_ids = {
            candidate.candidate_id
            for candidate in candidates
        }

        filenames = {
            document.filename
            for document in documents
        }

        filenames.update(
            candidate.resume_filename
            for candidate in candidates
        )

    print(f"Candidates found : {len(candidate_ids)}")
    print(f"Documents found  : {len(document_ids)}")
    print(f"Resume filenames  : {len(filenames)}")

    if filenames:
        print()
        print("Files to remove:")
        for filename in sorted(filenames):
            print(f"  - {filename}")

    # ---------------------------------------------------------------
    # 2. Safety check
    # ---------------------------------------------------------------

    if not candidate_ids and not document_ids and not filenames:
        print()
        print("Nothing to clean.")
        print("=" * 70)
        return

    print()
    print("Only the above candidate/resume data will be removed.")
    print("The knowledge base itself will NOT be deleted.")
    print()

    confirmation = input(
        "Type CLEAN_LEVEL3 to continue: "
    ).strip()

    if confirmation != "CLEAN_LEVEL3":
        print()
        print("Cleanup cancelled.")
        return

    # ---------------------------------------------------------------
    # 3. Remove Chroma vectors
    # ---------------------------------------------------------------

    print()
    print("Removing Chroma vectors...")

    vector_store = ChromaVectorStore(
        persist_directory=settings.chroma_directory,
    )

    vector_count_before = vector_store.count

    for doc_id in sorted(document_ids):
        vector_store.delete_document(doc_id)

    vector_count_after = vector_store.count

    print(
        "Chroma vectors removed: "
        f"{vector_count_before - vector_count_after}"
    )

    # ---------------------------------------------------------------
    # 4. Remove SQL database records
    # ---------------------------------------------------------------

    print("Removing database records...")

    with SessionLocal() as session:

        # -----------------------------------------------------------
        # 4a. Delete lexical chunks
        # -----------------------------------------------------------

        if document_ids:
            chunk_result = session.execute(
                delete(ChunkModel).where(
                    ChunkModel.doc_id.in_(
                        document_ids
                    )
                )
            )
        else:
            chunk_result = None

        chunk_count = (
            chunk_result.rowcount
            if chunk_result is not None
            else 0
        )

        # -----------------------------------------------------------
        # 4b. Delete ingestion jobs belonging to these resumes
        # -----------------------------------------------------------

        job_result = session.execute(
            delete(IngestionJobModel).where(
                IngestionJobModel.knowledge_base_id
                == KNOWLEDGE_BASE_ID,
                IngestionJobModel.filename.startswith(
                    RESUME_PREFIX
                ),
            )
        )

        job_count = job_result.rowcount

        # -----------------------------------------------------------
        # 4c. Delete candidates
        # -----------------------------------------------------------

        candidate_result = session.execute(
            delete(CandidateModel).where(
                CandidateModel.knowledge_base_id
                == KNOWLEDGE_BASE_ID,
                CandidateModel.resume_filename.startswith(
                    RESUME_PREFIX
                ),
            )
        )

        candidate_count = candidate_result.rowcount

        # -----------------------------------------------------------
        # 4d. Delete document records
        # -----------------------------------------------------------

        if document_ids:
            document_result = session.execute(
                delete(DocumentModel).where(
                    DocumentModel.doc_id.in_(
                        document_ids
                    )
                )
            )

            document_count = document_result.rowcount

        else:
            document_count = 0

        session.commit()

    print(f"Chunks removed       : {chunk_count}")
    print(f"Ingestion jobs removed: {job_count}")
    print(f"Candidates removed   : {candidate_count}")
    print(f"Documents removed    : {document_count}")

    # ---------------------------------------------------------------
    # 5. Remove uploaded physical files
    # ---------------------------------------------------------------

    print("Removing uploaded resume files...")

    removed_files = 0

    upload_directory = settings.upload_directory

    if upload_directory.exists():

        for filename in filenames:

            file_path = upload_directory / Path(filename).name

            if file_path.exists() and file_path.is_file():
                file_path.unlink()
                removed_files += 1
                print(f"  removed: {file_path}")

    print(f"Uploaded files removed: {removed_files}")

    # ---------------------------------------------------------------
    # 6. Final verification
    # ---------------------------------------------------------------

    print()
    print("Running final verification...")

    with SessionLocal() as session:

        remaining_candidates = session.scalars(
            select(CandidateModel).where(
                CandidateModel.knowledge_base_id
                == KNOWLEDGE_BASE_ID,
                CandidateModel.resume_filename.startswith(
                    RESUME_PREFIX
                ),
            )
        ).all()

        remaining_documents = session.scalars(
            select(DocumentModel).where(
                DocumentModel.knowledge_base_id
                == KNOWLEDGE_BASE_ID,
                DocumentModel.filename.startswith(
                    RESUME_PREFIX
                ),
            )
        ).all()

        remaining_chunks = session.scalars(
            select(ChunkModel).where(
                ChunkModel.knowledge_base_id
                == KNOWLEDGE_BASE_ID,
                ChunkModel.doc_id.in_(
                    [
                        document.doc_id
                        for document in remaining_documents
                    ]
                ),
            )
        ).all()

    print()
    print("Remaining candidate records:", len(remaining_candidates))
    print("Remaining resume documents:", len(remaining_documents))
    print("Remaining resume chunks   :", len(remaining_chunks))

    if remaining_candidates:
        print()
        print("WARNING: Some candidate records remain.")

    if remaining_documents:
        print()
        print("WARNING: Some resume documents remain.")

    if remaining_chunks:
        print()
        print("WARNING: Some resume chunks remain.")

    print()
    print("=" * 70)

    if (
        not remaining_candidates
        and not remaining_documents
        and not remaining_chunks
    ):
        print("LEVEL 3 CLEANUP COMPLETED SUCCESSFULLY")
    else:
        print("LEVEL 3 CLEANUP COMPLETED WITH WARNINGS")

    print("=" * 70)
    print()


if __name__ == "__main__":
    main()

