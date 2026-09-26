from sqlalchemy import delete

from app.database import SessionLocal
from app.dependencies import get_vector_store
from app.models.chunk import ChunkModel
from app.models.document import DocumentModel
from app.models.ingestion_job import IngestionJobModel


KEEP_DOC_ID = "doc_fa55c2bb1ef4490c85fa0370284b9122"
REMOVE_DOC_ID = "doc_96b19724bb2d44a6a6e38e2fb0ac41c4"


def main() -> None:
    vector_store = get_vector_store()

    print(f"Canonical document : {KEEP_DOC_ID}")
    print(f"Duplicate document  : {REMOVE_DOC_ID}")
    print()

    # ------------------------------------------------------------------
    # Step 1: Verify the duplicate exists in PostgreSQL.
    # ------------------------------------------------------------------

    with SessionLocal() as session:
        document = session.get(
            DocumentModel,
            REMOVE_DOC_ID,
        )

        if document is None:
            print(
                f"Document {REMOVE_DOC_ID} "
                "does not exist. Nothing to remove."
            )
            return

        print("Duplicate document found:")
        print(f"  filename     : {document.filename}")
        print(f"  content_hash : {document.content_hash}")
        print(f"  stored_path  : {document.stored_path}")

    # ------------------------------------------------------------------
    # Step 2: Remove the duplicate vector from Chroma.
    # ------------------------------------------------------------------

    print()
    print("Removing duplicate document from Chroma...")

    vector_store.delete_document(
        REMOVE_DOC_ID
    )

    print("Chroma cleanup completed.")

    # ------------------------------------------------------------------
    # Step 3: Remove the duplicate PostgreSQL records.
    # ------------------------------------------------------------------

    print()
    print("Removing duplicate PostgreSQL records...")

    with SessionLocal() as session:
        session.execute(
            delete(IngestionJobModel).where(
                IngestionJobModel.doc_id == REMOVE_DOC_ID
            )
        )

        session.execute(
            delete(ChunkModel).where(
                ChunkModel.doc_id == REMOVE_DOC_ID
            )
        )

        document = session.get(
            DocumentModel,
            REMOVE_DOC_ID,
        )

        if document is not None:
            session.delete(document)

        session.commit()

    print("PostgreSQL cleanup completed.")

    # ------------------------------------------------------------------
    # Step 4: Verify.
    # ------------------------------------------------------------------

    with SessionLocal() as session:
        remaining_document = session.get(
            DocumentModel,
            REMOVE_DOC_ID,
        )

        remaining_chunks = session.execute(
            ChunkModel.__table__.select().where(
                ChunkModel.doc_id == REMOVE_DOC_ID
            )
        ).fetchall()

        remaining_jobs = session.execute(
            IngestionJobModel.__table__.select().where(
                IngestionJobModel.doc_id == REMOVE_DOC_ID
            )
        ).fetchall()

    remaining_vectors = [
        chunk
        for chunk in vector_store.get_all_chunks()
        if chunk.doc_id == REMOVE_DOC_ID
    ]

    print()
    print("VERIFICATION")
    print("=" * 80)
    print(f"PostgreSQL document : {remaining_document}")
    print(f"PostgreSQL chunks   : {len(remaining_chunks)}")
    print(f"PostgreSQL jobs     : {len(remaining_jobs)}")
    print(f"Chroma vectors      : {len(remaining_vectors)}")
    print()

    if (
        remaining_document is None
        and not remaining_chunks
        and not remaining_jobs
        and not remaining_vectors
    ):
        print("SUCCESS: duplicate document completely removed.")
    else:
        raise RuntimeError(
            "Duplicate cleanup verification failed."
        )


if __name__ == "__main__":
    main()