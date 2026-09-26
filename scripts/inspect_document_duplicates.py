from collections import defaultdict

from sqlalchemy import select

from app.database import SessionLocal
from app.models.chunk import ChunkModel
from app.models.document import DocumentModel
from app.models.ingestion_job import IngestionJobModel


def main() -> None:
    with SessionLocal() as session:
        documents = session.execute(
            select(DocumentModel).order_by(DocumentModel.created_at)
        ).scalars().all()

        chunks = session.execute(
            select(ChunkModel).order_by(
                ChunkModel.doc_id,
                ChunkModel.chunk_index,
            )
        ).scalars().all()

        jobs = session.execute(
            select(IngestionJobModel).order_by(
                IngestionJobModel.created_at
            )
        ).scalars().all()

    chunks_by_doc = defaultdict(list)

    for chunk in chunks:
        chunks_by_doc[chunk.doc_id].append(chunk)

    jobs_by_doc = defaultdict(list)

    for job in jobs:
        if job.doc_id:
            jobs_by_doc[job.doc_id].append(job)

    print("\nDOCUMENTS")
    print("=" * 80)

    for document in documents:
        print(f"doc_id       : {document.doc_id}")
        print(f"filename     : {document.filename}")
        print(f"content_hash : {document.content_hash}")
        print(f"stored_path  : {document.stored_path}")
        print(f"chunks       : {len(chunks_by_doc[document.doc_id])}")
        print(f"jobs         : {len(jobs_by_doc[document.doc_id])}")
        print("-" * 80)

        for chunk in chunks_by_doc[document.doc_id]:
            print(
                f"  chunk: {chunk.chunk_id} "
                f"(page={chunk.page_number}, index={chunk.chunk_index})"
            )

        for job in jobs_by_doc[document.doc_id]:
            print(
                f"  job: {job.job_id} "
                f"(status={job.status})"
            )

    print("\nDUPLICATE HASHES")
    print("=" * 80)

    documents_by_hash = defaultdict(list)

    for document in documents:
        if document.content_hash:
            documents_by_hash[document.content_hash].append(document)

    for content_hash, matching_documents in documents_by_hash.items():
        if len(matching_documents) <= 1:
            continue

        print(f"Hash: {content_hash}")

        for document in matching_documents:
            print(
                f"  {document.doc_id} "
                f"-> {document.filename}"
            )


if __name__ == "__main__":
    main()