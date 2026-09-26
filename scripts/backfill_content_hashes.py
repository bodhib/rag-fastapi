import hashlib
from pathlib import Path

from sqlalchemy import select

from app.database import SessionLocal
from app.models.document import DocumentModel


def calculate_sha256(file_path: Path) -> str:
    sha256 = hashlib.sha256()

    with file_path.open("rb") as file:
        while chunk := file.read(1024 * 1024):
            sha256.update(chunk)

    return sha256.hexdigest()


def main() -> None:
    with SessionLocal() as session:
        documents = session.execute(
            select(DocumentModel)
        ).scalars().all()

        print(f"Documents found: {len(documents)}")

        for document in documents:
            file_path = Path(document.stored_path)

            if not file_path.exists():
                print(
                    f"SKIPPED: {document.doc_id} "
                    f"-> file not found: {file_path}"
                )
                continue

            content_hash = calculate_sha256(file_path)

            document.content_hash = content_hash

            print(
                f"UPDATED: {document.doc_id} "
                f"-> {document.filename} "
                f"-> {content_hash}"
            )

        session.commit()

    print("Content hash backfill completed.")


if __name__ == "__main__":
    main()