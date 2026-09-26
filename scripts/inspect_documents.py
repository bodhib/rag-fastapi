from sqlalchemy import text

from app.database import engine


with engine.connect() as conn:
    rows = conn.execute(
        text(
            """
            SELECT
                doc_id,
                filename,
                stored_path,
                content_hash
            FROM documents
            ORDER BY created_at
            """
        )
    ).fetchall()


print(f"Documents: {len(rows)}")

for row in rows:
    print(row)