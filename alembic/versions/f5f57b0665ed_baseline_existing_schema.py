"""baseline existing schema

Revision ID: f5f57b0665ed
Revises:
Create Date: 2026-09-21 00:42:09.991346

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "f5f57b0665ed"
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create the schema that existed before subsequent migrations."""

    op.create_table(
        "documents",
        sa.Column(
            "doc_id",
            sa.String(length=64),
            nullable=False,
        ),
        sa.Column(
            "filename",
            sa.String(length=255),
            nullable=False,
        ),
        sa.Column(
            "stored_path",
            sa.String(length=1024),
            nullable=False,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("doc_id"),
    )

    op.create_table(
        "document_chunks",
        sa.Column(
            "chunk_id",
            sa.String(length=128),
            nullable=False,
        ),
        sa.Column(
            "doc_id",
            sa.String(length=64),
            nullable=False,
        ),
        sa.Column(
            "text",
            sa.Text(),
            nullable=False,
        ),
        sa.Column(
            "source",
            sa.String(length=1024),
            nullable=False,
        ),
        sa.Column(
            "page_number",
            sa.Integer(),
            nullable=False,
        ),
        sa.Column(
            "chunk_index",
            sa.Integer(),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("chunk_id"),
    )

    op.create_index(
        "ix_document_chunks_doc_id",
        "document_chunks",
        ["doc_id"],
        unique=False,
    )

    op.create_table(
        "ingestion_jobs",
        sa.Column(
            "job_id",
            sa.String(length=64),
            nullable=False,
        ),
        sa.Column(
            "filename",
            sa.String(length=255),
            nullable=False,
        ),
        sa.Column(
            "file_path",
            sa.String(length=1024),
            nullable=False,
        ),
        sa.Column(
            "status",
            sa.String(length=32),
            nullable=False,
        ),
        sa.Column(
            "doc_id",
            sa.String(length=64),
            nullable=True,
        ),
        sa.Column(
            "page_count",
            sa.Integer(),
            nullable=True,
        ),
        sa.Column(
            "chunk_count",
            sa.Integer(),
            nullable=True,
        ),
        sa.Column(
            "error_message",
            sa.Text(),
            nullable=True,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(),
            nullable=False,
        ),
        sa.Column(
            "started_at",
            sa.DateTime(),
            nullable=True,
        ),
        sa.Column(
            "completed_at",
            sa.DateTime(),
            nullable=True,
        ),
        sa.PrimaryKeyConstraint("job_id"),
    )


def downgrade() -> None:
    """Drop the original schema."""

    op.drop_table("ingestion_jobs")

    op.drop_index(
        "ix_document_chunks_doc_id",
        table_name="document_chunks",
    )

    op.drop_table("document_chunks")
    op.drop_table("documents")