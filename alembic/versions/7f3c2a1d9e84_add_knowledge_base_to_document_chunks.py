"""add knowledge base to document chunks

Revision ID: 7f3c2a1d9e84
Revises: 9b70f1b9f352
Create Date: 2026-09-29
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "7f3c2a1d9e84"
down_revision: Union[str, Sequence[str], None] = "9b70f1b9f352"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ---------------------------------------------------------
    # 1. Add the column as nullable initially.
    # ---------------------------------------------------------
    op.add_column(
        "document_chunks",
        sa.Column(
            "knowledge_base_id",
            sa.String(length=64),
            nullable=True,
        ),
    )

    # ---------------------------------------------------------
    # 2. Remove orphaned chunks.
    #
    # These chunks reference a doc_id that no longer exists in
    # the documents table, so there is no reliable knowledge
    # base that can be assigned to them.
    # ---------------------------------------------------------
    op.execute(
        sa.text(
            """
            DELETE FROM document_chunks AS c
            WHERE NOT EXISTS (
                SELECT 1
                FROM documents AS d
                WHERE d.doc_id = c.doc_id
            )
            """
        )
    )

    # ---------------------------------------------------------
    # 3. Backfill the knowledge base from the parent document.
    # ---------------------------------------------------------
    op.execute(
        sa.text(
            """
            UPDATE document_chunks AS c
            SET knowledge_base_id = d.knowledge_base_id
            FROM documents AS d
            WHERE c.doc_id = d.doc_id
            """
        )
    )

    # ---------------------------------------------------------
    # 4. Safety check.
    #
    # Do not silently continue if something unexpected remains.
    # ---------------------------------------------------------
    connection = op.get_bind()

    null_count = connection.execute(
        sa.text(
            """
            SELECT COUNT(*)
            FROM document_chunks
            WHERE knowledge_base_id IS NULL
            """
        )
    ).scalar_one()

    if null_count != 0:
        raise RuntimeError(
            "Cannot make document_chunks.knowledge_base_id NOT NULL: "
            f"{null_count} rows still have NULL knowledge_base_id."
        )

    # ---------------------------------------------------------
    # 5. Make the column mandatory.
    # ---------------------------------------------------------
    op.alter_column(
        "document_chunks",
        "knowledge_base_id",
        existing_type=sa.String(length=64),
        nullable=False,
    )

    # ---------------------------------------------------------
    # 6. Add the foreign-key constraint.
    # ---------------------------------------------------------
    op.create_foreign_key(
        "fk_document_chunks_knowledge_base_id",
        "document_chunks",
        "knowledge_bases",
        ["knowledge_base_id"],
        ["knowledge_base_id"],
        ondelete="RESTRICT",
    )

    # ---------------------------------------------------------
    # 7. Add an index for KB-scoped retrieval.
    # ---------------------------------------------------------
    op.create_index(
        "ix_document_chunks_knowledge_base_id",
        "document_chunks",
        ["knowledge_base_id"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(
        "ix_document_chunks_knowledge_base_id",
        table_name="document_chunks",
    )

    op.drop_constraint(
        "fk_document_chunks_knowledge_base_id",
        "document_chunks",
        type_="foreignkey",
    )

    op.drop_column(
        "document_chunks",
        "knowledge_base_id",
    )