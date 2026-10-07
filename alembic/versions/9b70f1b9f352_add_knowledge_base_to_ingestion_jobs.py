"""add knowledge base to ingestion jobs

Revision ID: 9b70f1b9f352
Revises: 1bf958aa8fdd
Create Date: 2026-09-29 12:40:16.410356

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '9b70f1b9f352'
down_revision: Union[str, Sequence[str], None] = '1bf958aa8fdd'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "ingestion_jobs",
        sa.Column(
            "knowledge_base_id",
            sa.String(length=64),
            nullable=True,
        ),
    )

    op.execute(
        sa.text(
            """
            UPDATE ingestion_jobs
            SET knowledge_base_id = :knowledge_base_id
            WHERE knowledge_base_id IS NULL
            """
        ).bindparams(
            knowledge_base_id="kb_default",
        )
    )

    op.alter_column(
        "ingestion_jobs",
        "knowledge_base_id",
        existing_type=sa.String(length=64),
        nullable=False,
    )

    op.create_foreign_key(
        "fk_ingestion_jobs_knowledge_base_id",
        "ingestion_jobs",
        "knowledge_bases",
        ["knowledge_base_id"],
        ["knowledge_base_id"],
        ondelete="RESTRICT",
    )

    op.create_index(
        "ix_ingestion_jobs_knowledge_base_id",
        "ingestion_jobs",
        ["knowledge_base_id"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(
        "ix_ingestion_jobs_knowledge_base_id",
        table_name="ingestion_jobs",
    )

    op.drop_constraint(
        "fk_ingestion_jobs_knowledge_base_id",
        "ingestion_jobs",
        type_="foreignkey",
    )

    op.drop_column(
        "ingestion_jobs",
        "knowledge_base_id",
    )
