"""add knowledge bases

Revision ID: 1bf958aa8fdd
Revises: 70390ef3a3f1
Create Date: 2026-09-29 09:45:26.729522

"""
from datetime import datetime
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '1bf958aa8fdd'
down_revision: Union[str, Sequence[str], None] = '70390ef3a3f1'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # 1. Create the knowledge_bases table first.
    op.create_table(
        "knowledge_bases",
        sa.Column(
            "knowledge_base_id",
            sa.String(length=64),
            nullable=False,
        ),
        sa.Column(
            "name",
            sa.String(length=255),
            nullable=False,
        ),
        sa.Column(
            "slug",
            sa.String(length=128),
            nullable=False,
        ),
        sa.Column(
            "description",
            sa.Text(),
            nullable=True,
        ),
        sa.Column(
            "active",
            sa.Boolean(),
            nullable=False,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("knowledge_base_id"),
        sa.UniqueConstraint(
            "slug",
            name="uq_knowledge_bases_slug",
        ),
    )

    # 2. Create the default knowledge base.
    #
    # Existing documents need a valid knowledge base before the
    # documents.knowledge_base_id column can become NOT NULL.
    op.execute(
        sa.text(
            """
            INSERT INTO knowledge_bases (
                knowledge_base_id,
                name,
                slug,
                description,
                active,
                created_at
            )
            VALUES (
                :knowledge_base_id,
                :name,
                :slug,
                :description,
                :active,
                :created_at
            )
            """
        ).bindparams(
            knowledge_base_id="kb_default",
            name="My Documents",
            slug="my-documents",
            description="Default knowledge base for existing documents.",
            active=True,
            created_at=datetime.utcnow(),
        )
    )

    # 3. Add the new document column temporarily as nullable.
    op.add_column(
        "documents",
        sa.Column(
            "knowledge_base_id",
            sa.String(length=64),
            nullable=True,
        ),
    )

    # 4. Assign every existing document to the default KB.
    op.execute(
        sa.text(
            """
            UPDATE documents
            SET knowledge_base_id = :knowledge_base_id
            WHERE knowledge_base_id IS NULL
            """
        ).bindparams(
            knowledge_base_id="kb_default",
        )
    )

    # 5. The column can now safely become NOT NULL.
    op.alter_column(
        "documents",
        "knowledge_base_id",
        existing_type=sa.String(length=64),
        nullable=False,
    )

    # 6. Add the foreign-key relationship.
    op.create_foreign_key(
        "fk_documents_knowledge_base_id",
        "documents",
        "knowledge_bases",
        ["knowledge_base_id"],
        ["knowledge_base_id"],
        ondelete="RESTRICT",
    )

    # 7. Add an index for KB-scoped document queries.
    op.create_index(
        "ix_documents_knowledge_base_id",
        "documents",
        ["knowledge_base_id"],
        unique=False,
    )


def downgrade() -> None:
    # Reverse the operations in dependency-safe order.
    op.drop_index(
        "ix_documents_knowledge_base_id",
        table_name="documents",
    )

    op.drop_constraint(
        "fk_documents_knowledge_base_id",
        "documents",
        type_="foreignkey",
    )

    op.drop_column(
        "documents",
        "knowledge_base_id",
    )

    op.drop_table("knowledge_bases")
