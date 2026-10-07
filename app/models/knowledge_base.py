from datetime import datetime

from sqlalchemy import (
    Boolean,
    DateTime,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


class KnowledgeBaseModel(Base):
    """
    SQLAlchemy model representing an isolated collection of documents.

    A knowledge base defines the retrieval boundary for a group of
    documents, such as personal documents, company knowledge, or
    recruitment resumes.
    """

    __tablename__ = "knowledge_bases"

    __table_args__ = (
        UniqueConstraint(
            "slug",
            name="uq_knowledge_bases_slug",
        ),
    )

    knowledge_base_id: Mapped[str] = mapped_column(
        String(64),
        primary_key=True,
    )

    name: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )

    slug: Mapped[str] = mapped_column(
        String(128),
        nullable=False,
    )

    description: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    active: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
    )