from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    Boolean,
    DateTime,
    ForeignKey,
    JSON,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


class CandidateModel(Base):
    __tablename__ = "candidates"

    __table_args__ = (
        UniqueConstraint(
            "knowledge_base_id",
            "resume_filename",
            name="uq_candidates_kb_resume_filename",
        ),
    )

    candidate_id: Mapped[str] = mapped_column(
        String(64),
        primary_key=True,
    )

    knowledge_base_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey(
            "knowledge_bases.knowledge_base_id",
            ondelete="RESTRICT",
        ),
        nullable=False,
        index=True,
    )

    full_name: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )

    email: Mapped[str | None] = mapped_column(
        String(320),
        nullable=True,
    )

    phone: Mapped[str | None] = mapped_column(
        String(64),
        nullable=True,
    )

    location: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
        index=True,
    )

    current_title: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
        index=True,
    )

    current_company: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
    )

    years_experience: Mapped[float | None] = mapped_column(
        nullable=True,
        index=True,
    )

    skills: Mapped[list[str]] = mapped_column(
        JSON,
        nullable=False,
        default=list,
    )

    education: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    availability: Mapped[str | None] = mapped_column(
        String(128),
        nullable=True,
    )

    resume_filename: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )

    resume_document_id: Mapped[str | None] = mapped_column(
        String(64),
        ForeignKey(
            "documents.doc_id",
            ondelete="SET NULL",
        ),
        nullable=True,
        unique=True,
    )

    active: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=True,
        index=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
    )