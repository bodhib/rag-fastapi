from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


class IngestionJobModel(Base):
    """
    SQLAlchemy model representing one document ingestion job.

    The job record allows the API to track ingestion independently
    from the HTTP request that initiated it.
    """

    __tablename__ = "ingestion_jobs"

    job_id: Mapped[str] = mapped_column(
        String(64),
        primary_key=True,
    )

    filename: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )

    file_path: Mapped[str] = mapped_column(
        String(1024),
        nullable=False,
    )

    status: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
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

    doc_id: Mapped[str | None] = mapped_column(
        String(64),
        nullable=True,
    )

    page_count: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )

    chunk_count: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )

    error_message: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
    )

    started_at: Mapped[datetime | None] = mapped_column(
        DateTime,
        nullable=True,
    )

    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime,
        nullable=True,
    )

