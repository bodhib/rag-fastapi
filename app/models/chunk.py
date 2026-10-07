from sqlalchemy import ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


class ChunkModel(Base):
    """
    SQLAlchemy model representing a searchable document chunk.

    Chunks are persisted in PostgreSQL so lexical retrieval can be
    shared across multiple API instances.
    """

    __tablename__ = "document_chunks"

    chunk_id: Mapped[str] = mapped_column(
        String(128),
        primary_key=True,
    )

    doc_id: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        index=True,
    )

    text: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    source: Mapped[str] = mapped_column(
        String(1024),
        nullable=False,
    )

    page_number: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )

    chunk_index: Mapped[int] = mapped_column(
        Integer,
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