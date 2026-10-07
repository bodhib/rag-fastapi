from __future__ import annotations

from datetime import datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.knowledge_base import KnowledgeBaseModel


class KnowledgeBase:
    """
    Application-level representation of a knowledge base.
    """

    def __init__(
        self,
        knowledge_base_id: str,
        name: str,
        slug: str,
        description: str | None,
        active: bool,
        created_at: datetime,
    ) -> None:
        self.knowledge_base_id = knowledge_base_id
        self.name = name
        self.slug = slug
        self.description = description
        self.active = active
        self.created_at = created_at


class KnowledgeBaseStore:
    """
    Persistence layer for knowledge bases.
    """

    def __init__(self, session_factory) -> None:
        self.session_factory = session_factory

    def create(
        self,
        knowledge_base_id: str,
        name: str,
        slug: str,
        description: str | None = None,
        active: bool = True,
    ) -> KnowledgeBase:
        created_at = datetime.utcnow()

        with self.session_factory() as session:
            model = KnowledgeBaseModel(
                knowledge_base_id=knowledge_base_id,
                name=name,
                slug=slug,
                description=description,
                active=active,
                created_at=created_at,
            )

            session.add(model)
            session.commit()
            session.refresh(model)

            return self._to_domain(model)

    def get(
        self,
        knowledge_base_id: str,
    ) -> KnowledgeBase | None:
        with self.session_factory() as session:
            statement = select(KnowledgeBaseModel).where(
                KnowledgeBaseModel.knowledge_base_id == knowledge_base_id
            )

            model = session.scalar(statement)

            if model is None:
                return None

            return self._to_domain(model)

    def get_by_slug(
        self,
        slug: str,
    ) -> KnowledgeBase | None:
        with self.session_factory() as session:
            statement = select(KnowledgeBaseModel).where(
                KnowledgeBaseModel.slug == slug
            )

            model = session.scalar(statement)

            if model is None:
                return None

            return self._to_domain(model)

    def list(
        self,
        active_only: bool = False,
    ) -> list[KnowledgeBase]:
        with self.session_factory() as session:
            statement = select(KnowledgeBaseModel).order_by(
                KnowledgeBaseModel.created_at
            )

            if active_only:
                statement = statement.where(
                    KnowledgeBaseModel.active.is_(True)
                )

            models = session.scalars(statement).all()

            return [
                self._to_domain(model)
                for model in models
            ]

    @staticmethod
    def _to_domain(
        model: KnowledgeBaseModel,
    ) -> KnowledgeBase:
        return KnowledgeBase(
            knowledge_base_id=model.knowledge_base_id,
            name=model.name,
            slug=model.slug,
            description=model.description,
            active=model.active,
            created_at=model.created_at,
        )