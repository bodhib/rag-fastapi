from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from uuid import uuid4

from sqlalchemy import select
from sqlalchemy.orm import sessionmaker

from app.models.candidate import CandidateModel

from app.document_store import DocumentStore


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    knowledge_base_id: str
    full_name: str
    email: str | None
    phone: str | None
    location: str | None
    current_title: str | None
    current_company: str | None
    years_experience: float | None
    skills: list[str]
    education: str | None
    availability: str | None
    resume_filename: str
    resume_document_id: str | None
    active: bool
    created_at: datetime
    updated_at: datetime


class CandidateNotFoundError(ValueError):
    pass


class DuplicateCandidateError(ValueError):
    pass


class CandidateStore:
    def __init__(
    self,
    session_factory: sessionmaker,
    document_store: DocumentStore,
) -> None:
        self.session_factory = session_factory
        self.document_store = document_store

    def create(
        self,
        *,
        knowledge_base_id: str,
        full_name: str,
        email: str | None = None,
        phone: str | None = None,
        location: str | None = None,
        current_title: str | None = None,
        current_company: str | None = None,
        years_experience: float | None = None,
        skills: list[str] | None = None,
        education: str | None = None,
        availability: str | None = None,
        resume_filename: str,
        resume_document_id: str | None = None,
    ) -> Candidate:

        now = datetime.now()

        if resume_document_id is None:
            document = self.document_store.get_by_filename(
                knowledge_base_id=knowledge_base_id,
                filename=resume_filename,
            )

            if document is not None:
                resume_document_id = document.doc_id

        candidate = CandidateModel(
            candidate_id=f"cand_{uuid4().hex[:16]}",
            knowledge_base_id=knowledge_base_id,
            full_name=full_name.strip(),
            email=email.strip() if email else None,
            phone=phone.strip() if phone else None,
            location=location.strip() if location else None,
            current_title=(
                current_title.strip()
                if current_title
                else None
            ),
            current_company=(
                current_company.strip()
                if current_company
                else None
            ),
            years_experience=years_experience,
            skills=_normalize_skills(skills or []),
            education=education.strip() if education else None,
            availability=(
                availability.strip()
                if availability
                else None
            ),
            resume_filename=resume_filename.strip(),
            resume_document_id=resume_document_id,
            active=True,
            created_at=now,
            updated_at=now,
        )

        with self.session_factory() as session:
            duplicate = session.scalar(
                select(CandidateModel).where(
                    CandidateModel.knowledge_base_id
                    == knowledge_base_id,
                    CandidateModel.resume_filename
                    == candidate.resume_filename,
                )
            )

            if duplicate is not None:
                raise DuplicateCandidateError(
                    "A candidate already uses this resume filename "
                    "in this knowledge base."
                )

            session.add(candidate)
            session.commit()
            session.refresh(candidate)

            return self._to_domain(candidate)

    def get(
        self,
        candidate_id: str,
    ) -> Candidate | None:

        with self.session_factory() as session:
            candidate = session.get(
                CandidateModel,
                candidate_id,
            )

            if candidate is None:
                return None

            return self._to_domain(candidate)

    def list(
        self,
        *,
        knowledge_base_id: str,
        active_only: bool = True,
    ) -> list[Candidate]:

        with self.session_factory() as session:
            statement = (
                select(CandidateModel)
                .where(
                    CandidateModel.knowledge_base_id
                    == knowledge_base_id
                )
                .order_by(
                    CandidateModel.created_at.desc()
                )
            )

            if active_only:
                statement = statement.where(
                    CandidateModel.active.is_(True)
                )

            candidates = session.scalars(statement).all()

            return [
                self._to_domain(candidate)
                for candidate in candidates
            ]

    def find_matching(
        self,
        *,
        knowledge_base_id: str,
        skills: list[str] | None = None,
        min_years_experience: float | None = None,
        max_years_experience: float | None = None,
        location: str | None = None,
        title: str | None = None,
    ) -> list[Candidate]:

        normalized_skills = {
            skill.strip().lower()
            for skill in (skills or [])
            if skill.strip()
        }

        normalized_location = (
            location.strip().lower()
            if location
            else None
        )

        normalized_title = (
            title.strip().lower()
            if title
            else None
        )

        candidates = self.list(
            knowledge_base_id=knowledge_base_id,
            active_only=True,
        )

        results: list[Candidate] = []

        for candidate in candidates:

            if (
                min_years_experience is not None
                and (
                    candidate.years_experience is None
                    or candidate.years_experience
                    < min_years_experience
                )
            ):
                continue

            if (
                max_years_experience is not None
                and (
                    candidate.years_experience is None
                    or candidate.years_experience
                    > max_years_experience
                )
            ):
                continue

            if normalized_location:
                candidate_location = (
                    candidate.location or ""
                ).lower()

                if normalized_location not in candidate_location:
                    continue

            if normalized_title:
                candidate_title = (
                    candidate.current_title or ""
                ).lower()

                if normalized_title not in candidate_title:
                    continue

            if normalized_skills:
                candidate_skills = {
                    skill.lower()
                    for skill in candidate.skills
                }

                if not normalized_skills.issubset(
                    candidate_skills
                ):
                    continue

            results.append(candidate)

        return results

    def update(
        self,
        candidate_id: str,
        **changes,
    ) -> Candidate:

        allowed_fields = {
            "full_name",
            "email",
            "phone",
            "location",
            "current_title",
            "current_company",
            "years_experience",
            "skills",
            "education",
            "availability",
            "resume_filename",
            "resume_document_id",
            "active",
        }

        unknown_fields = set(changes) - allowed_fields

        if unknown_fields:
            raise ValueError(
                "Unsupported candidate fields: "
                + ", ".join(sorted(unknown_fields))
            )

        with self.session_factory() as session:
            candidate = session.get(
                CandidateModel,
                candidate_id,
            )

            if candidate is None:
                raise CandidateNotFoundError(
                    f"Candidate not found: {candidate_id}"
                )

            if "skills" in changes:
                changes["skills"] = _normalize_skills(
                    changes["skills"] or []
                )

            for field, value in changes.items():
                if isinstance(value, str):
                    value = value.strip()

                setattr(candidate, field, value)

            candidate.updated_at = datetime.now()

            session.commit()
            session.refresh(candidate)

            return self._to_domain(candidate)

    
    def attach_resume_document(
        self,
        *,
        knowledge_base_id: str,
        resume_filename: str,
        resume_document_id: str,
    ) -> int:

        with self.session_factory() as session:
            candidates = session.scalars(
                select(CandidateModel).where(
                    CandidateModel.knowledge_base_id
                    == knowledge_base_id,
                    CandidateModel.resume_filename
                    == resume_filename,
                    CandidateModel.resume_document_id.is_(None),
                )
            ).all()

            for candidate in candidates:
                candidate.resume_document_id = (
                    resume_document_id
                )
                candidate.updated_at = datetime.now()

            session.commit()

            return len(candidates)
        
    
    @staticmethod
    def _to_domain(
        candidate: CandidateModel,
    ) -> Candidate:

        return Candidate(
            candidate_id=candidate.candidate_id,
            knowledge_base_id=candidate.knowledge_base_id,
            full_name=candidate.full_name,
            email=candidate.email,
            phone=candidate.phone,
            location=candidate.location,
            current_title=candidate.current_title,
            current_company=candidate.current_company,
            years_experience=candidate.years_experience,
            skills=list(candidate.skills or []),
            education=candidate.education,
            availability=candidate.availability,
            resume_filename=candidate.resume_filename,
            resume_document_id=candidate.resume_document_id,
            active=candidate.active,
            created_at=candidate.created_at,
            updated_at=candidate.updated_at,
        )


def _normalize_skills(
    skills: list[str],
) -> list[str]:

    normalized = {
        skill.strip().lower()
        for skill in skills
        if skill and skill.strip()
    }

    return sorted(normalized)
