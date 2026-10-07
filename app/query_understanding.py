from __future__ import annotations

import json
from dataclasses import dataclass

from google import genai
from google.genai import types

from app.config import Settings


@dataclass(frozen=True)
class RecruiterQueryIntent:
    semantic_query: str
    skills: list[str]
    min_years_experience: float | None
    max_years_experience: float | None
    location: str | None
    title: str | None


class RecruiterQueryUnderstandingError(ValueError):
    pass


class RecruiterQueryUnderstandingService:
    """
    Converts a natural-language recruiter query into:

    1. Structured candidate filters
    2. A semantic retrieval query

    Gemini is used only for query interpretation.
    Candidate eligibility remains deterministic.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

        self.client = genai.Client(
            api_key=settings.google_api_key,
        )

    def understand(self, query: str) -> RecruiterQueryIntent:
        query = query.strip()

        if not query:
            raise RecruiterQueryUnderstandingError(
                "Recruiter query cannot be empty."
            )

        prompt = self._build_prompt(query)

        try:
            response = self.client.models.generate_content(
                model=self.settings.gemini_model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0,
                    response_mime_type="application/json",
                ),
            )
        except Exception as exc:
            raise RecruiterQueryUnderstandingError(
                "Failed to understand recruiter query."
            ) from exc

        raw_content = response.text

        if not raw_content:
            raise RecruiterQueryUnderstandingError(
                "Gemini returned an empty response."
            )

        data = self._parse_json(raw_content)

        return self._validate_and_normalize(data)

    @staticmethod
    def _build_prompt(query: str) -> str:
        return f"""
You are a recruitment search query parser.

Convert the recruiter's natural-language request into JSON.

Return exactly these fields:

{{
  "semantic_query": "string",
  "skills": ["string"],
  "min_years_experience": number or null,
  "max_years_experience": number or null,
  "location": "string or null",
  "title": "string or null"
}}

Rules:

1. semantic_query:
   - Preserve the core meaning of the recruitment request.
   - Include important role and technical requirements.
   - Do not invent requirements.

2. skills:
   - Extract explicitly requested technical/professional skills.
   - Normalize obvious capitalization differences.
   - Do not invent skills.

3. min_years_experience:
   - "5+ years" means 5.
   - "at least 5 years" means 5.
   - "minimum 5 years" means 5.
   - Otherwise null.

4. max_years_experience:
   - "up to 5 years" means 5.
   - "maximum 5 years" means 5.
   - Otherwise null.

5. location:
   - Extract an explicitly requested city, region, or country.
   - Otherwise null.

6. title:
   - Extract an explicitly requested job title or role.
   - Examples:
     Backend Engineer
     Data Scientist
     DevOps Engineer
   - Otherwise null.

7. Never invent candidate information.

8. Return only valid JSON.
   Do not return markdown.
   Do not add explanations.

Recruiter query:

{query}
""".strip()

    @staticmethod
    def _parse_json(raw_content: str) -> dict:
        cleaned = raw_content.strip()

        if cleaned.startswith("```json"):
            cleaned = cleaned[len("```json"):].strip()

        elif cleaned.startswith("```"):
            cleaned = cleaned[len("```"):].strip()

        if cleaned.endswith("```"):
            cleaned = cleaned[:-3].strip()

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            raise RecruiterQueryUnderstandingError(
                "Gemini returned invalid JSON."
            ) from exc

        if not isinstance(data, dict):
            raise RecruiterQueryUnderstandingError(
                "Gemini response must be a JSON object."
            )

        return data

    @staticmethod
    def _validate_and_normalize(
        data: dict,
    ) -> RecruiterQueryIntent:
        semantic_query = data.get("semantic_query")

        if not isinstance(semantic_query, str):
            raise RecruiterQueryUnderstandingError(
                "semantic_query must be a string."
            )

        semantic_query = semantic_query.strip()

        if not semantic_query:
            raise RecruiterQueryUnderstandingError(
                "semantic_query cannot be empty."
            )

        raw_skills = data.get("skills", [])

        if not isinstance(raw_skills, list):
            raise RecruiterQueryUnderstandingError(
                "skills must be a list."
            )

        skills: list[str] = []

        for skill in raw_skills:
            if not isinstance(skill, str):
                continue

            normalized = skill.strip()

            if normalized and normalized not in skills:
                skills.append(normalized)

        min_years = RecruiterQueryUnderstandingService._normalize_number(
            data.get("min_years_experience"),
            "min_years_experience",
        )

        max_years = RecruiterQueryUnderstandingService._normalize_number(
            data.get("max_years_experience"),
            "max_years_experience",
        )

        if (
            min_years is not None
            and max_years is not None
            and min_years > max_years
        ):
            raise RecruiterQueryUnderstandingError(
                "Minimum experience cannot exceed maximum experience."
            )

        location = RecruiterQueryUnderstandingService._normalize_string(
            data.get("location")
        )

        title = RecruiterQueryUnderstandingService._normalize_string(
            data.get("title")
        )

        return RecruiterQueryIntent(
            semantic_query=semantic_query,
            skills=skills,
            min_years_experience=min_years,
            max_years_experience=max_years,
            location=location,
            title=title,
        )

    @staticmethod
    def _normalize_number(
        value: object,
        field_name: str,
    ) -> float | None:
        if value is None:
            return None

        if isinstance(value, bool):
            raise RecruiterQueryUnderstandingError(
                f"{field_name} must be a number or null."
            )

        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise RecruiterQueryUnderstandingError(
                f"{field_name} must be a number or null."
            ) from exc

        if number < 0 or number > 60:
            raise RecruiterQueryUnderstandingError(
                f"{field_name} must be between 0 and 60."
            )

        return number

    @staticmethod
    def _normalize_string(value: object) -> str | None:
        if value is None:
            return None

        if not isinstance(value, str):
            raise RecruiterQueryUnderstandingError(
                "Expected a string or null."
            )

        value = value.strip()

        return value or None