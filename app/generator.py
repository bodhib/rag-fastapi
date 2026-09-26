"""
Gemini-based answer generation.

This module generates final answers from retrieved document
contexts using Google's Gemini model.
"""

from dataclasses import dataclass

from google import genai

from app.config import Settings, settings
from app.reranker import RerankedResult


@dataclass
class GenerationResult:
    """
    Represent the result of LLM answer generation.
    """

    answer: str
    contexts: list[RerankedResult]


class GeminiGenerator:
    """
    Generate answers using Google's Gemini model.
    """

    DEFAULT_MODEL = "gemini-2.5-flash-lite"

    def __init__(
        self,
        api_key: str | None = None,
        model_name: str | None = None,
        config: Settings | None = None,
        client=None,
    ) -> None:

        self.config = config or settings

        resolved_api_key = (
            api_key
            if api_key is not None
            else self.config.gemini_api_key
        )

        resolved_model_name = (
            model_name
            if model_name is not None
            else self.config.gemini_model
        )

        if not resolved_api_key.strip():
            raise ValueError(
                "Gemini API key is not configured"
            )

        if not resolved_model_name.strip():
            raise ValueError(
                "Gemini model name cannot be empty"
            )

        self.model_name = resolved_model_name

        self._client = client or genai.Client(
            api_key=resolved_api_key
        )

    def generate(
        self,
        query: str,
        contexts: list[RerankedResult],
    ) -> GenerationResult:

        if not query.strip():
            raise ValueError(
                "query cannot be empty"
            )

        if not contexts:
            raise ValueError(
                "contexts cannot be empty"
            )

        prompt = self._build_prompt(
            query=query,
            contexts=contexts,
        )

        response = self._client.models.generate_content(
            model=self.model_name,
            contents=prompt,
        )

        answer = response.text

        if not answer or not answer.strip():
            raise ValueError(
                "Gemini returned an empty response"
            )

        return GenerationResult(
            answer=answer.strip(),
            contexts=contexts,
        )

    @staticmethod
    def _build_prompt(
        query: str,
        contexts: list[RerankedResult],
    ) -> str:

        context_sections = []

        for index, context in enumerate(
            contexts,
            start=1,
        ):
            context_sections.append(
                (
                    f"[{index}] "
                    f"Source: {context.source}, "
                    f"Page: {context.page_number}, "
                    f"Chunk: {context.chunk_index}\n"
                    f"{context.text}"
                )
            )

        context_text = "\n\n".join(
            context_sections
        )

        return f"""
You are a helpful question-answering assistant.

Answer the user's question using only the provided
document context.

If the answer is supported by the context, cite the
relevant context using citation markers such as [1],
[2], etc.

If the answer cannot be determined from the provided
context, clearly say that the information is not
available in the provided documents.

User question:
{query}

Document context:
{context_text}
""".strip()