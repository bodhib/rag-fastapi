import pytest

from app.generator import GeminiGenerator
from app.reranker import RerankedResult
from app.config import Settings


class FakeResponse:
    text = (
        "You have 30 days to request a refund. [1]"
    )


class FakeModels:
    def __init__(self) -> None:
        self.last_model = None
        self.last_contents = None

    def generate_content(
        self,
        model: str,
        contents: str,
    ) -> FakeResponse:
        self.last_model = model
        self.last_contents = contents

        return FakeResponse()


class FakeGeminiClient:
    def __init__(self) -> None:
        self.models = FakeModels()


def create_contexts() -> list[RerankedResult]:
    return [
        RerankedResult(
            text="Refunds are available within 30 days of purchase.",
            source="policy.pdf",
            page_number=4,
            chunk_index=12,
            dense_score=0.9,
            bm25_score=0.8,
            hybrid_score=0.85,
            rerank_score=9.2,
        ),
        RerankedResult(
            text="Refund requests require the original order number.",
            source="policy.pdf",
            page_number=4,
            chunk_index=13,
            dense_score=0.8,
            bm25_score=0.9,
            hybrid_score=0.85,
            rerank_score=8.4,
        ),
        RerankedResult(
            text="Approved refunds are normally processed within five business days.",
            source="policy.pdf",
            page_number=5,
            chunk_index=14,
            dense_score=0.7,
            bm25_score=0.6,
            hybrid_score=0.65,
            rerank_score=7.1,
        ),
    ]


def test_generate_returns_answer() -> None:
    client = FakeGeminiClient()

    generator = GeminiGenerator(
        api_key="test-key",
        client=client,
    )

    result = generator.generate(
        query="How long do I have to request a refund?",
        contexts=create_contexts(),
    )

    assert result.answer == (
        "You have 30 days to request a refund. [1]"
    )


def test_generator_sends_correct_model() -> None:
    client = FakeGeminiClient()

    generator = GeminiGenerator(
        api_key="test-key",
        client=client,
    )

    generator.generate(
        query="How long do I have to request a refund?",
        contexts=create_contexts(),
    )

    assert (
        client.models.last_model
        == "gemini-2.5-flash-lite"
    )


def test_generator_prompt_contains_query() -> None:
    client = FakeGeminiClient()

    generator = GeminiGenerator(
        api_key="test-key",
        client=client,
    )

    generator.generate(
        query="How long do I have to request a refund?",
        contexts=create_contexts(),
    )

    prompt = client.models.last_contents

    assert (
        "How long do I have to request a refund?"
        in prompt
    )


def test_generator_prompt_contains_context() -> None:
    client = FakeGeminiClient()

    generator = GeminiGenerator(
        api_key="test-key",
        client=client,
    )

    generator.generate(
        query="How long do I have to request a refund?",
        contexts=create_contexts(),
    )

    prompt = client.models.last_contents

    assert (
        "Refunds are available within 30 days of purchase."
        in prompt
    )

    assert (
        "Refund requests require the original order number."
        in prompt
    )


def test_generator_prompt_contains_citation_metadata() -> None:
    client = FakeGeminiClient()

    generator = GeminiGenerator(
        api_key="test-key",
        client=client,
    )

    generator.generate(
        query="How long do I have to request a refund?",
        contexts=create_contexts(),
    )

    prompt = client.models.last_contents

    assert "[1]" in prompt
    assert "[2]" in prompt
    assert "[3]" in prompt

    assert "Source: policy.pdf" in prompt
    assert "Page: 4" in prompt
    assert "Page: 5" in prompt


def test_generator_preserves_contexts() -> None:
    client = FakeGeminiClient()

    contexts = create_contexts()

    generator = GeminiGenerator(
        api_key="test-key",
        client=client,
    )

    result = generator.generate(
        query="refund",
        contexts=contexts,
    )

    assert result.contexts == contexts


def test_empty_query_is_rejected() -> None:
    client = FakeGeminiClient()

    generator = GeminiGenerator(
        api_key="test-key",
        client=client,
    )

    try:
        generator.generate(
            query="",
            contexts=create_contexts(),
        )
    except ValueError as exc:
        assert str(exc) == "query cannot be empty"
    else:
        raise AssertionError(
            "Expected ValueError"
        )


def test_empty_context_is_rejected() -> None:
    client = FakeGeminiClient()

    generator = GeminiGenerator(
        api_key="test-key",
        client=client,
    )

    try:
        generator.generate(
            query="refund",
            contexts=[],
        )
    except ValueError as exc:
        assert str(exc) == "contexts cannot be empty"
    else:
        raise AssertionError(
            "Expected ValueError"
        )


def test_empty_api_key_is_rejected() -> None:
    try:
        GeminiGenerator(
            api_key="",
        )
    except ValueError as exc:
        assert str(exc) == (
            "Gemini API key is not configured"
        )
    else:
        raise AssertionError(
            "Expected ValueError"
        )


def test_empty_model_name_is_rejected() -> None:
    try:
        GeminiGenerator(
            api_key="test-key",
            model_name="",
        )
    except ValueError as exc:
        assert str(exc) == (
            "Gemini model name cannot be empty"
        )
    else:
        raise AssertionError(
            "Expected ValueError"
        )
    
@pytest.mark.integration
def test_real_gemini_generation() -> None:
    generator = GeminiGenerator(api_key="AQ.Ab8RN6JjBZrHGcVgHfE1c6WR013WZ_A7870_7gNCBwuWba9HRw")

    result = generator.generate(
        query="How long do I have to request a refund?",
        contexts=create_contexts(),
    )

    assert result.answer
    assert len(result.answer.strip()) > 0

def test_generator_uses_settings_configuration():
    settings = Settings(
        gemini_api_key="settings-key",
        gemini_model="settings-model",
    )

    generator = GeminiGenerator(
        config=settings,
        client=object(),
    )

    assert generator.model_name == "settings-model"      

def test_generator_rejects_missing_api_key():
    settings = Settings(
        gemini_api_key="",
        gemini_model="settings-model",
    )

    with pytest.raises(
        ValueError,
        match="Gemini API key is not configured",
    ):
        GeminiGenerator(
            config=settings,
            client=object(),
        )

def test_generator_does_not_expose_api_key_in_repr():
    from app.config import Settings
    from app.generator import GeminiGenerator

    secret = "super-secret-gemini-key"

    config = Settings(
        gemini_api_key=secret,
        gemini_model="test-model",
    )

    generator = GeminiGenerator(
        config=config,
        client=object(),
    )

    assert secret not in repr(generator)              

def test_settings_reads_gemini_key_from_environment(monkeypatch):
    from app.config import Settings

    monkeypatch.setenv(
        "GEMINI_API_KEY",
        "environment-test-key",
    )

    config = Settings()

    assert config.gemini_api_key == "environment-test-key"    