import numpy as np
import pytest

from app.chunker import DocumentChunk
from app.embeddings import EmbeddingService

from app.config import Settings


class FakeEmbeddingModel:
    """Small fake model used for unit testing."""

    def get_sentence_embedding_dimension(self) -> int:
        """Return the fake embedding dimension."""

        return 3

    def encode(
        self,
        texts,
        normalize_embeddings=True,
        show_progress_bar=False,
    ):
        """Return deterministic fake embeddings."""

        if isinstance(texts, str):
            return np.array(
                [0.1, 0.2, 0.3],
                dtype=np.float32,
            )

        return np.array(
            [
                [0.1, 0.2, 0.3],
                [0.4, 0.5, 0.6],
            ],
            dtype=np.float32,
        )


@pytest.fixture
def embedding_service() -> EmbeddingService:
    """Create an embedding service with a fake model."""

    service = object.__new__(EmbeddingService)

    service.model_name = "fake-model"
    service._model = FakeEmbeddingModel()

    return service


def test_embedding_dimension(
    embedding_service: EmbeddingService,
) -> None:
    """Verify embedding dimension is exposed."""

    assert embedding_service.dimension == 3


def test_embed_text(
    embedding_service: EmbeddingService,
) -> None:
    """Verify a single text can be embedded."""

    result = embedding_service.embed_text(
        "This is a test."
    )

    assert result == [
        pytest.approx(0.1),
        pytest.approx(0.2),
        pytest.approx(0.3),
    ]


def test_empty_text_is_rejected(
    embedding_service: EmbeddingService,
) -> None:
    """Verify empty text cannot be embedded."""

    with pytest.raises(
        ValueError,
        match="empty text",
    ):
        embedding_service.embed_text("")


def test_embed_chunks(
    embedding_service: EmbeddingService,
) -> None:
    """Verify multiple chunks can be embedded."""

    chunks = [
        DocumentChunk(
            doc_id="doc_001",
            text="First chunk.",
            source="sample.pdf",
            page_number=1,
            chunk_index=0,
        ),
        DocumentChunk(
            doc_id="doc_001",
            text="Second chunk.",
            source="sample.pdf",
            page_number=2,
            chunk_index=1,
        ),
    ]

    result = embedding_service.embed_chunks(
        chunks
    )

    assert len(result) == 2

    assert result[0].chunk == chunks[0]
    assert result[1].chunk == chunks[1]

    assert result[0].embedding == [
        pytest.approx(0.1),
        pytest.approx(0.2),
        pytest.approx(0.3),
    ]

    assert result[1].embedding == [
        pytest.approx(0.4),
        pytest.approx(0.5),
        pytest.approx(0.6),
    ]


def test_empty_chunks_are_rejected(
    embedding_service: EmbeddingService,
) -> None:
    """Verify empty chunk lists are rejected."""

    with pytest.raises(
        ValueError,
        match="empty chunk list",
    ):
        embedding_service.embed_chunks([])

@pytest.mark.integration
def test_real_embedding_model() -> None:
    """Verify the real SentenceTransformer model works."""

    service = EmbeddingService()

    result = service.embed_text(
        "This is a real embedding test."
    )

    assert len(result) == 384

    assert all(
        isinstance(value, float)
        for value in result
    )        

def test_embedding_service_uses_settings(monkeypatch):
    settings = Settings(
        embedding_model="test-embedding-model"
    )

    fake_model = FakeEmbeddingModel()

    monkeypatch.setattr(
        "app.embeddings.SentenceTransformer",
        lambda model_name: fake_model,
    )

    service = EmbeddingService(
        config=settings
    )

    assert service.model_name == (
        "test-embedding-model"
    )

def test_embedding_service_rejects_empty_configured_model():
    config = Settings(
        embedding_model=""
    )

    with pytest.raises(
        ValueError,
        match="Embedding model name cannot be empty",
    ):
        EmbeddingService(
            config=config
        )        