from app.hybrid_retriever import HybridResult
from app.reranker import CrossEncoderReranker

import pytest

from app.config import Settings


class FakeCrossEncoder:
    def predict(
        self,
        pairs: list[list[str]],
    ) -> list[float]:
        assert len(pairs) == 3

        assert pairs[0] == [
            "How long do I have to get a refund?",
            "Refunds are available within 30 days.",
        ]

        return [
            2.0,
            8.0,
            5.0,
        ]


def create_candidates() -> list[HybridResult]:
    return [
        HybridResult(
            text="Refunds are available within 30 days.",
            source="policy.pdf",
            page_number=4,
            chunk_index=0,
            dense_score=0.9,
            bm25_score=0.4,
            score=0.65,
        ),
        HybridResult(
            text="Office hours are from 9 AM to 5 PM.",
            source="office.pdf",
            page_number=2,
            chunk_index=1,
            dense_score=0.3,
            bm25_score=0.2,
            score=0.25,
        ),
        HybridResult(
            text="Refund requests require an order number.",
            source="policy.pdf",
            page_number=4,
            chunk_index=1,
            dense_score=0.7,
            bm25_score=0.8,
            score=0.75,
        ),
    ]


def test_reranks_candidates() -> None:
    reranker = object.__new__(
        CrossEncoderReranker
    )

    reranker._model = FakeCrossEncoder()

    results = reranker.rerank(
        query="How long do I have to get a refund?",
        candidates=create_candidates(),
        top_k=3,
    )

    assert len(results) == 3

    assert results[0].rerank_score == 8.0
    assert results[1].rerank_score == 5.0
    assert results[2].rerank_score == 2.0


def test_reranking_preserves_metadata() -> None:
    reranker = object.__new__(
        CrossEncoderReranker
    )

    reranker._model = FakeCrossEncoder()

    results = reranker.rerank(
        query="How long do I have to get a refund?",
        candidates=create_candidates(),
    )

    assert results[0].source == "office.pdf"
    assert results[0].page_number == 2
    assert results[0].chunk_index == 1


def test_reranking_preserves_original_scores() -> None:
    reranker = object.__new__(
        CrossEncoderReranker
    )

    reranker._model = FakeCrossEncoder()

    results = reranker.rerank(
        query="How long do I have to get a refund?",
        candidates=create_candidates(),
    )

    assert results[0].dense_score == 0.3
    assert results[0].bm25_score == 0.2
    assert results[0].hybrid_score == 0.25


def test_top_k_limits_results() -> None:
    reranker = object.__new__(
        CrossEncoderReranker
    )

    reranker._model = FakeCrossEncoder()

    results = reranker.rerank(
        query="How long do I have to get a refund?",
        candidates=create_candidates(),
        top_k=2,
    )

    assert len(results) == 2
    assert results[0].rerank_score == 8.0
    assert results[1].rerank_score == 5.0


def test_empty_candidates_returns_empty_list() -> None:
    reranker = object.__new__(
        CrossEncoderReranker
    )

    reranker._model = FakeCrossEncoder()

    results = reranker.rerank(
        query="How long do I have to get a refund?",
        candidates=[],
    )

    assert results == []


def test_empty_query_is_rejected() -> None:
    reranker = object.__new__(
        CrossEncoderReranker
    )

    reranker._model = FakeCrossEncoder()

    try:
        reranker.rerank(
            query="",
            candidates=create_candidates(),
        )
    except ValueError as exc:
        assert str(exc) == "query cannot be empty"
    else:
        raise AssertionError(
            "Expected ValueError"
        )


def test_whitespace_query_is_rejected() -> None:
    reranker = object.__new__(
        CrossEncoderReranker
    )

    reranker._model = FakeCrossEncoder()

    try:
        reranker.rerank(
            query="   ",
            candidates=create_candidates(),
        )
    except ValueError as exc:
        assert str(exc) == "query cannot be empty"
    else:
        raise AssertionError(
            "Expected ValueError"
        )


def test_invalid_top_k_is_rejected() -> None:
    reranker = object.__new__(
        CrossEncoderReranker
    )

    reranker._model = FakeCrossEncoder()

    try:
        reranker.rerank(
            query="refund",
            candidates=create_candidates(),
            top_k=0,
        )
    except ValueError as exc:
        assert str(exc) == (
            "top_k must be greater than zero"
        )
    else:
        raise AssertionError(
            "Expected ValueError"
        )
    
@pytest.mark.integration
def test_real_cross_encoder_reranking() -> None:
    reranker = CrossEncoderReranker()

    candidates = [
        HybridResult(
            text="Customers can request a refund within 30 days of purchase.",
            source="policy.pdf",
            page_number=4,
            chunk_index=0,
            dense_score=0.8,
            bm25_score=0.7,
            score=0.75,
        ),
        HybridResult(
            text="The office is open from nine in the morning until five.",
            source="office.pdf",
            page_number=2,
            chunk_index=1,
            dense_score=0.5,
            bm25_score=0.1,
            score=0.3,
        ),
    ]

    results = reranker.rerank(
        query="How long do I have to get a refund?",
        candidates=candidates,
        top_k=2,
    )

    assert len(results) == 2

    assert results[0].source == "policy.pdf"

    assert (
        "refund"
        in results[0].text.lower()
    )

    assert (
        results[0].rerank_score
        > results[1].rerank_score
    )

def test_reranker_uses_settings(monkeypatch):
    config = Settings(
        reranker_model="test-reranker-model"
    )

    fake_model = FakeCrossEncoder()

    monkeypatch.setattr(
        "app.reranker.CrossEncoder",
        lambda model_name: fake_model,
    )

    reranker = CrossEncoderReranker(
        config=config
    )

    assert reranker.model_name == (
        "test-reranker-model"
    )

def test_reranker_rejects_empty_configured_model():
    config = Settings(
        reranker_model=""
    )

    with pytest.raises(
        ValueError,
        match="Reranker model name cannot be empty",
    ):
        CrossEncoderReranker(
            config=config
        )            