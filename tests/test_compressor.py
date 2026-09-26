from app.compressor import ContextCompressor
from app.reranker import RerankedResult


def create_results() -> list[RerankedResult]:
    return [
        RerankedResult(
            text="Refunds are available within 30 days.",
            source="policy.pdf",
            page_number=4,
            chunk_index=0,
            dense_score=0.9,
            bm25_score=0.8,
            hybrid_score=0.85,
            rerank_score=9.2,
        ),
        RerankedResult(
            text="Refund requests require an order number.",
            source="policy.pdf",
            page_number=4,
            chunk_index=1,
            dense_score=0.8,
            bm25_score=0.9,
            hybrid_score=0.85,
            rerank_score=8.4,
        ),
        RerankedResult(
            text="Refunds are processed within five business days.",
            source="policy.pdf",
            page_number=5,
            chunk_index=2,
            dense_score=0.7,
            bm25_score=0.6,
            hybrid_score=0.65,
            rerank_score=7.1,
        ),
        RerankedResult(
            text="Customers can exchange damaged products.",
            source="returns.pdf",
            page_number=2,
            chunk_index=3,
            dense_score=0.5,
            bm25_score=0.4,
            hybrid_score=0.45,
            rerank_score=3.2,
        ),
        RerankedResult(
            text="The office is open from 9 AM to 5 PM.",
            source="office.pdf",
            page_number=2,
            chunk_index=4,
            dense_score=0.2,
            bm25_score=0.1,
            hybrid_score=0.15,
            rerank_score=1.4,
        ),
    ]


def test_compressor_returns_top_three() -> None:
    compressor = ContextCompressor()

    results = compressor.compress(
        create_results(),
        top_k=3,
    )

    assert len(results) == 3

    assert results[0].rerank_score == 9.2
    assert results[1].rerank_score == 8.4
    assert results[2].rerank_score == 7.1


def test_compressor_preserves_original_order() -> None:
    compressor = ContextCompressor()

    results = compressor.compress(
        create_results(),
        top_k=3,
    )

    assert [
        result.chunk_index
        for result in results
    ] == [0, 1, 2]


def test_compressor_preserves_text() -> None:
    compressor = ContextCompressor()

    results = compressor.compress(
        create_results(),
        top_k=2,
    )

    assert results[0].text == (
        "Refunds are available within 30 days."
    )

    assert results[1].text == (
        "Refund requests require an order number."
    )


def test_compressor_preserves_metadata() -> None:
    compressor = ContextCompressor()

    results = compressor.compress(
        create_results(),
        top_k=1,
    )

    assert results[0].source == "policy.pdf"
    assert results[0].page_number == 4
    assert results[0].chunk_index == 0


def test_top_k_can_be_smaller_than_three() -> None:
    compressor = ContextCompressor()

    results = compressor.compress(
        create_results(),
        top_k=2,
    )

    assert len(results) == 2


def test_top_k_larger_than_results_returns_all_results() -> None:
    compressor = ContextCompressor()

    results = compressor.compress(
        create_results(),
        top_k=10,
    )

    assert len(results) == 5


def test_empty_results_return_empty_list() -> None:
    compressor = ContextCompressor()

    results = compressor.compress(
        [],
        top_k=3,
    )

    assert results == []


def test_invalid_top_k_is_rejected() -> None:
    compressor = ContextCompressor()

    try:
        compressor.compress(
            create_results(),
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


def test_negative_top_k_is_rejected() -> None:
    compressor = ContextCompressor()

    try:
        compressor.compress(
            create_results(),
            top_k=-1,
        )
    except ValueError as exc:
        assert str(exc) == (
            "top_k must be greater than zero"
        )
    else:
        raise AssertionError(
            "Expected ValueError"
        )