from app.bm25_retriever import BM25Retriever
from app.chunker import DocumentChunk


def create_chunks() -> list[DocumentChunk]:
    """Create deterministic test chunks."""

    return [
        DocumentChunk(
            doc_id="doc_001",
            text=(
                "Customers can request a refund "
                "within 30 days of purchase."
            ),
            source="policy.pdf",
            page_number=4,
            chunk_index=0,
        ),
        DocumentChunk(
            doc_id="doc_002",
            text=(
                "The office is open from "
                "nine in the morning until five."
            ),
            source="office.pdf",
            page_number=2,
            chunk_index=1,
        ),
        DocumentChunk(
            doc_id="doc_003",
            text=(
                "Product SKU FBH001-RED is "
                "available for purchase."
            ),
            source="catalog.txt",
            page_number=1,
            chunk_index=2,
        ),
    ]


def test_bm25_returns_matching_results() -> None:
    """Verify BM25 retrieves lexically matching chunks."""

    retriever = BM25Retriever(
        create_chunks()
    )

    results = retriever.retrieve(
        "refund",
        top_k=5,
    )

    assert len(results) >= 1

    assert (
        results[0].source
        == "policy.pdf"
    )

    assert "refund" in results[0].text.lower()


def test_bm25_handles_exact_terms() -> None:
    """Verify BM25 works well with exact identifiers."""

    retriever = BM25Retriever(
        create_chunks()
    )

    results = retriever.retrieve(
        "FBH001-RED",
        top_k=5,
    )

    assert len(results) >= 1

    assert (
        results[0].source
        == "catalog.txt"
    )

    assert (
        "FBH001-RED"
        in results[0].text
    )


def test_bm25_preserves_metadata() -> None:
    """Verify retrieved metadata is preserved."""

    retriever = BM25Retriever(
        create_chunks()
    )

    results = retriever.retrieve(
        "refund",
        top_k=1,
    )

    assert len(results) == 1

    result = results[0]

    assert result.source == "policy.pdf"
    assert result.page_number == 4
    assert result.chunk_index == 0


def test_bm25_results_have_scores() -> None:
    """Verify BM25 produces positive relevance scores."""

    retriever = BM25Retriever(
        create_chunks()
    )

    results = retriever.retrieve(
        "refund",
        top_k=1,
    )

    assert len(results) == 1
    assert results[0].score > 0


def test_bm25_results_are_ranked() -> None:
    """Verify results are ordered by decreasing BM25 score."""

    retriever = BM25Retriever(
        create_chunks()
    )

    results = retriever.retrieve(
        "refund purchase",
        top_k=5,
    )

    assert len(results) >= 1

    scores = [
        result.score
        for result in results
    ]

    assert scores == sorted(
        scores,
        reverse=True,
    )


def test_bm25_unmatched_query_returns_empty() -> None:
    """Verify unmatched queries do not return irrelevant chunks."""

    retriever = BM25Retriever(
        create_chunks()
    )

    results = retriever.retrieve(
        "quantum physics",
        top_k=5,
    )

    assert results == []


def test_empty_query_is_rejected() -> None:
    """Verify empty queries are rejected."""

    retriever = BM25Retriever(
        create_chunks()
    )

    try:
        retriever.retrieve("")
        assert False
    except ValueError as exc:
        assert str(exc) == "query cannot be empty"


def test_whitespace_query_is_rejected() -> None:
    """Verify whitespace-only queries are rejected."""

    retriever = BM25Retriever(
        create_chunks()
    )

    try:
        retriever.retrieve("   ")
        assert False
    except ValueError as exc:
        assert str(exc) == "query cannot be empty"


def test_invalid_top_k_is_rejected() -> None:
    """Verify invalid top_k values are rejected."""

    retriever = BM25Retriever(
        create_chunks()
    )

    try:
        retriever.retrieve(
            "refund",
            top_k=0,
        )
        assert False
    except ValueError as exc:
        assert str(exc) == "top_k must be greater than zero"


def test_retrieve_before_build_is_rejected() -> None:
    """Verify retrieval cannot happen before index creation."""

    retriever = BM25Retriever()

    try:
        retriever.retrieve("refund")
        assert False
    except ValueError as exc:
        assert (
            str(exc)
            == "BM25 index has not been built"
        )


def test_empty_chunks_are_rejected() -> None:
    """Verify empty chunk collections are rejected."""

    retriever = BM25Retriever()

    try:
        retriever.build([])

        assert False

    except ValueError as exc:
        assert (
            str(exc)
            == "Cannot build BM25 index from empty chunk list"
        )

def create_chunk(
    doc_id: str,
    text: str,
    chunk_index: int,
) -> DocumentChunk:
    return DocumentChunk(
        doc_id=doc_id,
        text=text,
        source=f"{doc_id}.txt",
        page_number=1,
        chunk_index=chunk_index,
    )


def test_remove_document_removes_only_target_document() -> None:
    chunks = [
        create_chunk(
            doc_id="doc_a",
            text="annual leave policy for employees",
            chunk_index=0,
        ),
        create_chunk(
            doc_id="doc_b",
            text="medical insurance policy for employees",
            chunk_index=0,
        ),
    ]

    retriever = BM25Retriever(chunks)

    retriever.remove_document("doc_a")

    print("CHUNKS:", retriever._chunks)
    print("BM25:", retriever._bm25)

    results = retriever.retrieve("annual leave")

    assert results == []

    results = retriever.retrieve("medical insurance")

    assert len(results) == 1
    assert results[0].source == "doc_b.txt"

def test_remove_last_document_clears_bm25_index() -> None:
    chunks = [
        create_chunk(
            doc_id="doc_a",
            text="annual leave policy",
            chunk_index=0,
        ),
    ]

    retriever = BM25Retriever(chunks)

    retriever.remove_document("doc_a")

    assert retriever._chunks == []
    assert retriever._bm25 is None        