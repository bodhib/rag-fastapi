from app.bm25_retriever import BM25Retriever
from app.chunker import DocumentChunk


def create_chunk(
    doc_id: str,
    text: str,
    chunk_index: int,
    knowledge_base_id: str = "kb_default",
) -> DocumentChunk:
    return DocumentChunk(
        doc_id=doc_id,
        text=text,
        source=f"{doc_id}.txt",
        page_number=1,
        chunk_index=chunk_index,
        knowledge_base_id=knowledge_base_id,
    )


def create_chunks() -> list[DocumentChunk]:
    return [
        create_chunk(
            doc_id="doc_001",
            text=(
                "Customers can request a refund "
                "within 30 days of purchase."
            ),
            chunk_index=0,
        ),
        create_chunk(
            doc_id="doc_002",
            text=(
                "The office is open from "
                "nine in the morning until five."
            ),
            chunk_index=1,
        ),
        create_chunk(
            doc_id="doc_003",
            text=(
                "Product SKU FBH001-RED is "
                "available for purchase."
            ),
            chunk_index=2,
        ),
    ]


def test_bm25_returns_matching_results() -> None:
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
        == "doc_001.txt"
    )

    assert "refund" in results[0].text.lower()


def test_bm25_handles_exact_terms() -> None:
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
        == "doc_003.txt"
    )

    assert (
        "FBH001-RED"
        in results[0].text
    )


def test_bm25_preserves_metadata() -> None:
    retriever = BM25Retriever(
        create_chunks()
    )

    results = retriever.retrieve(
        "refund",
        top_k=1,
    )

    assert len(results) == 1

    result = results[0]

    assert result.source == "doc_001.txt"
    assert result.page_number == 1
    assert result.chunk_index == 0


def test_bm25_results_have_scores() -> None:
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
    retriever = BM25Retriever(
        create_chunks()
    )

    results = retriever.retrieve(
        "quantum physics",
        top_k=5,
    )

    assert results == []


def test_bm25_filters_results_by_knowledge_base() -> None:
    chunks = [
        create_chunk(
            doc_id="doc_default",
            text="Annual leave policy for default knowledge base.",
            chunk_index=0,
            knowledge_base_id="kb_default",
        ),
        create_chunk(
            doc_id="doc_company",
            text="Annual leave policy for company knowledge base.",
            chunk_index=0,
            knowledge_base_id="kb_company",
        ),
    ]

    retriever = BM25Retriever(chunks)

    results = retriever.retrieve(
        "annual leave",
        top_k=5,
        knowledge_base_id="kb_company",
    )

    assert len(results) == 1
    assert results[0].source == "doc_company.txt"


def test_bm25_filtering_still_returns_top_k_matching_kb() -> None:
    chunks = [
        create_chunk(
            doc_id="doc_default_1",
            text="Annual leave policy default one.",
            chunk_index=0,
            knowledge_base_id="kb_default",
        ),
        create_chunk(
            doc_id="doc_default_2",
            text="Annual leave policy default two.",
            chunk_index=1,
            knowledge_base_id="kb_default",
        ),
        create_chunk(
            doc_id="doc_company_1",
            text="Annual leave company one.",
            chunk_index=0,
            knowledge_base_id="kb_company",
        ),
        create_chunk(
            doc_id="doc_company_2",
            text="Annual leave company two.",
            chunk_index=1,
            knowledge_base_id="kb_company",
        ),
    ]

    retriever = BM25Retriever(chunks)

    results = retriever.retrieve(
        "annual leave",
        top_k=2,
        knowledge_base_id="kb_company",
    )

    assert len(results) == 2

    assert all(
        result.source.startswith("doc_company")
        for result in results
    )


def test_empty_knowledge_base_id_is_rejected() -> None:
    retriever = BM25Retriever(
        create_chunks()
    )

    try:
        retriever.retrieve(
            "refund",
            knowledge_base_id="   ",
        )
        assert False
    except ValueError as exc:
        assert (
            str(exc)
            == "knowledge_base_id cannot be empty"
        )


def test_empty_query_is_rejected() -> None:
    retriever = BM25Retriever(
        create_chunks()
    )

    try:
        retriever.retrieve("")
        assert False
    except ValueError as exc:
        assert str(exc) == "query cannot be empty"


def test_whitespace_query_is_rejected() -> None:
    retriever = BM25Retriever(
        create_chunks()
    )

    try:
        retriever.retrieve("   ")
        assert False
    except ValueError as exc:
        assert str(exc) == "query cannot be empty"


def test_invalid_top_k_is_rejected() -> None:
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
    retriever = BM25Retriever()

    try:
        retriever.build([])

        assert False

    except ValueError as exc:
        assert (
            str(exc)
            == "Cannot build BM25 index from empty chunk list"
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