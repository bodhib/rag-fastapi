from app.bm25_retriever import BM25Retriever
from app.chunker import DocumentChunk
from app.retrieval_index import RetrievalIndexManager


def create_chunks() -> list[DocumentChunk]:
    return [
        DocumentChunk(
            doc_id="doc-1",
            text=(
                "The annual leave policy allows employees "
                "to take five days of annual leave."
            ),
            source="policy.txt",
            page_number=1,
            chunk_index=0,
        ),
        DocumentChunk(
            doc_id="doc-1",
            text=(
                "Employees must request sick leave "
                "through their manager."
            ),
            source="policy.txt",
            page_number=1,
            chunk_index=1,
        ),
        DocumentChunk(
            doc_id="doc-1",
            text=(
                "Parental leave is available to eligible "
                "employees after qualifying events."
            ),
            source="policy.txt",
            page_number=2,
            chunk_index=2,
        ),
    ]


def test_retrieval_index_manager_rebuilds_lexical_index() -> None:
    chunks = create_chunks()

    lexical_index = BM25Retriever()

    manager = RetrievalIndexManager(
        lexical_index=lexical_index,
    )

    count = manager.rebuild(chunks)

    assert count == len(chunks)

    results = lexical_index.retrieve(
        query="annual leave",
        top_k=5,
    )

    assert results
    assert results[0].source == "policy.txt"
    assert results[0].chunk_index == 0


def test_retrieval_index_manager_returns_zero_for_empty_chunks() -> None:
    lexical_index = BM25Retriever()

    manager = RetrievalIndexManager(
        lexical_index=lexical_index,
    )

    count = manager.rebuild([])

    assert count == 0

