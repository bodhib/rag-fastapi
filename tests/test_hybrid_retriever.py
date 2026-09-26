from app.bm25_retriever import LexicalIndex
from app.hybrid_retriever import HybridResult, HybridRetriever
from app.retrieval import RetrievalResult


class FakeLexicalIndex:
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[RetrievalResult]:
        return [
            RetrievalResult(
                text="lexical result",
                source="test.txt",
                page_number=1,
                chunk_index=0,
                score=0.8,
            )
        ]


class FakeDenseRetriever:
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[RetrievalResult]:
        return [
            RetrievalResult(
                text="dense result",
                source="test.txt",
                page_number=1,
                chunk_index=1,
                score=0.9,
            )
        ]


def test_fake_lexical_index_satisfies_protocol() -> None:
    lexical_index = FakeLexicalIndex()

    assert isinstance(
        lexical_index,
        LexicalIndex,
    )


def test_hybrid_retriever_accepts_lexical_index() -> None:
    lexical_index = FakeLexicalIndex()

    retriever = HybridRetriever(
        dense_retriever=FakeDenseRetriever(),
        lexical_retriever=lexical_index,
    )

    results = retriever.retrieve(
        "test query",
        top_k=5,
    )

    assert len(results) == 2

    assert isinstance(
        results[0],
        HybridResult,
    )

    assert results[0].text == "dense result"
    assert results[1].text == "lexical result"


def test_hybrid_retriever_preserves_dense_score() -> None:
    retriever = HybridRetriever(
        dense_retriever=FakeDenseRetriever(),
        lexical_retriever=FakeLexicalIndex(),
    )

    results = retriever.retrieve(
        "test query",
        top_k=5,
    )

    dense_result = results[0]

    assert dense_result.dense_score == 0.9
    assert dense_result.bm25_score == 0.0
    assert dense_result.score == 0.9


def test_hybrid_retriever_preserves_lexical_score() -> None:
    retriever = HybridRetriever(
        dense_retriever=FakeDenseRetriever(),
        lexical_retriever=FakeLexicalIndex(),
    )

    results = retriever.retrieve(
        "test query",
        top_k=5,
    )

    lexical_result = results[1]

    assert lexical_result.dense_score == 0.0
    assert lexical_result.bm25_score == 0.8
    assert lexical_result.score == 0.8


def test_hybrid_retriever_accepts_any_lexical_index() -> None:
    lexical_index = FakeLexicalIndex()

    retriever = HybridRetriever(
        dense_retriever=FakeDenseRetriever(),
        lexical_retriever=lexical_index,
    )

    results = retriever.retrieve(
        "annual leave",
        top_k=5,
    )

    assert results
