from app.bm25_retriever import (
    BM25Retriever,
    LexicalIndex,
    RebuildableLexicalIndex,
)


def test_bm25_retriever_satisfies_lexical_index_protocol() -> None:
    retriever = BM25Retriever()

    assert isinstance(
        retriever,
        LexicalIndex,
    )


def test_bm25_retriever_satisfies_rebuildable_lexical_index_protocol() -> None:
    retriever = BM25Retriever()

    assert isinstance(
        retriever,
        RebuildableLexicalIndex,
    )
