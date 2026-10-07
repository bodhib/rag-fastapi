# tests/test_reranker.py

from app.hybrid_retriever import HybridResult
from app.reranker import RerankedResult


def test_reranked_result_preserves_doc_id():

    hybrid_result = HybridResult(
        text="Alice has extensive FastAPI experience.",
        source="alice_resume.txt.txt",
        page_number=1,
        chunk_index=0,
        dense_score=0.90,
        bm25_score=0.80,
        score=0.85,
        doc_id="doc_alice_123",
    )

    reranked_result = RerankedResult(
        text=hybrid_result.text,
        source=hybrid_result.source,
        page_number=hybrid_result.page_number,
        chunk_index=hybrid_result.chunk_index,
        dense_score=hybrid_result.dense_score,
        bm25_score=hybrid_result.bm25_score,
        hybrid_score=hybrid_result.score,
        rerank_score=7.5,
        doc_id=hybrid_result.doc_id,
    )

    assert reranked_result.doc_id == "doc_alice_123"
    assert reranked_result.source == "alice_resume.txt.txt"
    assert reranked_result.rerank_score == 7.5


def test_rerank_score_and_doc_id_have_different_responsibilities():

    alice = RerankedResult(
        text="Alice has FastAPI experience.",
        source="alice_resume.txt.txt",
        page_number=1,
        chunk_index=0,
        dense_score=0.90,
        bm25_score=0.80,
        hybrid_score=0.85,
        rerank_score=7.5,
        doc_id="doc_alice",
    )

    rahul = RerankedResult(
        text="Rahul has Python backend experience.",
        source="rahul_resume.txt.txt",
        page_number=1,
        chunk_index=0,
        dense_score=0.70,
        bm25_score=0.60,
        hybrid_score=0.65,
        rerank_score=5.7,
        doc_id="doc_rahul",
    )

    assert alice.doc_id != rahul.doc_id
    assert alice.rerank_score > rahul.rerank_score