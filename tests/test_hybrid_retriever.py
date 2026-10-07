# tests/test_hybrid_retriever.py

from app.hybrid_retriever import HybridResult


def test_hybrid_result_preserves_doc_id():

    result = HybridResult(
        text="Alice has extensive FastAPI experience.",
        source="alice_resume.txt.txt",
        page_number=1,
        chunk_index=0,
        dense_score=0.90,
        bm25_score=0.80,
        score=0.85,
        doc_id="doc_alice_123",
    )

    assert result.doc_id == "doc_alice_123"
    assert result.source == "alice_resume.txt.txt"
    assert result.page_number == 1
    assert result.chunk_index == 0


def test_hybrid_result_can_represent_two_documents_with_same_filename():

    alice = HybridResult(
        text="Alice has FastAPI experience.",
        source="resume.txt",
        page_number=1,
        chunk_index=0,
        dense_score=0.90,
        bm25_score=0.80,
        score=0.85,
        doc_id="doc_alice",
    )

    rahul = HybridResult(
        text="Rahul has Django experience.",
        source="resume.txt",
        page_number=1,
        chunk_index=0,
        dense_score=0.70,
        bm25_score=0.60,
        score=0.65,
        doc_id="doc_rahul",
    )

    assert alice.source == rahul.source
    assert alice.doc_id != rahul.doc_id