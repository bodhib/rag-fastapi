from app.hybrid_retriever import HybridResult
from app.reranker import RerankedResult
from app.search_service import SearchService


class FakeHybridRetriever:
    def __init__(self, result):
        self.result = result

    def retrieve(self, query, top_k, knowledge_base_id=None):
        return [self.result]


class FakeReranker:
    def __init__(self, result):
        self.result = result

    def rerank(self, query, candidates, top_k):
        return [self.result]


class FakeConfig:
    retrieval_top_k = 10
    rerank_top_k = 5
    context_top_k = 3


def test_search_result_preserves_doc_id():
    hybrid_result = HybridResult(
        text="Alice has FastAPI backend experience.",
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

    service = SearchService(
        hybrid_retriever=FakeHybridRetriever(hybrid_result),
        reranker=FakeReranker(reranked_result),
        config=FakeConfig(),
    )

    results = service.search(
        query="FastAPI backend",
        knowledge_base_id="kb_recruitment",
        top_k=5,
    )

    assert len(results) == 1
    assert results[0].doc_id == "doc_alice_123"
    assert results[0].source == "alice_resume.txt.txt"
    assert results[0].rerank_score == 7.5


def test_search_result_identity_is_not_based_on_filename():
    alice = RerankedResult(
        text="Alice FastAPI experience.",
        source="resume.txt",
        page_number=1,
        chunk_index=0,
        dense_score=0.90,
        bm25_score=0.80,
        hybrid_score=0.85,
        rerank_score=7.5,
        doc_id="doc_alice",
    )

    rahul = RerankedResult(
        text="Rahul Python experience.",
        source="resume.txt",
        page_number=1,
        chunk_index=0,
        dense_score=0.70,
        bm25_score=0.60,
        hybrid_score=0.65,
        rerank_score=5.7,
        doc_id="doc_rahul",
    )

    assert alice.source == rahul.source
    assert alice.doc_id != rahul.doc_id