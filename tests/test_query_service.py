import pytest

from app.compressor import ContextCompressor
from app.config import Settings
from app.hybrid_retriever import HybridResult
from app.query_service import QueryService
from app.reranker import RerankedResult


class FakeHybridRetriever:
    def retrieve(
            self, 
            query: str, 
            top_k: int, 
            # candidate_k: int
            ):
        assert query == "How long do I have to get a refund?"
        assert top_k == 10
        # assert candidate_k == 10

        return [
            HybridResult(
                text="Refunds are available within 30 days.",
                source="policy.pdf",
                page_number=1,
                chunk_index=0,
                dense_score=0.9,
                bm25_score=0.8,
                score=0.85,
            ),
            HybridResult(
                text="Refund requests must include the original receipt.",
                source="policy.pdf",
                page_number=2,
                chunk_index=1,
                dense_score=0.7,
                bm25_score=0.6,
                score=0.65,
            ),
        ]


class FakeReranker:
    def rerank(self, query: str, candidates: list[HybridResult], top_k: int):
        assert query == "How long do I have to get a refund?"
        assert len(candidates) == 2
        assert top_k == 5

        return [
            RerankedResult(
                text="Refunds are available within 30 days.",
                source="policy.pdf",
                page_number=1,
                chunk_index=0,
                dense_score=0.9,
                bm25_score=0.8,
                hybrid_score=0.85,
                rerank_score=9.2,
            ),
            RerankedResult(
                text="Refund requests must include the original receipt.",
                source="policy.pdf",
                page_number=2,
                chunk_index=1,
                dense_score=0.7,
                bm25_score=0.6,
                hybrid_score=0.65,
                rerank_score=7.1,
            ),
        ][:top_k]


class FakeCompressor:
    def compress(
        self,
        results: list[RerankedResult],
        top_k: int,
    ) -> list[RerankedResult]:
        assert len(results) == 2
        assert top_k == 3

        return results[:top_k]


class FakeGenerationResult:
    def __init__(self, answer: str, contexts):
        self.answer = answer
        self.contexts = contexts


class FakeGenerator:
    def generate(
        self,
        query: str,
        contexts: list[RerankedResult],
    ):
        assert query == "How long do I have to get a refund?"
        assert len(contexts) > 0

        return FakeGenerationResult(
            answer="You can request a refund within 30 days.",
            contexts=contexts,
        ) 


class RecordingHybridRetriever:
    """
    Fake hybrid retriever that records the parameters supplied by QueryService.
    """

    def __init__(self):
        self.last_top_k = None
        self.last_candidate_k = None

    def retrieve(
        self,
        query: str,
        top_k: int,
        # candidate_k: int,
    ) -> list[HybridResult]:
        self.last_top_k = top_k
        # self.last_candidate_k = candidate_k

        return [
            HybridResult(
                text="Refunds are available within 30 days.",
                source="policy.pdf",
                page_number=1,
                chunk_index=0,
                dense_score=0.9,
                bm25_score=0.8,
                score=0.85,
            )
        ]


class RecordingReranker:
    """
    Fake reranker that records the top_k supplied by QueryService.
    """

    def __init__(self):
        self.last_top_k = None

    def rerank(
        self,
        query: str,
        candidates: list[HybridResult],
        top_k: int,
    ) -> list[RerankedResult]:
        self.last_top_k = top_k

        return [
            RerankedResult(
                text="Refunds are available within 30 days.",
                source="policy.pdf",
                page_number=1,
                chunk_index=0,
                dense_score=0.9,
                bm25_score=0.8,
                hybrid_score=0.85,
                rerank_score=9.2,
            )
        ]


class RecordingCompressor:
    """
    Fake compressor that records the top_k supplied by QueryService.
    """

    def __init__(self):
        self.last_top_k = None

    def compress(
        self,
        results: list[RerankedResult],
        top_k: int,
    ) -> list[RerankedResult]:
        self.last_top_k = top_k
        return results[:top_k]


def test_query_service_runs_pipeline():
    service = QueryService(
        hybrid_retriever=FakeHybridRetriever(),
        reranker=FakeReranker(),
        compressor=FakeCompressor(),
        generator=FakeGenerator(),
    )

    result = service.query(
        question="How long do I have to get a refund?"
    )

    assert result.answer == "You can request a refund within 30 days."
    assert len(result.contexts) == 2
    assert result.contexts[0].source == "policy.pdf"
    assert result.contexts[0].page_number == 1


def test_query_service_rejects_empty_question():
    service = QueryService(
        hybrid_retriever=FakeHybridRetriever(),
        reranker=FakeReranker(),
        compressor=FakeCompressor(),
        generator=FakeGenerator(),
    )

    with pytest.raises(ValueError, match="question cannot be empty"):
        service.query(question="   ")


def test_query_service_rejects_invalid_retrieval_top_k():
    service = QueryService(
        hybrid_retriever=FakeHybridRetriever(),
        reranker=FakeReranker(),
        compressor=FakeCompressor(),
        generator=FakeGenerator(),
    )

    with pytest.raises(
        ValueError,
        match="retrieval_top_k must be greater than zero",
    ):
        service.query(
            question="How long do I have to get a refund?",
            retrieval_top_k=0,
        )


def test_query_service_rejects_invalid_rerank_top_k():
    service = QueryService(
        hybrid_retriever=FakeHybridRetriever(),
        reranker=FakeReranker(),
        compressor=FakeCompressor(),
        generator=FakeGenerator(),
    )

    with pytest.raises(
        ValueError,
        match="rerank_top_k must be greater than zero",
    ):
        service.query(
            question="How long do I have to get a refund?",
            rerank_top_k=0,
        )


def test_query_service_rejects_invalid_context_top_k():
    service = QueryService(
        hybrid_retriever=FakeHybridRetriever(),
        reranker=FakeReranker(),
        compressor=FakeCompressor(),
        generator=FakeGenerator(),
    )

    with pytest.raises(
        ValueError,
        match="context_top_k must be greater than zero",
    ):
        service.query(
            question="How long do I have to get a refund?",
            context_top_k=0,
        )


def test_query_service_rejects_no_hybrid_results():
    class EmptyHybridRetriever:
        def retrieve(
                self, 
                query: str, 
                top_k: int, 
                # candidate_k: int
                ):
            return []

    service = QueryService(
        hybrid_retriever=EmptyHybridRetriever(),
        reranker=FakeReranker(),
        compressor=FakeCompressor(),
        generator=FakeGenerator(),
    )

    with pytest.raises(
        ValueError,
        match="No relevant documents found",
    ):
        service.query(
            question="How long do I have to get a refund?"
        )


def test_query_service_rejects_no_reranked_results():
    class EmptyReranker:
        def rerank(
            self,
            query: str,
            candidates: list[HybridResult],
            top_k: int,
        ):
            return []

    service = QueryService(
        hybrid_retriever=FakeHybridRetriever(),
        reranker=EmptyReranker(),
        compressor=FakeCompressor(),
        generator=FakeGenerator(),
    )

    with pytest.raises(
        ValueError,
        match="No relevant documents remained after reranking",
    ):
        service.query(
            question="How long do I have to get a refund?"
        )


def test_query_service_rejects_no_context():
    class EmptyCompressor:
        def compress(
            self,
            results: list[RerankedResult],
            top_k: int,
        ):
            return []

    service = QueryService(
        hybrid_retriever=FakeHybridRetriever(),
        reranker=FakeReranker(),
        compressor=EmptyCompressor(),
        generator=FakeGenerator(),
    )

    with pytest.raises(
        ValueError,
        match="No context available for generation",
    ):
        service.query(
            question="How long do I have to get a refund?"
        )


def test_query_service_uses_settings():
    config = Settings(
        retrieval_top_k=20,
        rerank_top_k=7,
        context_top_k=2,
    )

    hybrid_retriever = RecordingHybridRetriever()
    reranker = RecordingReranker()
    compressor = RecordingCompressor()
    generator = FakeGenerator()

    service = QueryService(
        hybrid_retriever=hybrid_retriever,
        reranker=reranker,
        compressor=compressor,
        generator=generator,
        config=config,
    )

    result = service.query(
        question="How long do I have to get a refund?"
    )

    assert result.answer

    assert hybrid_retriever.last_top_k == 20
    # assert hybrid_retriever.last_candidate_k == 20

    assert reranker.last_top_k == 7

    assert compressor.last_top_k == 2


def test_query_service_explicit_values_override_settings():
    config = Settings(
        retrieval_top_k=20,
        rerank_top_k=7,
        context_top_k=2,
    )

    hybrid_retriever = RecordingHybridRetriever()
    reranker = RecordingReranker()
    compressor = RecordingCompressor()
    generator = FakeGenerator()

    service = QueryService(
        hybrid_retriever=hybrid_retriever,
        reranker=reranker,
        compressor=compressor,
        generator=generator,
        config=config,
    )

    service.query(
        question="How long do I have to get a refund?",
        retrieval_top_k=5,
        rerank_top_k=3,
        context_top_k=1,
    )

    assert hybrid_retriever.last_top_k == 5
    # assert hybrid_retriever.last_candidate_k == 5

    assert reranker.last_top_k == 3

    assert compressor.last_top_k == 1