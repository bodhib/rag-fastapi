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
        knowledge_base_id: str | None = None,
    ):
        assert query == "How long do I have to get a refund?"
        assert top_k == 10
        assert knowledge_base_id == "kb_default"

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
                text=(
                    "Refund requests must include "
                    "the original receipt."
                ),
                source="policy.pdf",
                page_number=2,
                chunk_index=1,
                dense_score=0.7,
                bm25_score=0.6,
                score=0.65,
            ),
        ]


class FakeReranker:
    def rerank(
        self,
        query: str,
        candidates: list[HybridResult],
        top_k: int,
    ):
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
                text=(
                    "Refund requests must include "
                    "the original receipt."
                ),
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
    def __init__(
        self,
        answer: str,
        contexts,
    ):
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
    def __init__(self):
        self.last_top_k = None
        self.last_knowledge_base_id = None

    def retrieve(
        self,
        query: str,
        top_k: int,
        knowledge_base_id: str | None = None,
    ) -> list[HybridResult]:
        self.last_top_k = top_k
        self.last_knowledge_base_id = knowledge_base_id

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
        question="How long do I have to get a refund?",
        knowledge_base_id="kb_default",
    )

    assert (
        result.answer
        == "You can request a refund within 30 days."
    )

    assert len(result.contexts) == 2
    assert result.contexts[0].source == "policy.pdf"
    assert result.contexts[0].page_number == 1


def test_query_service_forwards_knowledge_base_id():
    hybrid_retriever = RecordingHybridRetriever()

    service = QueryService(
        hybrid_retriever=hybrid_retriever,
        reranker=RecordingReranker(),
        compressor=RecordingCompressor(),
        generator=FakeGenerator(),
    )

    service.query(
        question="How long do I have to get a refund?",
        knowledge_base_id="kb_company",
    )

    assert (
        hybrid_retriever.last_knowledge_base_id
        == "kb_company"
    )


def test_query_service_rejects_empty_knowledge_base_id():
    service = QueryService(
        hybrid_retriever=FakeHybridRetriever(),
        reranker=FakeReranker(),
        compressor=FakeCompressor(),
        generator=FakeGenerator(),
    )

    with pytest.raises(
        ValueError,
        match="knowledge_base_id cannot be empty",
    ):
        service.query(
            question="How long do I have to get a refund?",
            knowledge_base_id="   ",
        )


def test_query_service_rejects_empty_question():
    service = QueryService(
        hybrid_retriever=FakeHybridRetriever(),
        reranker=FakeReranker(),
        compressor=FakeCompressor(),
        generator=FakeGenerator(),
    )

    with pytest.raises(
        ValueError,
        match="question cannot be empty",
    ):
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
            knowledge_base_id: str | None = None,
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
    assert (
        hybrid_retriever.last_knowledge_base_id
        == "kb_default"
    )

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
        knowledge_base_id="kb_company",
        retrieval_top_k=5,
        rerank_top_k=3,
        context_top_k=1,
    )

    assert hybrid_retriever.last_top_k == 5
    assert (
        hybrid_retriever.last_knowledge_base_id
        == "kb_company"
    )

    assert reranker.last_top_k == 3
    assert compressor.last_top_k == 1

def test_query_service_defaults_to_default_knowledge_base():
    hybrid_retriever = RecordingHybridRetriever()

    service = QueryService(
        hybrid_retriever=hybrid_retriever,
        reranker=RecordingReranker(),
        compressor=RecordingCompressor(),
        generator=FakeGenerator(),
    )

    service.query(
        question="How long do I have to get a refund?",
    )

    assert (
        hybrid_retriever.last_knowledge_base_id
        == "kb_default"
    )   

def test_query_service_defaults_to_default_knowledge_base():
    hybrid_retriever = RecordingHybridRetriever()

    service = QueryService(
        hybrid_retriever=hybrid_retriever,
        reranker=RecordingReranker(),
        compressor=RecordingCompressor(),
        generator=FakeGenerator(),
    )

    service.query(
        question="How long do I have to get a refund?",
    )

    assert (
        hybrid_retriever.last_knowledge_base_id
        == "kb_default"
    )     