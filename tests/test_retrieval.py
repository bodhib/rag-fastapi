from app.chunker import DocumentChunk
from app.retrieval import DenseRetriever, RetrievalResult

class FakeEmbeddingService:
    def embed_text(self, text: str) -> list[float]:
        return [1.0, 0.0, 0.0]

class FakeVectorStore:
    def search(
    self,
    query_embedding,
    top_k: int,
    knowledge_base_id: str | None = None,
    ):
        chunk = DocumentChunk(
        doc_id="doc_alice_123",
        text="Alice has extensive FastAPI experience.",
        source="alice_resume.txt.txt",
        page_number=1,
        chunk_index=0,
        knowledge_base_id=knowledge_base_id or "kb_default",
        )


        return [
            {
                "text": chunk.text,
                "metadata": {
                    "doc_id": chunk.doc_id,
                    "source": chunk.source,
                    "page_number": chunk.page_number,
                    "chunk_index": chunk.chunk_index,
                    "knowledge_base_id": chunk.knowledge_base_id,
                },
                "distance": 0.09,
            }
        ]


def test_retrieval_result_preserves_doc_id():
    result = RetrievalResult(
    doc_id="doc_alice_123",
    text="Alice has extensive FastAPI experience.",
    source="alice_resume.txt.txt",
    page_number=1,
    chunk_index=0,
    score=0.91,
    )


    assert result.doc_id == "doc_alice_123"
    assert result.source == "alice_resume.txt.txt"


def test_dense_retrieval_preserves_doc_id():
    retriever = DenseRetriever(
    embedding_service=FakeEmbeddingService(),
    vector_store=FakeVectorStore(),
    )


    results = retriever.retrieve(
        query="FastAPI backend",
        top_k=5,
        knowledge_base_id="kb_recruitment",
    )

    assert len(results) == 1

    result = results[0]

    assert result.doc_id == "doc_alice_123"
    assert result.text == "Alice has extensive FastAPI experience."
    assert result.source == "alice_resume.txt.txt"
    assert result.page_number == 1
    assert result.chunk_index == 0
    assert result.score == 0.91

