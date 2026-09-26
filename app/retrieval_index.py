### `app/retrieval_index.py`

from app.bm25_retriever import RebuildableLexicalIndex
from app.chunker import DocumentChunk


class RetrievalIndexManager:
    """
    Coordinate rebuilding of the configured lexical index.

    The manager does not know whether the lexical implementation
    is BM25, PostgreSQL FTS, or another future implementation.
    """

    def __init__(
        self,
        lexical_index: RebuildableLexicalIndex,
    ) -> None:
        self.lexical_index = lexical_index

    def rebuild(
        self,
        chunks: list[DocumentChunk],
    ) -> int:
        if not chunks:
            return 0

        self.lexical_index.build(chunks)

        return len(chunks)
    
    def remove_document(
        self,
        doc_id: str,
    ) -> None:
        if not doc_id.strip():
            raise ValueError("doc_id cannot be empty")

        self.lexical_index.remove_document(doc_id)
