"""
Vector storage using Chroma.

This module provides a small abstraction around Chroma so that the
rest of the application does not depend directly on Chroma's API.
"""

from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import chromadb

from app.chunker import DocumentChunk
from app.embeddings import EmbeddedChunk

@runtime_checkable
class VectorStore(Protocol):
    """
    Contract implemented by vector-store backends.
    """

    @property
    def count(self) -> int:
        ...

    def add_chunks(
        self,
        embedded_chunks: list[EmbeddedChunk],
    ) -> None:
        ...

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        ...

    def get_all_chunks(self) -> list[DocumentChunk]:
        ...

    def delete_document(
        self,
        doc_id: str,
    ) -> None:
        ...

    def delete_source(
        self,
        source: str,
    ) -> None:
        ...

    def clear(self) -> None:
        ...

class ChromaVectorStore:
    """
    Persistent vector store backed by Chroma.

    Chroma uses an HNSW-based index for approximate nearest-neighbor
    vector search.
    """

    DEFAULT_COLLECTION_NAME = "documents"

    def __init__(
        self,
        persist_directory: Path,
        collection_name: str = DEFAULT_COLLECTION_NAME,
    ) -> None:
        """
        Initialize the Chroma vector store.

        Args:
            persist_directory: Directory where Chroma stores its data.
            collection_name: Name of the Chroma collection.

        Raises:
            ValueError: If the collection name is empty.
        """

        if not collection_name.strip():
            raise ValueError(
                "collection_name cannot be empty"
            )

        self.persist_directory = persist_directory
        self.collection_name = collection_name

        self.persist_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        self._client = chromadb.PersistentClient(
            path=str(self.persist_directory)
        )

        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={
                "hnsw:space": "cosine",
            },
        )

    @property
    def count(self) -> int:
        """
        Return the number of vectors stored.

        Returns:
            Number of records in the collection.
        """

        return self._collection.count()

    def add_chunks(
        self,
        embedded_chunks: list[EmbeddedChunk],
    ) -> None:
        """
        Add embedded document chunks to Chroma.

        Args:
            embedded_chunks: Chunks and their corresponding vectors.

        Raises:
            ValueError: If the supplied list is empty or a chunk has
                an empty document ID.
        """

        if not embedded_chunks:
            raise ValueError(
                "Cannot add an empty chunk list"
            )

        for item in embedded_chunks:
            if not item.chunk.doc_id.strip():
                raise ValueError(
                    "doc_id cannot be empty"
                )

        ids = [
            self._build_id(chunk)
            for chunk in embedded_chunks
        ]

        documents = [
            item.chunk.text
            for item in embedded_chunks
        ]

        embeddings = [
            item.embedding
            for item in embedded_chunks
        ]

        metadatas = [
            {
                "doc_id": item.chunk.doc_id,
                "source": item.chunk.source,
                "page_number": item.chunk.page_number,
                "chunk_index": item.chunk.chunk_index,
            }
            for item in embedded_chunks
        ]

        self._collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """
        Search the vector store for similar chunks.

        Args:
            query_embedding: Embedding of the user query.
            top_k: Maximum number of results to return.

        Returns:
            Retrieved documents with metadata and distances.

        Raises:
            ValueError: If the query embedding is empty or top_k
                is invalid.
        """

        if not query_embedding:
            raise ValueError(
                "query_embedding cannot be empty"
            )

        if top_k <= 0:
            raise ValueError(
                "top_k must be greater than zero"
            )

        if self.count == 0:
            return []

        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, self.count),
            include=[
                "documents",
                "metadatas",
                "distances",
            ],
        )

        return self._format_results(results)

    def get_all_chunks(self) -> list[DocumentChunk]:
        """
        Return all document chunks currently stored in Chroma.

        This method is primarily used to rebuild in-memory retrieval
        indexes such as BM25 after application startup.

        Returns:
            All stored document chunks.
        """

        if self.count == 0:
            return []

        results = self._collection.get(
            include=[
                "documents",
                "metadatas",
            ]
        )

        documents = results.get(
            "documents",
            [],
        )

        metadatas = results.get(
            "metadatas",
            [],
        )

        chunks: list[DocumentChunk] = []

        for document, metadata in zip(
            documents,
            metadatas,
        ):
            chunks.append(
                DocumentChunk(
                    doc_id=str(metadata["doc_id"]),
                    text=document,
                    source=str(metadata["source"]),
                    page_number=int(metadata["page_number"]),
                    chunk_index=int(metadata["chunk_index"]),
                )
            )

        chunks.sort(
            key=lambda chunk: (
                chunk.doc_id,
                chunk.chunk_index,
            )
        )

        return chunks

    def delete_document(
        self,
        doc_id: str,
    ) -> None:
        """
        Delete all chunks belonging to a document.

        Args:
            doc_id: Unique identifier of the document.

        Raises:
            ValueError: If doc_id is empty.
        """

        if not doc_id.strip():
            raise ValueError(
                "doc_id cannot be empty"
            )

        self._collection.delete(
            where={
                "doc_id": doc_id,
            }
        )

    def delete_source(
        self,
        source: str,
    ) -> None:
        """
        Delete all chunks belonging to a source filename.

        This method is retained for compatibility with the earlier
        vector-store API. New document lifecycle code should use
        delete_document() because doc_id is the canonical identity.

        Args:
            source: Original document filename.
        """

        if not source.strip():
            raise ValueError(
                "source cannot be empty"
            )

        self._collection.delete(
            where={
                "source": source,
            }
        )

    def clear(self) -> None:
        """Delete all vectors from the collection."""

        existing_ids = self._collection.get(
            include=[]
        )["ids"]

        if existing_ids:
            self._collection.delete(
                ids=existing_ids
            )

    @staticmethod
    def _build_id(
        embedded_chunk: EmbeddedChunk,
    ) -> str:
        """
        Build a deterministic unique ID for a document chunk.

        Args:
            embedded_chunk: Chunk for which to create the ID.

        Returns:
            Unique chunk ID based on doc_id and chunk index.
        """

        chunk = embedded_chunk.chunk

        return (
            f"{chunk.doc_id}:"
            f"{chunk.chunk_index}"
        )

    @staticmethod
    def _format_results(
        results: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """
        Convert Chroma's nested query response into a flat list.

        Args:
            results: Raw Chroma query response.

        Returns:
            List of individual search results.
        """

        documents = results.get(
            "documents",
            [[]],
        )[0]

        metadatas = results.get(
            "metadatas",
            [[]],
        )[0]

        distances = results.get(
            "distances",
            [[]],
        )[0]

        formatted_results = []

        for document, metadata, distance in zip(
            documents,
            metadatas,
            distances,
        ):
            formatted_results.append(
                {
                    "text": document,
                    "metadata": metadata,
                    "distance": distance,
                }
            )

        return formatted_results