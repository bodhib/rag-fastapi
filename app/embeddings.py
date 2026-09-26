"""
Text embedding service.

This module converts document chunks into dense numerical vectors
using a SentenceTransformer model.
"""

from dataclasses import dataclass

import numpy as np
from sentence_transformers import SentenceTransformer

from app.chunker import DocumentChunk

from app.config import Settings, settings


@dataclass
class EmbeddedChunk:
    """
    Represent a document chunk together with its embedding.

    Attributes:
        chunk: Original document chunk.
        embedding: Dense numerical representation of the chunk.
    """

    chunk: DocumentChunk
    embedding: list[float]


class EmbeddingService:
    """
    Generate dense embeddings for document chunks.

    The underlying embedding model is intentionally hidden behind
    this service so that the rest of the application does not need
    to know which embedding implementation is being used.
    """

    # DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

    def __init__(
    self,
    model_name: str | None = None,
    config: Settings | None = None,
) -> None:
        self.config = config or settings

        resolved_model_name = (
            model_name
            if model_name is not None
            else self.config.embedding_model
        )

        if not resolved_model_name.strip():
            raise ValueError(
                "Embedding model name cannot be empty"
            )

        self.model_name = resolved_model_name

        self._model = SentenceTransformer(
            resolved_model_name
        )
    
    @property
    def dimension(self) -> int:
        """
        Return the dimensionality of generated embeddings.

        Returns:
            Number of dimensions in each embedding vector.
        """

        return self._model.get_sentence_embedding_dimension()

    def embed_text(self, text: str) -> list[float]:
        """
        Generate an embedding for a single text.

        Args:
            text: Text to embed.

        Returns:
            Embedding represented as a list of floats.

        Raises:
            ValueError: If the supplied text is empty.
        """

        if not text.strip():
            raise ValueError(
                "Cannot generate embedding for empty text"
            )

        embedding = self._model.encode(
            text,
            normalize_embeddings=True,
        )

        return self._to_float_list(embedding)

    def embed_chunks(
        self,
        chunks: list[DocumentChunk],
    ) -> list[EmbeddedChunk]:
        """
        Generate embeddings for document chunks.

        Args:
            chunks: Document chunks to embed.

        Returns:
            Embedded chunks containing the original chunk and
            its corresponding vector.

        Raises:
            ValueError: If no chunks are supplied.
        """

        if not chunks:
            raise ValueError(
                "Cannot generate embeddings for empty chunk list"
            )

        texts = [
            chunk.text
            for chunk in chunks
        ]

        embeddings = self._model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        return [
            EmbeddedChunk(
                chunk=chunk,
                embedding=self._to_float_list(embedding),
            )
            for chunk, embedding
            in zip(chunks, embeddings)
        ]

    @staticmethod
    def _to_float_list(
        embedding: np.ndarray,
    ) -> list[float]:
        """
        Convert a NumPy embedding array into plain Python floats.

        Args:
            embedding: NumPy embedding array.

        Returns:
            List of Python floating-point values.
        """

        return embedding.astype(float).tolist()