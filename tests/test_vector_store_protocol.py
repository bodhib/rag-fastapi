from pathlib import Path

from app.vector_store import (
    ChromaVectorStore,
    VectorStore,
)


def test_chroma_vector_store_satisfies_vector_store_protocol(
    tmp_path: Path,
) -> None:
    store = ChromaVectorStore(
        persist_directory=tmp_path / "chroma",
    )

    assert isinstance(
        store,
        VectorStore,
    )