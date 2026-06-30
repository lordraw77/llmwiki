"""Factory that builds the configured :class:`VectorStore`.

The backend is selected by ``VECTOR_BACKEND`` (``chroma`` or ``qdrant``). The
embedding ``dimension`` is required up front because Qdrant collections are
created with a fixed vector size.
"""

from __future__ import annotations

import os

from wikillm.config import Settings
from wikillm.logging_setup import get_logger
from wikillm.store.base import VectorStore

logger = get_logger(__name__)


def build_vector_store(settings: Settings, dimension: int) -> VectorStore:
    """Construct the vector store described by ``settings``.

    Args:
        settings: The loaded application settings.
        dimension: The embedding dimension (needed to create Qdrant collections).

    Returns:
        A ready-to-use :class:`VectorStore`.

    Raises:
        ValueError: If ``VECTOR_BACKEND`` is not a recognised value.
    """
    backend = settings.vector_backend
    os.makedirs(settings.data_dir, exist_ok=True)

    if backend == "chroma":
        from wikillm.store.chroma_store import ChromaStore

        os.makedirs(settings.chroma_path, exist_ok=True)
        return ChromaStore(path=settings.chroma_path, collection_name=settings.collection_name)

    if backend == "qdrant":
        from wikillm.store.qdrant_store import QdrantStore

        if not settings.qdrant_url:
            os.makedirs(settings.qdrant_path, exist_ok=True)
        return QdrantStore(
            collection_name=settings.collection_name,
            dimension=dimension,
            path=settings.qdrant_path,
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key,
        )

    raise ValueError(f"Unknown VECTOR_BACKEND: {backend!r} (expected 'chroma' or 'qdrant').")
