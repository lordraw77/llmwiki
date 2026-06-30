"""Vector stores and their factory.

A *vector store* persists chunk embeddings together with their text/metadata and
answers nearest-neighbour queries. Two interchangeable backends implement the
common :class:`~wikillm.store.base.VectorStore` contract:

* :class:`~wikillm.store.chroma_store.ChromaStore` — embedded ChromaDB (default).
* :class:`~wikillm.store.qdrant_store.QdrantStore` — Qdrant (local path or
  remote server).

Use :func:`~wikillm.store.factory.build_vector_store` to construct one from
:class:`~wikillm.config.Settings`.
"""

from __future__ import annotations

from wikillm.store.base import StoredChunk, VectorStore
from wikillm.store.factory import build_vector_store

__all__ = ["StoredChunk", "VectorStore", "build_vector_store"]
