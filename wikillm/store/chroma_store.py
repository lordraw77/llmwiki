"""ChromaDB-backed vector store (the default, embedded backend).

ChromaDB is used in *persistent client* mode: data lives on disk under
``CHROMA_PATH`` and survives restarts with zero external services. We supply our
own embeddings (the collection is created without an embedding function), so
Chroma is used purely as a vector index + metadata store.
"""

from __future__ import annotations

import sys
from typing import Any, List

# ChromaDB requires sqlite3 >= 3.35. Many older Linux distributions ship an
# older system sqlite, so transparently swap in the bundled `pysqlite3-binary`
# when it is available and newer. This must happen before importing chromadb.
try:  # pragma: no cover - environment dependent
    import sqlite3 as _stdlib_sqlite3

    if _stdlib_sqlite3.sqlite_version_info < (3, 35, 0):
        import pysqlite3  # type: ignore

        sys.modules["sqlite3"] = pysqlite3
        sys.modules["sqlite3.dbapi2"] = pysqlite3.dbapi2  # type: ignore[attr-defined]
except ImportError:
    # pysqlite3 not installed; chromadb will raise a clear error if needed.
    pass

from wikillm.logging_setup import get_logger
from wikillm.models import DocumentSummary, SearchHit
from wikillm.store.base import StoredChunk, VectorStore

logger = get_logger(__name__)


class ChromaStore(VectorStore):
    """A :class:`VectorStore` backed by a persistent ChromaDB collection."""

    def __init__(self, path: str, collection_name: str) -> None:
        """Open (or create) a persistent ChromaDB collection.

        Args:
            path: Directory where ChromaDB persists its data.
            collection_name: Name of the collection to use.
        """
        import chromadb
        from chromadb.config import Settings as ChromaSettings

        self._client = chromadb.PersistentClient(
            path=path,
            settings=ChromaSettings(anonymized_telemetry=False, allow_reset=False),
        )
        # Cosine space matches our normalised embeddings; distance is in [0, 2].
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info("ChromaStore ready at '%s' (collection='%s').", path, collection_name)

    def add(self, chunks: List[StoredChunk]) -> None:
        """Upsert chunks into the collection."""
        if not chunks:
            return
        self._collection.upsert(
            ids=[chunk.id for chunk in chunks],
            embeddings=[chunk.embedding for chunk in chunks],
            documents=[chunk.text for chunk in chunks],
            metadatas=[self._encode_metadata(chunk) for chunk in chunks],
        )

    def search(self, embedding: List[float], k: int) -> List[SearchHit]:
        """Return the ``k`` nearest chunks to ``embedding``."""
        if k <= 0 or self.count() == 0:
            return []
        result = self._collection.query(
            query_embeddings=[embedding],
            n_results=min(k, self.count()),
            include=["documents", "metadatas", "distances"],
        )

        ids = result.get("ids", [[]])[0]
        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]

        hits: List[SearchHit] = []
        for chunk_id, text, metadata, distance in zip(ids, documents, metadatas, distances):
            metadata = metadata or {}
            # Cosine distance in [0, 2] -> similarity in [0, 1].
            score = max(0.0, 1.0 - float(distance) / 2.0)
            hits.append(
                SearchHit(
                    chunk_id=chunk_id,
                    document_id=str(metadata.get("document_id", "")),
                    title=str(metadata.get("title", "")),
                    source=str(metadata.get("source", "")),
                    text=text or "",
                    score=score,
                    metadata=metadata,
                )
            )
        return hits

    def list_documents(self) -> List[DocumentSummary]:
        """Return one summary per distinct document in the collection."""
        records = self._collection.get(include=["metadatas"])
        metadatas = records.get("metadatas", []) or []
        return _summarise_documents(metadatas)

    def get_document(self, document_id: str) -> List[StoredChunk]:
        """Return all chunks of a document, ordered by chunk index."""
        records = self._collection.get(
            where={"document_id": document_id},
            include=["documents", "metadatas", "embeddings"],
        )
        # Note: do not use ``value or []`` here — Chroma returns ``embeddings``
        # as a NumPy array, whose truth value is ambiguous. Use explicit
        # ``is None`` checks instead.
        ids = _as_list(records.get("ids"))
        documents = _as_list(records.get("documents"))
        metadatas = _as_list(records.get("metadatas"))
        embeddings = _as_list(records.get("embeddings"))

        chunks: List[StoredChunk] = []
        for index, chunk_id in enumerate(ids):
            metadata = metadatas[index] if index < len(metadatas) else {}
            embedding = embeddings[index] if index < len(embeddings) else []
            chunks.append(
                StoredChunk(
                    id=chunk_id,
                    document_id=document_id,
                    text=documents[index] if index < len(documents) else "",
                    embedding=list(embedding),
                    metadata=metadata or {},
                )
            )
        chunks.sort(key=lambda c: int(c.metadata.get("chunk_index", 0)))
        return chunks

    def delete_document(self, document_id: str) -> int:
        """Delete every chunk of a document and return the count removed."""
        existing = self._collection.get(where={"document_id": document_id})
        ids = existing.get("ids", []) or []
        if ids:
            self._collection.delete(ids=ids)
        return len(ids)

    def count(self) -> int:
        """Return the total number of chunks in the collection."""
        return int(self._collection.count())

    @staticmethod
    def _encode_metadata(chunk: StoredChunk) -> dict[str, Any]:
        """Build a Chroma-safe metadata dict for a chunk.

        Chroma only accepts scalar metadata values, so the document id is
        always present and other values are passed through as-is (callers store
        only scalars).

        Args:
            chunk: The chunk being stored.

        Returns:
            A flat metadata dict including ``document_id``.
        """
        metadata = dict(chunk.metadata)
        metadata["document_id"] = chunk.document_id
        return metadata


def _as_list(value: Any) -> list:
    """Normalise a Chroma result field to a plain list.

    Chroma may return ``None`` (field not requested) or a NumPy array
    (embeddings). Using ``value or []`` is unsafe on arrays because their truth
    value is ambiguous, so this helper handles both cases explicitly.

    Args:
        value: A raw value from a Chroma result dict.

    Returns:
        A list (empty when ``value`` is ``None``).
    """
    if value is None:
        return []
    return list(value)


def _summarise_documents(metadatas: List[dict[str, Any]]) -> List[DocumentSummary]:
    """Aggregate per-chunk metadata into per-document summaries.

    Args:
        metadatas: Flat metadata dicts, one per stored chunk.

    Returns:
        One :class:`~wikillm.models.DocumentSummary` per distinct document id.
    """
    grouped: dict[str, dict[str, Any]] = {}
    for metadata in metadatas:
        metadata = metadata or {}
        document_id = str(metadata.get("document_id", ""))
        if not document_id:
            continue
        entry = grouped.setdefault(
            document_id,
            {
                "title": metadata.get("title", ""),
                "source": metadata.get("source", ""),
                "content_type": metadata.get("content_type", ""),
                "created_at": metadata.get("created_at"),
                "chunk_count": 0,
            },
        )
        entry["chunk_count"] += 1

    return [
        DocumentSummary(
            document_id=document_id,
            title=str(entry["title"]),
            source=str(entry["source"]),
            content_type=str(entry["content_type"]),
            chunk_count=int(entry["chunk_count"]),
            created_at=entry["created_at"],
        )
        for document_id, entry in grouped.items()
    ]
