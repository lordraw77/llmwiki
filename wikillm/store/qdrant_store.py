"""Qdrant-backed vector store (alternative backend).

Qdrant can run embedded (a local on-disk path, no server) or against a remote
server (URL + optional API key). The collection is created lazily with the
embedding dimension supplied at construction time, using cosine distance to
match our normalised embeddings.

Qdrant point ids must be integers or UUIDs, while our chunk ids are strings
(``<document_id>::<index>``). We therefore derive a deterministic UUID5 from the
chunk id for the point id and keep the original chunk id inside the payload.
"""

from __future__ import annotations

import uuid
from typing import Any, List, Optional

from wikillm.logging_setup import get_logger
from wikillm.models import DocumentSummary, SearchHit
from wikillm.store.base import StoredChunk, VectorStore
from wikillm.store.chroma_store import _summarise_documents

logger = get_logger(__name__)

# Stable namespace so the same chunk id always maps to the same point id.
_POINT_NAMESPACE = uuid.UUID("9f8d2c4e-6b1a-4f3e-9c7d-0a1b2c3d4e5f")


class QdrantStore(VectorStore):
    """A :class:`VectorStore` backed by a Qdrant collection."""

    def __init__(
        self,
        collection_name: str,
        dimension: int,
        path: Optional[str] = None,
        url: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> None:
        """Open or create a Qdrant collection.

        Args:
            collection_name: Name of the collection.
            dimension: Embedding vector size (used when creating the collection).
            path: Local on-disk path for embedded mode. Ignored if ``url`` set.
            url: Remote Qdrant server URL. Takes precedence over ``path``.
            api_key: API key for the remote server (optional).

        Raises:
            ValueError: If neither ``path`` nor ``url`` is provided.
        """
        from qdrant_client import QdrantClient
        from qdrant_client.http import models as qmodels

        self._qmodels = qmodels
        self._collection = collection_name

        self._is_server = bool(url)
        if url:
            self._client = QdrantClient(url=url, api_key=api_key)
            logger.info("QdrantStore connected to server '%s'.", url)
        elif path:
            self._client = QdrantClient(path=path)
            logger.info("QdrantStore using embedded path '%s'.", path)
        else:
            raise ValueError("QdrantStore requires either 'path' (embedded) or 'url' (server).")

        self._ensure_collection(dimension)

    def _ensure_collection(self, dimension: int) -> None:
        """Create the collection with cosine distance if it does not exist."""
        if self._client.collection_exists(self._collection):
            return
        self._client.create_collection(
            collection_name=self._collection,
            vectors_config=self._qmodels.VectorParams(
                size=dimension,
                distance=self._qmodels.Distance.COSINE,
            ),
        )
        # Index document_id so filtering / deletion by document is efficient.
        # Payload indexes are a no-op in embedded (local) mode, so only create
        # one when talking to a real Qdrant server.
        if self._is_server:
            self._client.create_payload_index(
                collection_name=self._collection,
                field_name="document_id",
                field_schema=self._qmodels.PayloadSchemaType.KEYWORD,
            )
        logger.info("Created Qdrant collection '%s' (dim=%d).", self._collection, dimension)

    def add(self, chunks: List[StoredChunk]) -> None:
        """Upsert chunks as Qdrant points."""
        if not chunks:
            return
        points = []
        for chunk in chunks:
            payload = dict(chunk.metadata)
            payload["document_id"] = chunk.document_id
            payload["chunk_id"] = chunk.id
            payload["text"] = chunk.text
            points.append(
                self._qmodels.PointStruct(
                    id=self._point_id(chunk.id),
                    vector=chunk.embedding,
                    payload=payload,
                )
            )
        self._client.upsert(collection_name=self._collection, points=points)

    def search(self, embedding: List[float], k: int) -> List[SearchHit]:
        """Return the ``k`` nearest chunks to ``embedding``."""
        if k <= 0:
            return []
        results = self._client.query_points(
            collection_name=self._collection,
            query=embedding,
            limit=k,
            with_payload=True,
        ).points

        hits: List[SearchHit] = []
        for point in results:
            payload = point.payload or {}
            # Qdrant cosine score is already a similarity in [-1, 1]; clamp to [0, 1].
            score = max(0.0, min(1.0, float(point.score)))
            hits.append(
                SearchHit(
                    chunk_id=str(payload.get("chunk_id", point.id)),
                    document_id=str(payload.get("document_id", "")),
                    title=str(payload.get("title", "")),
                    source=str(payload.get("source", "")),
                    text=str(payload.get("text", "")),
                    score=score,
                    metadata={k: v for k, v in payload.items() if k != "text"},
                )
            )
        return hits

    def list_documents(self) -> List[DocumentSummary]:
        """Return one summary per distinct document in the collection."""
        metadatas = [payload for payload in self._scroll_payloads()]
        return _summarise_documents(metadatas)

    def get_document(self, document_id: str) -> List[StoredChunk]:
        """Return all chunks of a document, ordered by chunk index."""
        records, _ = self._client.scroll(
            collection_name=self._collection,
            scroll_filter=self._document_filter(document_id),
            with_payload=True,
            with_vectors=True,
            limit=10_000,
        )
        chunks: List[StoredChunk] = []
        for record in records:
            payload = record.payload or {}
            chunks.append(
                StoredChunk(
                    id=str(payload.get("chunk_id", record.id)),
                    document_id=document_id,
                    text=str(payload.get("text", "")),
                    embedding=list(record.vector) if record.vector else [],
                    metadata={k: v for k, v in payload.items() if k != "text"},
                )
            )
        chunks.sort(key=lambda c: int(c.metadata.get("chunk_index", 0)))
        return chunks

    def delete_document(self, document_id: str) -> int:
        """Delete every point belonging to a document; return count removed."""
        existing = self.get_document(document_id)
        if not existing:
            return 0
        self._client.delete(
            collection_name=self._collection,
            points_selector=self._document_filter(document_id),
        )
        return len(existing)

    def count(self) -> int:
        """Return the total number of points in the collection."""
        return int(self._client.count(collection_name=self._collection, exact=True).count)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _document_filter(self, document_id: str):
        """Build a Qdrant filter matching all points of a document."""
        return self._qmodels.Filter(
            must=[
                self._qmodels.FieldCondition(
                    key="document_id",
                    match=self._qmodels.MatchValue(value=document_id),
                )
            ]
        )

    def _scroll_payloads(self) -> List[dict[str, Any]]:
        """Scroll the entire collection, yielding each point's payload."""
        payloads: List[dict[str, Any]] = []
        offset = None
        while True:
            records, offset = self._client.scroll(
                collection_name=self._collection,
                with_payload=True,
                with_vectors=False,
                limit=1000,
                offset=offset,
            )
            payloads.extend(record.payload or {} for record in records)
            if offset is None:
                break
        return payloads

    @staticmethod
    def _point_id(chunk_id: str) -> str:
        """Map a string chunk id to a deterministic UUID point id."""
        return str(uuid.uuid5(_POINT_NAMESPACE, chunk_id))
