"""Vector store abstraction shared by the ChromaDB and Qdrant backends.

The interface is deliberately tiny — add chunks, search, and manage documents —
so the knowledge base never couples to a specific database. Document-level
metadata (title, source, content_type, created_at) is stored on every chunk so
listing and per-document operations work without a separate documents table.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any, List

from wikillm.models import DocumentSummary, SearchHit


@dataclass
class StoredChunk:
    """A chunk plus its embedding, ready to be written to a vector store.

    Attributes:
        id: Unique chunk id.
        document_id: Id of the parent document.
        text: The chunk text.
        embedding: The chunk's embedding vector.
        metadata: Flat metadata dict. By convention it includes ``title``,
            ``source``, ``content_type``, ``created_at`` and ``chunk_index``.
    """

    id: str
    document_id: str
    text: str
    embedding: List[float]
    metadata: dict[str, Any] = field(default_factory=dict)


class VectorStore(abc.ABC):
    """Abstract persistence + similarity-search backend."""

    @abc.abstractmethod
    def add(self, chunks: List[StoredChunk]) -> None:
        """Insert (or upsert) a batch of chunks.

        Args:
            chunks: The chunks to store. Existing ids are overwritten.
        """

    @abc.abstractmethod
    def search(self, embedding: List[float], k: int) -> List[SearchHit]:
        """Return the ``k`` chunks most similar to ``embedding``.

        Args:
            embedding: The query embedding vector.
            k: Maximum number of results to return.

        Returns:
            Ranked :class:`~wikillm.models.SearchHit` results (best first), with
            ``score`` normalised to ``[0, 1]``.
        """

    @abc.abstractmethod
    def list_documents(self) -> List[DocumentSummary]:
        """Return a summary of every distinct document currently stored."""

    @abc.abstractmethod
    def get_document(self, document_id: str) -> List[StoredChunk]:
        """Return all chunks belonging to a document, ordered by chunk index.

        Args:
            document_id: The document id to fetch.

        Returns:
            The document's chunks (empty list if the document is unknown).
        """

    @abc.abstractmethod
    def delete_document(self, document_id: str) -> int:
        """Delete all chunks of a document.

        Args:
            document_id: The document id to delete.

        Returns:
            The number of chunks removed.
        """

    @abc.abstractmethod
    def count(self) -> int:
        """Return the total number of stored chunks."""
