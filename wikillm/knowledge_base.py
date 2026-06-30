"""The :class:`KnowledgeBase` orchestrator and its process-wide singleton.

This module wires the three lower layers together:

    parse → chunk → embed → store        (ingestion)
    embed query → store.search           (retrieval)

It is the single object both the REST API and the MCP server talk to, which
guarantees they operate on exactly the same index. A lazily-built singleton is
exposed via :func:`get_knowledge_base` so the embedding model / vector store are
constructed only once per process.
"""

from __future__ import annotations

import os
import threading
import uuid
from typing import Any, List, Optional

from wikillm.config import Settings, get_settings
from wikillm.embeddings import EmbeddingProvider, build_embedding_provider
from wikillm.ingestion import (
    chunk_text,
    iter_supported_files,
    parse_bytes,
    parse_path,
)
from wikillm.logging_setup import get_logger
from wikillm.models import (
    DocumentSummary,
    IngestResponse,
    ParsedDocument,
    SearchHit,
)
from wikillm.store import StoredChunk, VectorStore, build_vector_store

logger = get_logger(__name__)


class KnowledgeBase:
    """High-level ingestion + semantic search over a document corpus.

    Args:
        embedder: The embedding provider used for documents and queries.
        store: The vector store that persists chunks and answers queries.
        chunk_size: Target chunk length in characters.
        chunk_overlap: Overlap between consecutive chunks.
    """

    def __init__(
        self,
        embedder: EmbeddingProvider,
        store: VectorStore,
        chunk_size: int,
        chunk_overlap: int,
    ) -> None:
        self._embedder = embedder
        self._store = store
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------
    def ingest_document(self, document: ParsedDocument) -> int:
        """Chunk, embed and store an already-parsed document.

        Args:
            document: The parsed document to index.

        Returns:
            The number of chunks indexed.
        """
        texts = chunk_text(document.text, self._chunk_size, self._chunk_overlap)
        if not texts:
            logger.warning("Document '%s' produced no chunks; skipping.", document.title)
            return 0

        embeddings = self._embedder.embed_texts(texts)
        stored: List[StoredChunk] = []
        for index, (text, embedding) in enumerate(zip(texts, embeddings)):
            metadata: dict[str, Any] = {
                "title": document.title,
                "source": document.source,
                "content_type": document.content_type,
                "created_at": document.created_at,
                "chunk_index": index,
            }
            # Promote scalar custom metadata (vector stores only accept scalars).
            for key, value in document.metadata.items():
                if isinstance(value, (str, int, float, bool)):
                    metadata[key] = value
            stored.append(
                StoredChunk(
                    id=f"{document.id}::{index}",
                    document_id=document.id,
                    text=text,
                    embedding=embedding,
                    metadata=metadata,
                )
            )

        self._store.add(stored)
        logger.info("Indexed document '%s' (%d chunks).", document.title, len(stored))
        return len(stored)

    def ingest_text(
        self,
        text: str,
        title: str,
        source: str = "note",
        content_type: str = "note",
        metadata: Optional[dict[str, Any]] = None,
    ) -> IngestResponse:
        """Ingest a raw text string (e.g. a note).

        Args:
            text: The text body to index.
            title: Title for the resulting document.
            source: Logical source label.
            content_type: Logical content type.
            metadata: Optional extra metadata.

        Returns:
            An :class:`~wikillm.models.IngestResponse` describing the result.
        """
        document = ParsedDocument(
            id=uuid.uuid4().hex,
            title=title,
            source=source,
            content_type=content_type,
            text=text,
            metadata=metadata or {},
        )
        count = self.ingest_document(document)
        return IngestResponse(
            document_ids=[document.id],
            chunk_count=count,
            message=f"Ingested '{title}' as {count} chunk(s).",
        )

    def add_note(self, title: str, content: str, metadata: Optional[dict[str, Any]] = None) -> IngestResponse:
        """Add a free-text note to the wiki.

        A thin, intention-revealing wrapper around :meth:`ingest_text`.

        Args:
            title: Note title.
            content: Note body.
            metadata: Optional metadata.

        Returns:
            The ingestion result.
        """
        return self.ingest_text(content, title=title, source="note", content_type="note", metadata=metadata)

    def ingest_bytes(
        self,
        data: bytes,
        filename: str,
        mime_type: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> IngestResponse:
        """Ingest an uploaded file given its raw bytes.

        Args:
            data: Raw file content.
            filename: Original filename (drives format detection).
            mime_type: Optional MIME hint from the upload.
            metadata: Optional metadata.

        Returns:
            The ingestion result.

        Raises:
            UnsupportedFormatError: If the format is unsupported.
            ParseError: If text extraction fails.
        """
        document = parse_bytes(data, filename, mime_type=mime_type, metadata=metadata)
        count = self.ingest_document(document)
        return IngestResponse(
            document_ids=[document.id],
            chunk_count=count,
            message=f"Ingested '{document.title}' as {count} chunk(s).",
        )

    def ingest_path(
        self,
        path: str,
        recursive: bool = True,
        metadata: Optional[dict[str, Any]] = None,
    ) -> IngestResponse:
        """Ingest a file or every supported file under a directory.

        Args:
            path: A file or directory path on the local filesystem.
            recursive: When ``path`` is a directory, descend into subdirectories.
            metadata: Optional metadata applied to every ingested document.

        Returns:
            The aggregate ingestion result across all files.

        Raises:
            FileNotFoundError: If ``path`` does not exist.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Path does not exist: {path}")

        document_ids: List[str] = []
        total_chunks = 0
        errors: List[str] = []

        for file_path in iter_supported_files(path, recursive=recursive):
            try:
                document = parse_path(file_path, metadata=metadata)
                count = self.ingest_document(document)
                document_ids.append(document.id)
                total_chunks += count
            except Exception as exc:  # keep ingesting the rest of the batch
                logger.warning("Skipping '%s': %s", file_path, exc)
                errors.append(f"{file_path}: {exc}")

        message = f"Ingested {len(document_ids)} document(s), {total_chunks} chunk(s)."
        if errors:
            message += f" {len(errors)} file(s) skipped."
        return IngestResponse(document_ids=document_ids, chunk_count=total_chunks, message=message)

    # ------------------------------------------------------------------
    # Retrieval / management
    # ------------------------------------------------------------------
    def search(self, query: str, k: int = 5) -> List[SearchHit]:
        """Run a semantic search over the indexed corpus.

        Args:
            query: The natural-language query.
            k: Maximum number of results.

        Returns:
            Ranked search hits (best first).
        """
        query = query.strip()
        if not query:
            return []
        embedding = self._embedder.embed_query(query)
        return self._store.search(embedding, k=k)

    def list_documents(self) -> List[DocumentSummary]:
        """Return summaries for every indexed document."""
        return self._store.list_documents()

    def get_document(self, document_id: str) -> Optional[ParsedDocument]:
        """Reconstruct a stored document from its chunks.

        Args:
            document_id: The id of the document to fetch.

        Returns:
            The reconstructed document, or ``None`` if it is not indexed.
        """
        chunks = self._store.get_document(document_id)
        if not chunks:
            return None
        first = chunks[0].metadata
        reserved = {"title", "source", "content_type", "chunk_index", "created_at", "document_id"}
        kwargs: dict[str, Any] = {
            "id": document_id,
            "title": str(first.get("title", "")),
            "source": str(first.get("source", "")),
            "content_type": str(first.get("content_type", "")),
            "text": "\n\n".join(chunk.text for chunk in chunks),
            "metadata": {k: v for k, v in first.items() if k not in reserved},
        }
        # Preserve the original timestamp when present; otherwise let the model
        # apply its default factory.
        created_at = first.get("created_at")
        if created_at:
            kwargs["created_at"] = str(created_at)
        return ParsedDocument(**kwargs)

    def delete_document(self, document_id: str) -> int:
        """Delete a document and all its chunks.

        Args:
            document_id: The id of the document to delete.

        Returns:
            The number of chunks removed (0 if the document was unknown).
        """
        removed = self._store.delete_document(document_id)
        logger.info("Deleted document '%s' (%d chunks).", document_id, removed)
        return removed

    def stats(self) -> dict[str, Any]:
        """Return basic statistics about the knowledge base.

        Returns:
            A dict with the embedding provider name, dimension, vector backend,
            document count and chunk count.
        """
        documents = self._store.list_documents()
        return {
            "embedder": self._embedder.name,
            "vector_backend": self._store.__class__.__name__,
            "documents": len(documents),
            "chunks": self._store.count(),
        }


# ---------------------------------------------------------------------------
# Process-wide singleton
# ---------------------------------------------------------------------------
_INSTANCE: Optional[KnowledgeBase] = None
_LOCK = threading.Lock()


def build_knowledge_base(settings: Optional[Settings] = None) -> KnowledgeBase:
    """Construct a fresh :class:`KnowledgeBase` from settings.

    Args:
        settings: Optional settings override (defaults to :func:`get_settings`).

    Returns:
        A new knowledge base instance (not the shared singleton).
    """
    settings = settings or get_settings()
    embedder = build_embedding_provider(settings)
    store = build_vector_store(settings, dimension=embedder.dimension)
    return KnowledgeBase(
        embedder=embedder,
        store=store,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )


def get_knowledge_base() -> KnowledgeBase:
    """Return the lazily-built, process-wide :class:`KnowledgeBase` singleton.

    Thread-safe: the first caller builds the instance while holding a lock; all
    subsequent callers receive the same object.

    Returns:
        The shared knowledge base.
    """
    global _INSTANCE
    if _INSTANCE is None:
        with _LOCK:
            if _INSTANCE is None:
                _INSTANCE = build_knowledge_base()
    return _INSTANCE
