"""Pydantic data models shared across the ingestion, storage and API layers.

These models form the contract between layers:

* :class:`ParsedDocument` / :class:`Chunk` describe ingested content internally.
* :class:`SearchHit` is the unit returned by semantic search.
* The ``*Request`` / ``*Response`` models are the REST API's public schema and
  double as structured payloads for MCP tool results.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

from pydantic import BaseModel, Field


def _utcnow_iso() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


class Chunk(BaseModel):
    """A contiguous slice of a document's text, ready to be embedded.

    Attributes:
        id: Globally unique chunk id (``<document_id>::<index>``).
        document_id: Id of the parent document.
        index: Zero-based position of this chunk within the document.
        text: The chunk's textual content.
        metadata: Arbitrary metadata propagated from the parent document.
    """

    id: str
    document_id: str
    index: int
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class ParsedDocument(BaseModel):
    """The result of parsing a single source (file, upload or note).

    Attributes:
        id: Stable document id (UUID4 hex by default, assigned by the caller).
        title: Human-friendly title (often the filename or a note title).
        source: Where the content came from (filename, path, or ``"note"``).
        content_type: Detected logical type (``txt``/``md``/``pdf``/``docx``/``note``).
        text: Full extracted plain text.
        metadata: Extra key/value metadata stored alongside the document.
        created_at: ISO-8601 creation timestamp.
    """

    id: str
    title: str
    source: str
    content_type: str
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: str = Field(default_factory=_utcnow_iso)


class SearchHit(BaseModel):
    """A single semantic-search result.

    Attributes:
        chunk_id: Id of the matching chunk.
        document_id: Id of the document the chunk belongs to.
        title: Title of the parent document.
        source: Source of the parent document.
        text: The matching chunk's text.
        score: Similarity score in ``[0, 1]`` (higher is more relevant).
        metadata: Metadata associated with the chunk/document.
    """

    chunk_id: str
    document_id: str
    title: str
    source: str
    text: str
    score: float
    metadata: dict[str, Any] = Field(default_factory=dict)


class DocumentSummary(BaseModel):
    """Lightweight document descriptor used by listing endpoints.

    Attributes:
        document_id: The document id.
        title: Document title.
        source: Document source.
        content_type: Logical content type.
        chunk_count: Number of chunks indexed for this document.
        created_at: ISO-8601 creation timestamp.
    """

    document_id: str
    title: str
    source: str
    content_type: str
    chunk_count: int
    created_at: Optional[str] = None


# ---------------------------------------------------------------------------
# REST / MCP request & response models
# ---------------------------------------------------------------------------
class IngestResponse(BaseModel):
    """Response returned after ingesting one or more documents.

    Attributes:
        document_ids: Ids of the documents that were created.
        chunk_count: Total number of chunks indexed across those documents.
        message: Human-readable summary.
    """

    document_ids: list[str]
    chunk_count: int
    message: str


class NoteRequest(BaseModel):
    """Payload for creating a free-text note in the wiki.

    Attributes:
        title: Title of the note.
        content: Body text of the note.
        metadata: Optional metadata to attach.
    """

    title: str = Field(..., min_length=1)
    content: str = Field(..., min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)


class IngestPathRequest(BaseModel):
    """Payload for ingesting content from the local filesystem.

    Attributes:
        path: Absolute or relative path to a file or directory.
        recursive: When ``path`` is a directory, whether to descend into
            subdirectories.
        metadata: Optional metadata to attach to every ingested document.
    """

    path: str = Field(..., min_length=1)
    recursive: bool = True
    metadata: dict[str, Any] = Field(default_factory=dict)


class SearchResponse(BaseModel):
    """Response returned by the search endpoint / tool.

    Attributes:
        query: The original query string.
        hits: Ranked list of :class:`SearchHit` results.
    """

    query: str
    hits: list[SearchHit]
