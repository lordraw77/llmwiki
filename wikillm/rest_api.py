"""FastAPI application exposing the REST interface and the MCP HTTP/SSE mount.

A single process serves everything:

* REST endpoints for uploading files, ingesting from the filesystem, adding
  notes, searching and managing documents.
* The MCP server over HTTP/SSE, mounted at ``MCP_SSE_PATH`` (default ``/mcp``)
  via :func:`wikillm.mcp_server.build_sse_app`.

Every REST endpoint is guarded by the optional :func:`wikillm.auth.require_auth`
dependency (a no-op when ``API_KEY`` is unset).
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Optional

import uvicorn
from fastapi import Depends, FastAPI, File, HTTPException, Query, UploadFile, status

from wikillm.auth import require_auth
from wikillm.config import get_settings
from wikillm.ingestion import ParseError, UnsupportedFormatError
from wikillm.knowledge_base import get_knowledge_base
from wikillm.logging_setup import get_logger, setup_logging
from wikillm.mcp_server import build_sse_app
from wikillm.models import (
    DocumentSummary,
    IngestPathRequest,
    IngestResponse,
    NoteRequest,
    ParsedDocument,
    SearchResponse,
)

logger = get_logger(__name__)


@asynccontextmanager
async def _lifespan(app: FastAPI):
    """Initialise the knowledge base eagerly at startup.

    Building the embedder + vector store on the first request would make that
    request slow (and could surface configuration errors late). Doing it during
    startup fails fast and warms the model.
    """
    settings = get_settings()
    logger.info(
        "Starting Wiki-LLM HTTP server (backend=%s, auth=%s).",
        settings.vector_backend,
        "on" if settings.auth_enabled else "off",
    )
    get_knowledge_base()
    yield
    logger.info("Wiki-LLM HTTP server shutting down.")


def create_app() -> FastAPI:
    """Construct and configure the FastAPI application.

    Returns:
        The fully wired application, including the mounted MCP SSE sub-app.
    """
    settings = get_settings()
    app = FastAPI(
        title="Wiki-LLM",
        version="0.1.0",
        description="Multi-format knowledge base with REST and MCP (stdio + HTTP/SSE) interfaces.",
        lifespan=_lifespan,
    )

    # -- Health ------------------------------------------------------------
    @app.get("/health", tags=["system"])
    async def health() -> dict:
        """Liveness probe with basic knowledge-base statistics."""
        return {"status": "ok", **get_knowledge_base().stats()}

    # -- Ingestion: file upload -------------------------------------------
    @app.post(
        "/documents/upload",
        response_model=IngestResponse,
        tags=["ingestion"],
        dependencies=[Depends(require_auth)],
    )
    async def upload_document(file: UploadFile = File(...)) -> IngestResponse:
        """Upload and ingest a single document.

        The logical type is auto-detected from the filename/MIME type and routed
        to the matching parser (``.txt``/``.md``/``.pdf``/``.docx``).

        Args:
            file: The multipart file upload.

        Returns:
            The ingestion result.

        Raises:
            HTTPException: ``415`` for unsupported formats, ``422`` for parse
                failures, ``400`` for empty uploads.
        """
        data = await file.read()
        if not data:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, "Uploaded file is empty.")
        try:
            return get_knowledge_base().ingest_bytes(
                data, filename=file.filename or "upload", mime_type=file.content_type
            )
        except UnsupportedFormatError as exc:
            raise HTTPException(status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, str(exc)) from exc
        except ParseError as exc:
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(exc)) from exc

    # -- Ingestion: filesystem --------------------------------------------
    @app.post(
        "/documents/ingest-path",
        response_model=IngestResponse,
        tags=["ingestion"],
        dependencies=[Depends(require_auth)],
    )
    async def ingest_path(request: IngestPathRequest) -> IngestResponse:
        """Ingest a file or a directory tree from the server's filesystem.

        Args:
            request: The path, recursion flag and optional metadata.

        Returns:
            The aggregate ingestion result.

        Raises:
            HTTPException: ``404`` if the path does not exist.
        """
        try:
            return get_knowledge_base().ingest_path(
                request.path, recursive=request.recursive, metadata=request.metadata
            )
        except FileNotFoundError as exc:
            raise HTTPException(status.HTTP_404_NOT_FOUND, str(exc)) from exc

    # -- Notes -------------------------------------------------------------
    @app.post(
        "/notes",
        response_model=IngestResponse,
        tags=["ingestion"],
        dependencies=[Depends(require_auth)],
    )
    async def add_note(request: NoteRequest) -> IngestResponse:
        """Create a free-text note and index it.

        Args:
            request: The note title, content and optional metadata.

        Returns:
            The ingestion result.
        """
        return get_knowledge_base().add_note(
            title=request.title, content=request.content, metadata=request.metadata
        )

    # -- Search ------------------------------------------------------------
    @app.get(
        "/search",
        response_model=SearchResponse,
        tags=["search"],
        dependencies=[Depends(require_auth)],
    )
    async def search(
        q: str = Query(..., min_length=1, description="The search query."),
        k: int = Query(5, ge=1, le=50, description="Maximum number of results."),
    ) -> SearchResponse:
        """Run a semantic search over the indexed corpus.

        Args:
            q: The query string.
            k: Maximum number of results.

        Returns:
            The ranked search results.
        """
        hits = get_knowledge_base().search(q, k=k)
        return SearchResponse(query=q, hits=hits)

    # -- Document management ----------------------------------------------
    @app.get(
        "/documents",
        response_model=list[DocumentSummary],
        tags=["documents"],
        dependencies=[Depends(require_auth)],
    )
    async def list_documents() -> list[DocumentSummary]:
        """List every indexed document with its chunk count."""
        return get_knowledge_base().list_documents()

    @app.get(
        "/documents/{document_id}",
        response_model=ParsedDocument,
        tags=["documents"],
        dependencies=[Depends(require_auth)],
    )
    async def get_document(document_id: str) -> ParsedDocument:
        """Fetch the reconstructed text and metadata of a document.

        Args:
            document_id: The document id.

        Returns:
            The reconstructed document.

        Raises:
            HTTPException: ``404`` if the document is not indexed.
        """
        document = get_knowledge_base().get_document(document_id)
        if document is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, f"Document '{document_id}' not found.")
        return document

    @app.delete(
        "/documents/{document_id}",
        tags=["documents"],
        dependencies=[Depends(require_auth)],
    )
    async def delete_document(document_id: str) -> dict:
        """Delete a document and all of its chunks.

        Args:
            document_id: The document id.

        Returns:
            The number of chunks removed.

        Raises:
            HTTPException: ``404`` if the document is not indexed.
        """
        removed = get_knowledge_base().delete_document(document_id)
        if removed == 0:
            raise HTTPException(status.HTTP_404_NOT_FOUND, f"Document '{document_id}' not found.")
        return {"document_id": document_id, "chunks_removed": removed}

    # -- MCP HTTP/SSE mount ------------------------------------------------
    # Mounted last so its routes live under the configured prefix (e.g. /mcp):
    #   GET  {prefix}/sse        -> SSE stream
    #   POST {prefix}/messages/  -> client JSON-RPC messages
    app.mount(settings.mcp_sse_path, build_sse_app())
    logger.info("MCP SSE transport mounted at '%s/sse'.", settings.mcp_sse_path)

    return app


# Module-level ASGI app for `uvicorn wikillm.rest_api:app`.
app = create_app()


def main() -> None:
    """Console entry point: run the HTTP server with uvicorn."""
    settings = get_settings()
    setup_logging(level=settings.log_level, log_file=settings.log_file)
    uvicorn.run(
        "wikillm.rest_api:app",
        host=settings.app_host,
        port=settings.app_port,
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    main()
