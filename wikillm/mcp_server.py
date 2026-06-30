"""MCP server exposing the knowledge base over stdio and HTTP/SSE.

This module builds a single :class:`mcp.server.Server` (protocol revision
``2024-11-05``) that wraps the shared :class:`~wikillm.knowledge_base.KnowledgeBase`
and serves it over two transports:

* **stdio** — :func:`run_stdio` / :func:`main`, used by desktop MCP clients such
  as Claude Desktop. JSON-RPC framing flows over ``stdout``, so logging is kept
  on ``stderr`` (see :mod:`wikillm.logging_setup`).
* **HTTP/SSE** — :func:`build_sse_app` returns a Starlette application mounted by
  the REST server so a single HTTP process serves both interfaces.

Six tools are registered: ``search_wiki``, ``add_note``, ``ingest_path``,
``list_documents``, ``get_document`` and ``delete_document``.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import mcp.types as types
from mcp.server import Server
from mcp.server.sse import SseServerTransport
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Mount, Route

from wikillm.auth import is_request_authorized
from wikillm.config import get_settings
from wikillm.knowledge_base import KnowledgeBase, get_knowledge_base
from wikillm.logging_setup import get_logger, setup_logging

logger = get_logger(__name__)

SERVER_NAME = "wikillm"
SERVER_VERSION = "0.1.0"


# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------
def _tool_definitions() -> list[types.Tool]:
    """Return the static list of MCP tools advertised by the server.

    Returns:
        The tool catalogue with JSON-Schema input definitions.
    """
    return [
        types.Tool(
            name="search_wiki",
            description=(
                "Search the wiki knowledge base for passages relevant to a natural-language "
                "query and return the best-matching chunks with their source and score."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The natural-language search query."},
                    "k": {
                        "type": "integer",
                        "description": "Maximum number of results to return.",
                        "default": 5,
                        "minimum": 1,
                        "maximum": 50,
                    },
                },
                "required": ["query"],
            },
        ),
        types.Tool(
            name="add_note",
            description=(
                "Add a new free-text note to the wiki. The note is chunked, embedded and indexed "
                "so it becomes searchable immediately."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "title": {"type": "string", "description": "Title of the note."},
                    "content": {"type": "string", "description": "Body text of the note."},
                },
                "required": ["title", "content"],
            },
        ),
        types.Tool(
            name="ingest_path",
            description=(
                "Ingest a file or all supported files under a directory from the server's local "
                "filesystem (.txt, .md, .pdf, .docx)."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File or directory path on the server."},
                    "recursive": {
                        "type": "boolean",
                        "description": "Descend into subdirectories when path is a directory.",
                        "default": True,
                    },
                },
                "required": ["path"],
            },
        ),
        types.Tool(
            name="list_documents",
            description="List all documents currently indexed in the wiki, with chunk counts.",
            inputSchema={"type": "object", "properties": {}},
        ),
        types.Tool(
            name="get_document",
            description="Retrieve the full reconstructed text and metadata of a document by its id.",
            inputSchema={
                "type": "object",
                "properties": {
                    "document_id": {"type": "string", "description": "The id of the document to fetch."},
                },
                "required": ["document_id"],
            },
        ),
        types.Tool(
            name="delete_document",
            description="Delete a document and all of its chunks from the wiki by its id.",
            inputSchema={
                "type": "object",
                "properties": {
                    "document_id": {"type": "string", "description": "The id of the document to delete."},
                },
                "required": ["document_id"],
            },
        ),
    ]


def _dispatch_tool(kb: KnowledgeBase, name: str, arguments: dict[str, Any]) -> Any:
    """Execute a tool against the knowledge base and return a JSON-able result.

    Args:
        kb: The shared knowledge base.
        name: The tool name requested by the client.
        arguments: The validated tool arguments.

    Returns:
        A JSON-serialisable object describing the tool's result.

    Raises:
        ValueError: If the tool name is unknown or required arguments missing.
    """
    if name == "search_wiki":
        query = str(arguments["query"])
        k = int(arguments.get("k", 5))
        hits = kb.search(query, k=k)
        return {"query": query, "hits": [hit.model_dump() for hit in hits]}

    if name == "add_note":
        result = kb.add_note(title=str(arguments["title"]), content=str(arguments["content"]))
        return result.model_dump()

    if name == "ingest_path":
        result = kb.ingest_path(
            path=str(arguments["path"]),
            recursive=bool(arguments.get("recursive", True)),
        )
        return result.model_dump()

    if name == "list_documents":
        return {"documents": [doc.model_dump() for doc in kb.list_documents()]}

    if name == "get_document":
        document = kb.get_document(str(arguments["document_id"]))
        if document is None:
            return {"found": False, "document_id": arguments["document_id"]}
        return {"found": True, "document": document.model_dump()}

    if name == "delete_document":
        removed = kb.delete_document(str(arguments["document_id"]))
        return {"document_id": arguments["document_id"], "chunks_removed": removed}

    raise ValueError(f"Unknown tool: {name}")


# ---------------------------------------------------------------------------
# Server construction
# ---------------------------------------------------------------------------
def build_server() -> Server:
    """Build and configure the MCP :class:`Server` with all tool handlers.

    Returns:
        A ready-to-run MCP server bound to the shared knowledge base.
    """
    server: Server = Server(SERVER_NAME, version=SERVER_VERSION)

    @server.list_tools()
    async def list_tools() -> list[types.Tool]:
        """Advertise the available tools to the client."""
        return _tool_definitions()

    @server.call_tool()
    async def call_tool(name: str, arguments: dict[str, Any] | None) -> list[types.TextContent]:
        """Execute a tool call and return its result as text content.

        Tool work is potentially blocking (embedding, disk I/O), so it runs in a
        worker thread to avoid stalling the asyncio event loop.

        Args:
            name: The tool name.
            arguments: The tool arguments (may be ``None``).

        Returns:
            A single text content item carrying the JSON-encoded result.
        """
        arguments = arguments or {}
        kb = get_knowledge_base()
        try:
            result = await asyncio.to_thread(_dispatch_tool, kb, name, arguments)
            payload = json.dumps(result, ensure_ascii=False, indent=2)
            return [types.TextContent(type="text", text=payload)]
        except Exception as exc:
            logger.exception("Tool '%s' failed", name)
            error = json.dumps({"error": str(exc), "tool": name}, ensure_ascii=False)
            return [types.TextContent(type="text", text=error)]

    return server


# ---------------------------------------------------------------------------
# stdio transport
# ---------------------------------------------------------------------------
async def run_stdio() -> None:
    """Run the MCP server over the stdio transport until the client disconnects.

    Uses the official ``stdio_server`` context manager which provides the
    JSON-RPC read/write streams over ``stdin``/``stdout``.
    """
    from mcp.server.stdio import stdio_server

    server = build_server()
    init_options = server.create_initialization_options()
    logger.info("Starting MCP server '%s' on stdio transport.", SERVER_NAME)
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, init_options)


# ---------------------------------------------------------------------------
# HTTP / SSE transport
# ---------------------------------------------------------------------------
class _AuthMiddleware(BaseHTTPMiddleware):
    """Starlette middleware enforcing the optional API key on the MCP mount."""

    async def dispatch(self, request: Request, call_next):  # type: ignore[override]
        """Reject unauthorised requests before they reach the SSE handler."""
        if is_request_authorized({k.decode(): v.decode() for k, v in request.headers.raw}):
            return await call_next(request)
        return JSONResponse({"detail": "Missing or invalid API key."}, status_code=401)


def build_sse_app() -> Starlette:
    """Build the Starlette sub-application implementing the MCP SSE transport.

    The returned app exposes two routes (relative to the mount point chosen by
    the host application): ``/sse`` opens the server-sent events stream, and
    ``/messages/`` receives the client's JSON-RPC messages. The transport learns
    its own mount prefix at request time from the ASGI ``root_path``, so this
    function needs no knowledge of where it will be mounted.

    Returns:
        A configured :class:`starlette.applications.Starlette` instance.
    """
    server = build_server()
    # The endpoint is the path *within* this sub-app (e.g. "/messages/").
    # SseServerTransport.connect_sse prepends the ASGI ``root_path`` (the mount
    # prefix, e.g. "/mcp") when advertising the URL to the client, so this must
    # NOT include ``mount_path`` itself — otherwise the prefix is duplicated.
    sse = SseServerTransport("/messages/")

    async def handle_sse(request: Request) -> Response:
        """Open an SSE stream and run the MCP server over it."""
        async with sse.connect_sse(request.scope, request.receive, request._send) as streams:
            read_stream, write_stream = streams
            await server.run(read_stream, write_stream, server.create_initialization_options())
        # connect_sse fully handles the response lifecycle.
        return Response(status_code=200)

    middleware = [Middleware(_AuthMiddleware)] if get_settings().auth_enabled else []

    return Starlette(
        routes=[
            Route("/sse", endpoint=handle_sse, methods=["GET"]),
            Mount("/messages/", app=sse.handle_post_message),
        ],
        middleware=middleware,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    """Console entry point: configure logging and run the stdio server.

    Logging is initialised to ``stderr``/file so ``stdout`` stays reserved for
    the JSON-RPC framing.
    """
    settings = get_settings()
    setup_logging(level=settings.log_level, log_file=settings.log_file)
    asyncio.run(run_stdio())


if __name__ == "__main__":
    main()
