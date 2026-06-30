"""Wiki-LLM — a multi-format knowledge base with REST and MCP interfaces.

This package implements a small but production-shaped "Wiki-LLM" service:

* **Ingestion** of ``.txt``, ``.md``, ``.pdf`` and ``.docx`` documents, either
  via REST upload or directly from the filesystem.
* **Storage & semantic search** through a pluggable vector store
  (ChromaDB or Qdrant) combined with a pluggable embedding provider
  (local ``sentence-transformers`` or remote OpenAI-compatible APIs, with an
  optional round-robin + failover rotation across multiple providers/keys).
* **Two interfaces** over the same in-process :class:`~wikillm.knowledge_base.KnowledgeBase`:
  a FastAPI REST API and a Model Context Protocol (MCP) server exposed over both
  ``stdio`` and HTTP/SSE transports (protocol revision ``2024-11-05``).

The public entry points are:

* :func:`wikillm.rest_api.main` — run the HTTP server (REST + MCP/SSE).
* :func:`wikillm.mcp_server.main` — run the MCP server over stdio.
"""

from __future__ import annotations

__all__ = ["__version__"]

__version__ = "0.1.0"
