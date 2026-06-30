#!/usr/bin/env python3
"""Entry point: run the Wiki-LLM HTTP server (REST API + MCP HTTP/SSE).

Starts a single uvicorn process that serves both the REST endpoints and the MCP
SSE transport mounted at ``MCP_SSE_PATH``.

Usage:
    python3.12 run_http.py
"""

from __future__ import annotations

from wikillm.rest_api import main

if __name__ == "__main__":
    main()
