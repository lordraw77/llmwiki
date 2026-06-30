#!/usr/bin/env python3
"""Entry point: run the Wiki-LLM MCP server over the stdio transport.

This is the script MCP clients (e.g. Claude Desktop) launch. It must keep
``stdout`` exclusively for JSON-RPC framing, so logging is routed to
``stderr``/file by :func:`wikillm.mcp_server.main` before the server starts.

Usage:
    python3.12 run_stdio.py
"""

from __future__ import annotations

from wikillm.mcp_server import main

if __name__ == "__main__":
    main()
