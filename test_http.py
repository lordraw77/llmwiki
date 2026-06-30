#!/usr/bin/env python3
"""Manual integration test for the Wiki-LLM HTTP interface.

This script exercises the running HTTP server end to end:

1. ``GET /health``
2. ``POST /notes`` — add a note
3. ``POST /documents/upload`` — upload an in-memory Markdown file
4. ``GET /search`` — semantic search
5. ``GET /documents`` / ``GET /documents/{id}`` — listing & retrieval
6. ``DELETE /documents/{id}`` — deletion
7. **MCP over HTTP/SSE** — connect an MCP client to ``{base}/mcp/sse``, list the
   tools and call ``search_wiki``.

It is a *client*: start the server first in another terminal with
``python3.12 run_http.py``. Authentication is honoured via ``--api-key`` or the
``API_KEY`` environment variable.

Usage:
    python3.12 run_http.py                 # terminal 1
    python3.12 test_http.py                 # terminal 2
    python3.12 test_http.py --base-url http://localhost:8000 --api-key secret

Exit code is ``0`` when every check passes, ``1`` otherwise.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from typing import Optional

import httpx


def _auth_headers(api_key: Optional[str]) -> dict[str, str]:
    """Return the auth headers for a request, empty when no key is configured.

    Args:
        api_key: The API key, or ``None`` when authentication is disabled.

    Returns:
        A headers dict suitable for ``httpx``.
    """
    return {"Authorization": f"Bearer {api_key}"} if api_key else {}


def _check(label: str, condition: bool, detail: str = "") -> bool:
    """Print a PASS/FAIL line for a single assertion.

    Args:
        label: Human-readable name of the check.
        condition: Whether the check passed.
        detail: Optional extra context appended to the line.

    Returns:
        The value of ``condition`` (so callers can accumulate failures).
    """
    status = "PASS" if condition else "FAIL"
    suffix = f" — {detail}" if detail else ""
    print(f"[{status}] {label}{suffix}")
    return condition


def run_rest_checks(base_url: str, api_key: Optional[str]) -> bool:
    """Run the REST endpoint checks against the server.

    Args:
        base_url: Base URL of the running server (no trailing slash).
        api_key: Optional API key for authenticated requests.

    Returns:
        ``True`` if all REST checks pass.
    """
    headers = _auth_headers(api_key)
    ok = True

    with httpx.Client(base_url=base_url, headers=headers, timeout=60.0) as client:
        # 1. Health
        resp = client.get("/health")
        ok &= _check("GET /health", resp.status_code == 200 and resp.json().get("status") == "ok")

        # 2. Add a note
        resp = client.post(
            "/notes",
            json={
                "title": "HTTP Test Note",
                "content": (
                    "The Wiki-LLM HTTP test note. It mentions a unique token "
                    "PINEAPPLE_42 used to verify search relevance."
                ),
            },
        )
        ok &= _check("POST /notes", resp.status_code == 200 and resp.json()["chunk_count"] >= 1)

        # 3. Upload an in-memory Markdown document
        md = b"# Upload Test\n\nThis uploaded markdown talks about ZEBRA_99 satellites."
        resp = client.post(
            "/documents/upload",
            files={"file": ("upload_test.md", md, "text/markdown")},
        )
        ok &= _check("POST /documents/upload", resp.status_code == 200 and resp.json()["chunk_count"] >= 1)

        # 4. Search for the unique token
        resp = client.get("/search", params={"q": "PINEAPPLE_42 unique token", "k": 5})
        hits = resp.json().get("hits", []) if resp.status_code == 200 else []
        ok &= _check("GET /search", resp.status_code == 200 and len(hits) >= 1,
                     detail=f"{len(hits)} hit(s)")

        # 5. List documents + fetch the first one
        resp = client.get("/documents")
        documents = resp.json() if resp.status_code == 200 else []
        ok &= _check("GET /documents", resp.status_code == 200 and len(documents) >= 2,
                     detail=f"{len(documents)} document(s)")

        document_id = documents[0]["document_id"] if documents else None
        if document_id:
            resp = client.get(f"/documents/{document_id}")
            ok &= _check("GET /documents/{id}", resp.status_code == 200 and "text" in resp.json())

            # 6. Delete it, then confirm it is gone
            resp = client.delete(f"/documents/{document_id}")
            ok &= _check("DELETE /documents/{id}", resp.status_code == 200)
            resp = client.get(f"/documents/{document_id}")
            ok &= _check("GET deleted -> 404", resp.status_code == 404)

    return ok


async def run_mcp_sse_check(base_url: str, api_key: Optional[str]) -> bool:
    """Connect to the MCP HTTP/SSE transport and exercise a tool.

    Args:
        base_url: Base URL of the running server.
        api_key: Optional API key (sent as a Bearer header on the SSE request).

    Returns:
        ``True`` if the MCP/SSE round-trip succeeds.
    """
    try:
        from mcp.client.session import ClientSession
        from mcp.client.sse import sse_client
    except ImportError:
        print("[SKIP] MCP SSE check — 'mcp' client not importable")
        return True

    sse_url = f"{base_url}/mcp/sse"
    headers = _auth_headers(api_key)
    ok = True

    try:
        async with sse_client(sse_url, headers=headers) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                init = await session.initialize()
                ok &= _check("MCP/SSE initialize", bool(init.serverInfo.name),
                             detail=f"server={init.serverInfo.name}, proto={init.protocolVersion}")

                tools = await session.list_tools()
                tool_names = sorted(tool.name for tool in tools.tools)
                ok &= _check("MCP/SSE tools/list", "search_wiki" in tool_names,
                             detail=", ".join(tool_names))

                result = await session.call_tool("search_wiki", {"query": "ZEBRA_99", "k": 3})
                text = result.content[0].text if result.content else ""
                ok &= _check("MCP/SSE call search_wiki", "hits" in text)
    except Exception as exc:  # noqa: BLE001 - report any transport failure
        ok &= _check("MCP/SSE connection", False, detail=str(exc))

    return ok


def main() -> int:
    """Parse arguments, run all checks and return a process exit code."""
    parser = argparse.ArgumentParser(description="Integration test for the Wiki-LLM HTTP interface.")
    parser.add_argument(
        "--base-url",
        default=os.environ.get("WIKILLM_BASE_URL", "http://localhost:8000"),
        help="Base URL of the running server (default: http://localhost:8000).",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("API_KEY"),
        help="API key, if the server has authentication enabled.",
    )
    parser.add_argument(
        "--skip-mcp",
        action="store_true",
        help="Only run the REST checks; skip the MCP/SSE round-trip.",
    )
    args = parser.parse_args()
    base_url = args.base_url.rstrip("/")

    print(f"== Wiki-LLM HTTP test against {base_url} ==")
    try:
        rest_ok = run_rest_checks(base_url, args.api_key)
    except httpx.ConnectError:
        print(f"[FAIL] Could not connect to {base_url}. Is 'python3.12 run_http.py' running?")
        return 1

    mcp_ok = True
    if not args.skip_mcp:
        mcp_ok = asyncio.run(run_mcp_sse_check(base_url, args.api_key))

    print("-" * 40)
    if rest_ok and mcp_ok:
        print("RESULT: all checks passed ✅")
        return 0
    print("RESULT: some checks FAILED ❌")
    return 1


if __name__ == "__main__":
    sys.exit(main())
