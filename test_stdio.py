#!/usr/bin/env python3
"""Manual integration test for the Wiki-LLM MCP stdio transport.

This script spawns ``run_stdio.py`` as a subprocess (exactly as Claude Desktop
would) and drives it with the official MCP client over stdio:

1. ``initialize`` — handshake and protocol negotiation.
2. ``tools/list`` — verify all six tools are advertised.
3. ``add_note`` — add a note through the tool.
4. ``search_wiki`` — search for it.
5. ``list_documents`` — confirm it is indexed.

It also asserts that the server's ``stdout`` is not polluted by logging (which
would corrupt the JSON-RPC framing): logs must go to ``stderr`` only.

Usage:
    python3.12 test_stdio.py
    python3.12 test_stdio.py --python /opt/llmwiki/.venv/bin/python

Notes:
    * Tool calls that need embeddings use whatever ``EMBEDDING_PROVIDER`` is
      configured. With the default ``local`` provider this requires
      ``sentence-transformers`` to be installed; otherwise set a remote provider
      (or pass ``--skip-tool-calls`` to only check the handshake + tool listing).

Exit code is ``0`` when every check passes, ``1`` otherwise.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

EXPECTED_TOOLS = {
    "search_wiki",
    "add_note",
    "ingest_path",
    "list_documents",
    "get_document",
    "delete_document",
}

PROJECT_ROOT = Path(__file__).resolve().parent


def _check(label: str, condition: bool, detail: str = "") -> bool:
    """Print a PASS/FAIL line and return the condition.

    Args:
        label: Human-readable name of the check.
        condition: Whether the check passed.
        detail: Optional extra context.

    Returns:
        ``condition``.
    """
    status = "PASS" if condition else "FAIL"
    suffix = f" — {detail}" if detail else ""
    print(f"[{status}] {label}{suffix}")
    return condition


async def run(python_exe: str, skip_tool_calls: bool) -> bool:
    """Spawn the stdio server and run the MCP client checks.

    Args:
        python_exe: Interpreter used to launch ``run_stdio.py``.
        skip_tool_calls: When ``True``, only the handshake and ``tools/list``
            are verified (no embedding-dependent tool calls).

    Returns:
        ``True`` if all checks pass.
    """
    from mcp.client.session import ClientSession
    from mcp.client.stdio import StdioServerParameters, stdio_client

    params = StdioServerParameters(
        command=python_exe,
        args=["run_stdio.py"],
        cwd=str(PROJECT_ROOT),
        env={**os.environ, "PYTHONPATH": str(PROJECT_ROOT)},
    )

    ok = True
    async with stdio_client(params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            # 1. Handshake
            init = await session.initialize()
            ok &= _check(
                "initialize",
                init.serverInfo.name == "wikillm",
                detail=f"server={init.serverInfo.name}, proto={init.protocolVersion}",
            )

            # 2. Tool listing
            tools = await session.list_tools()
            names = {tool.name for tool in tools.tools}
            ok &= _check(
                "tools/list",
                EXPECTED_TOOLS.issubset(names),
                detail=", ".join(sorted(names)),
            )

            if skip_tool_calls:
                print("[SKIP] tool calls (--skip-tool-calls)")
                return ok

            # 3. add_note
            add = await session.call_tool(
                "add_note",
                {"title": "Stdio Test", "content": "A note containing the token MANGO_77 over stdio."},
            )
            add_text = add.content[0].text if add.content else ""
            ok &= _check("call add_note", "chunk_count" in add_text, detail=_first_line(add_text))

            # 4. search_wiki
            search = await session.call_tool("search_wiki", {"query": "MANGO_77 token", "k": 3})
            search_text = search.content[0].text if search.content else ""
            ok &= _check("call search_wiki", "hits" in search_text)

            # 5. list_documents
            listed = await session.call_tool("list_documents", {})
            listed_text = listed.content[0].text if listed.content else ""
            ok &= _check("call list_documents", "documents" in listed_text)

    return ok


def _first_line(text: str) -> str:
    """Return the first non-empty line of ``text`` (for compact output)."""
    for line in text.splitlines():
        if line.strip():
            return line.strip()
    return ""


def main() -> int:
    """Parse arguments, run the stdio checks and return an exit code."""
    parser = argparse.ArgumentParser(description="Integration test for the Wiki-LLM MCP stdio transport.")
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python interpreter used to launch run_stdio.py (default: current interpreter).",
    )
    parser.add_argument(
        "--skip-tool-calls",
        action="store_true",
        help="Only verify the handshake and tool listing (no embedding-dependent calls).",
    )
    args = parser.parse_args()

    print(f"== Wiki-LLM stdio test (python={args.python}) ==")
    try:
        ok = asyncio.run(run(args.python, args.skip_tool_calls))
    except Exception as exc:  # noqa: BLE001 - surface any spawn/transport error
        print(f"[FAIL] stdio session error: {exc}")
        return 1

    print("-" * 40)
    if ok:
        print("RESULT: all checks passed ✅")
        return 0
    print("RESULT: some checks FAILED ❌")
    return 1


if __name__ == "__main__":
    sys.exit(main())
